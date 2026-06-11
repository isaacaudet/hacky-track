"""HTTP end-to-end tests for dense_review_app against real batch rows.

Copies the real release batch template into a temp dir (so reviewed labels on
disk are never touched) and exercises the actual ThreadingHTTPServer +
DenseReviewHandler over real sockets.
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dense_review_app import (
    DenseReviewHandler,
    DenseReviewStore,
    TEMPLATE_NAME,
    read_jsonl,
)

REAL_BATCH = (
    Path(__file__).resolve().parents[1]
    / "runs/release-27-public/dense_trajectory_review_v2_with_sources"
)


@unittest.skipUnless(REAL_BATCH.exists(), "real review batch not present")
class DenseReviewHttpE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = TemporaryDirectory()
        batch_dir = Path(cls.tmp.name)
        rows = read_jsonl(REAL_BATCH / TEMPLATE_NAME)
        if not rows:
            raise AssertionError("real template is empty")
        # First clip only keeps the test fast while staying on real schema.
        first_clip = rows[0]["clip_id"]
        cls.rows = [r for r in rows if r["clip_id"] == first_clip]
        template = batch_dir / TEMPLATE_NAME
        template.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in cls.rows) + "\n",
            encoding="utf-8",
        )
        cls.store = DenseReviewStore(batch_dir)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), DenseReviewHandler)
        cls.server.store = cls.store  # type: ignore[attr-defined]
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def request(self, path: str, payload: dict | None = None) -> tuple[int, dict]:
        req = urllib.request.Request(self.base + path)
        if payload is not None:
            req.data = json.dumps(payload).encode("utf-8")
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read().decode("utf-8"))

    def key(self, row: dict) -> str:
        return f"{row['clip_id']}::{row['frame_index']}"

    def test_01_state_serves_real_rows(self) -> None:
        status, state = self.request("/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(len(state["rows"]), len(self.rows))
        self.assertEqual(state["rows"][0]["clip_id"], self.rows[0]["clip_id"])

    def test_02_root_serves_app_html(self) -> None:
        with urllib.request.urlopen(self.base + "/", timeout=10) as resp:
            body = resp.read().decode("utf-8")
        self.assertIn("acceptSpanBtn", body)
        self.assertIn("/api/label_bulk", body)

    def test_03_bulk_span_accept_over_http(self) -> None:
        span = self.rows[:4]
        items = [
            {
                "key": self.key(r),
                "patch": {
                    "x": 100.0 + i,
                    "y": 200.0 + i,
                    "visibility": "visible",
                    "occlusion": "none",
                    "quality": "reviewed",
                    "label_source": "human_dense_review+cotracker_span",
                },
            }
            for i, r in enumerate(span)
        ]
        status, body = self.request("/api/label_bulk", {"items": items})
        self.assertEqual(status, 200, body)
        self.assertEqual(len(body["rows"]), 4)
        for row in body["rows"]:
            self.assertEqual(row["quality"], "reviewed")
        # Persisted to the temp batch dir.
        saved = read_jsonl(self.store.labels_out)
        reviewed = {r["frame_index"] for r in saved if r["quality"] == "reviewed"}
        self.assertTrue(all(r["frame_index"] in reviewed for r in span))

    def test_04_bulk_atomic_failure_over_http(self) -> None:
        good_key = self.key(self.rows[-1])
        items = [
            {"key": good_key, "patch": {"quality": "reviewed", "visibility": "visible",
                                        "x": 1.0, "y": 2.0, "occlusion": "none"}},
            {"key": "no-such-clip::1", "patch": {"quality": "reviewed"}},
        ]
        status, body = self.request("/api/label_bulk", {"items": items})
        self.assertEqual(status, 400)
        self.assertIn("error", body)
        # The good row must not have been applied.
        status, state = self.request("/api/state")
        row = next(r for r in state["rows"] if self.key(r) == good_key)
        self.assertNotEqual(row["quality"], "reviewed")

    def test_05_bulk_rejects_locked_fields_over_http(self) -> None:
        items = [{"key": self.key(self.rows[5]), "patch": {"split": "test"}}]
        status, body = self.request("/api/label_bulk", {"items": items})
        self.assertEqual(status, 400)
        self.assertIn("locked", body["error"])

    def test_06_bulk_requires_items(self) -> None:
        status, body = self.request("/api/label_bulk", {"items": []})
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
