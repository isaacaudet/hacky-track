"""Tests for the dense review app store, including the bulk span-accept path."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dense_review_app import DenseReviewStore, TEMPLATE_NAME, read_jsonl


def make_template_rows(clip_id: str = "clip-a", count: int = 6) -> list[dict]:
    rows = []
    for i in range(count):
        rows.append(
            {
                "clip_id": clip_id,
                "frame_index": 100 + i,
                "frame_image": f"clips/{clip_id}/frames/frame_{100 + i:06d}.jpg",
                "time_sec": (100 + i) / 30.0,
                "source_video": "video-test.MOV",
                "video_path": "video-test.MOV",
                "split": "train",
                "training_use": "train_or_calibration",
                "x": None,
                "y": None,
                "radius": 10.0,
                "visibility": "unlabeled",
                "occlusion": "unknown",
                "quality": "pending",
                "label_source": "human_dense_review",
                "model_hints": {},
                "schema_version": 1,
            }
        )
    return rows


class DenseReviewStoreBulkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.batch_dir = Path(self.tmp.name)
        rows = make_template_rows()
        template = self.batch_dir / TEMPLATE_NAME
        template.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        self.store = DenseReviewStore(self.batch_dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def key(self, frame_index: int) -> str:
        return f"clip-a::{frame_index}"

    def span_items(self, frames: list[int]) -> list[dict]:
        return [
            {
                "key": self.key(f),
                "patch": {
                    "x": 10.0 + f,
                    "y": 20.0 + f,
                    "visibility": "visible",
                    "occlusion": "none",
                    "quality": "reviewed",
                    "label_source": "human_dense_review+cotracker_span",
                },
            }
            for f in frames
        ]

    def test_bulk_accept_span_marks_rows_reviewed(self) -> None:
        applied = self.store.update_labels_bulk(self.span_items([100, 101, 102]))
        self.assertEqual(len(applied), 3)
        for row in applied:
            self.assertEqual(row["quality"], "reviewed")
            self.assertEqual(row["visibility"], "visible")
            self.assertEqual(row["label_source"], "human_dense_review+cotracker_span")
            self.assertIsNotNone(row["x"])
            self.assertIn("reviewed_at", row)
        # Persisted once, readable back.
        saved = read_jsonl(self.store.labels_out)
        reviewed = [r for r in saved if r["quality"] == "reviewed"]
        self.assertEqual(len(reviewed), 3)

    def test_bulk_is_atomic_on_failure(self) -> None:
        items = self.span_items([100, 101])
        items.append({"key": "clip-a::999", "patch": {"quality": "reviewed"}})
        with self.assertRaises(KeyError):
            self.store.update_labels_bulk(items)
        # Nothing applied in memory or on disk.
        self.assertTrue(all(row["quality"] == "pending" for row in self.store.rows))
        self.assertFalse(self.store.labels_out.exists())

    def test_bulk_rejects_locked_fields(self) -> None:
        items = [{"key": self.key(100), "patch": {"split": "test"}}]
        with self.assertRaises(ValueError):
            self.store.update_labels_bulk(items)
        self.assertEqual(self.store.rows[0]["split"], "train")

    def test_bulk_uncertain_span_stays_pending_and_untrainable(self) -> None:
        items = [
            {
                "key": self.key(f),
                "patch": {
                    "visibility": "uncertain",
                    "occlusion": "unknown",
                    "quality": "pending",
                    "label_source": "human_dense_review+cotracker_lowvis",
                },
            }
            for f in [103, 104]
        ]
        applied = self.store.update_labels_bulk(items)
        for row in applied:
            self.assertEqual(row["visibility"], "uncertain")
            self.assertEqual(row["quality"], "pending")

    def test_single_update_still_works(self) -> None:
        row = self.store.update_label(
            self.key(105),
            {"x": 5.0, "y": 6.0, "visibility": "visible", "occlusion": "none", "quality": "reviewed"},
        )
        self.assertEqual(row["quality"], "reviewed")
        self.assertEqual(row["x"], 5.0)

    def test_bulk_does_not_mutate_audit_lock(self) -> None:
        # training_use must survive any bulk patch untouched.
        applied = self.store.update_labels_bulk(self.span_items([100]))
        self.assertEqual(applied[0]["training_use"], "train_or_calibration")


if __name__ == "__main__":
    unittest.main()
