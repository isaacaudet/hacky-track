from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import apply_review_decisions
import seed_review_batch


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class ReviewBatchIoTests(unittest.TestCase):
    def test_seed_batch_writes_to_requested_reviews_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_events = root / "qa" / "clip" / "qa_events.json"
            write_json(
                qa_events,
                {
                    "source_video": "clip.MOV",
                    "events": [
                        {
                            "type": "touch",
                            "time_sec": 1.0,
                            "qa_rally_id": 1,
                            "qa_touch_index": 1,
                            "contact_side": "right",
                            "contact_type": "foot",
                        }
                    ],
                    "rallies": [],
                },
            )
            manifest = root / "qa_manifest.json"
            write_json(manifest, {"runs": [{"video": "clip.MOV", "qa_events_path": str(qa_events)}]})
            batch = root / "latest_review_batch.json"
            write_json(
                batch,
                {
                    "batch_id": "batch-1",
                    "items": [
                        {
                            "review_stem": "clip",
                            "review_item_id": "cand-r001-t001-0001000",
                            "batch_item_id": "clip__cand-r001-t001-0001000",
                            "time_sec": 1.0,
                            "priority_score": 3.0,
                        },
                        {
                            "review_stem": "clip",
                            "source": "active_learning",
                            "review_item_id": "miss-r001-dropgap-0001800",
                            "batch_item_id": "clip__miss-r001-dropgap-0001800",
                            "kind": "drop_floor",
                            "time_sec": 1.8,
                            "contact_type": "ground",
                            "priority_score": 5.0,
                        }
                    ],
                },
            )
            assisted = root / "assisted_review_suggestions.json"
            write_json(assisted, {"items": []})
            reviews = root / "run_reviews"

            summary = seed_review_batch.seed_batch(batch, qa_manifest=manifest, reviews_dir=reviews, assisted_review=assisted)

            self.assertEqual(summary["review_files_written"], 1)
            self.assertEqual(summary["seeded_pending_items"], 2)
            review_doc = json.loads((reviews / "clip.review.json").read_text(encoding="utf-8"))
            self.assertEqual(review_doc["items"][0]["id"], "cand-r001-t001-0001000")
            self.assertTrue(review_doc["items"][0]["in_review_batch"])
            self.assertEqual(review_doc["items"][1]["id"], "miss-r001-dropgap-0001800")
            self.assertTrue(review_doc["items"][1]["active_learning_candidate"])

    def test_apply_decisions_uses_requested_reviews_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviews = root / "run_reviews"
            write_json(
                reviews / "clip.review.json",
                {
                    "source_video": "clip.MOV",
                    "items": [
                        {
                            "id": "cand-1",
                            "source": "candidate",
                            "kind": "touch",
                            "status": "pending",
                            "contact_side": "unknown",
                            "contact_type": "foot",
                        }
                    ],
                },
            )
            decisions = root / "decisions.json"
            write_json(
                decisions,
                {
                    "reviewed_at": "2026-05-16T00:00:00Z",
                    "decisions": [
                        {
                            "review_stem": "clip",
                            "item_id": "cand-1",
                            "status": "approved",
                            "time_sec": 1.04,
                            "x": 320,
                            "y": 820,
                            "ball_accuracy": "reviewed",
                            "contact_side": "right",
                            "evidence": "visible right-foot contact in review sheet",
                        }
                    ],
                },
            )

            summary = apply_review_decisions.apply_decisions(decisions, reviews_dir=reviews)

            self.assertEqual(summary["applied"], 1)
            review_doc = json.loads((reviews / "clip.review.json").read_text(encoding="utf-8"))
            item = review_doc["items"][0]
            self.assertEqual(item["status"], "approved")
            self.assertEqual(item["time_sec"], 1.04)
            self.assertEqual(item["x"], 320)
            self.assertEqual(item["ball_accuracy"], "reviewed")
            self.assertEqual(item["contact_side"], "right")
            self.assertEqual(item["review_decision_source"], str(decisions))


if __name__ == "__main__":
    unittest.main()
