from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_review_batch
import review_app


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class ActiveLearningReviewBatchTests(unittest.TestCase):
    def test_suppressed_events_become_active_learning_records(self) -> None:
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
                            "qa_ball_x": 100,
                            "qa_ball_y": 200,
                            "contact_type": "foot",
                        }
                    ],
                    "suppressed_events": [
                        {
                            "type": "touch",
                            "time_sec": 1.55,
                            "qa_rally_id": 1,
                            "qa_ball_x": 120,
                            "qa_ball_y": 220,
                            "contact_type": "foot_candidate",
                            "qa_suppressed_reason": "low_confidence_touch_candidate",
                        },
                        {
                            "type": "drop_floor",
                            "time_sec": 2.2,
                            "qa_rally_id": 1,
                            "qa_ball_x": 130,
                            "qa_ball_y": 880,
                            "contact_type": "ground",
                            "qa_suppressed_reason": "floor_reset_too_far_from_limb_context",
                        },
                    ],
                },
            )
            manifest = root / "qa" / "qa_manifest.json"
            write_json(
                manifest,
                {
                    "runs": [
                        {
                            "video": str(root / "clip.MOV"),
                            "qa_events_path": str(qa_events),
                        }
                    ]
                },
            )
            records = build_review_batch.collect_records(manifest)
            active = [record for record in records if record.get("source") == "active_learning"]
            self.assertEqual(len(active), 2)
            self.assertTrue(all(record.get("proposed_status") == "missing" for record in active))
            self.assertTrue(any("likely_missed_touch" in record["review_tags"] for record in active))
            selected = build_review_batch.select_records(records, max_items=4, per_video=2)
            self.assertTrue(any(record.get("source") == "active_learning" for record in selected))

    def test_active_learning_record_becomes_pending_manual_missing_proposal(self) -> None:
        record = {
            "source": "active_learning",
            "review_item_id": "miss-r001-touch-0001550",
            "kind": "touch",
            "time_sec": 1.55,
            "rally_id": 1,
            "qa_ball_x": 120,
            "qa_ball_y": 220,
            "contact_side": "right",
            "contact_type": "foot_candidate",
            "review_tags": ["active_learning", "likely_missed_touch"],
        }
        item = review_app.review_item_from_active_learning_record(record)
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item["source"], "manual")
        self.assertEqual(item["status"], "pending")
        self.assertEqual(item["proposed_status"], "missing")
        self.assertTrue(item["active_learning_candidate"])
        self.assertEqual(item["x"], 120)
        self.assertIn("likely_missed_touch", item["review_tags"])

    def test_gap_without_floor_reset_becomes_missing_drop_proposal(self) -> None:
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
                            "qa_ball_x": 300,
                            "qa_ball_y": 740,
                            "contact_type": "foot",
                        },
                        {
                            "type": "touch",
                            "time_sec": 5.0,
                            "qa_rally_id": 2,
                            "qa_touch_index": 1,
                            "qa_ball_x": 320,
                            "qa_ball_y": 880,
                            "contact_type": "foot",
                        },
                    ],
                    "rallies": [
                        {
                            "id": 1,
                            "end_sec": 1.0,
                            "touches": 6,
                            "quality_score": 80.0,
                            "ended_by_gap_without_floor_reset": True,
                            "next_contact_gap_sec": 4.0,
                        }
                    ],
                },
            )
            manifest = root / "qa" / "qa_manifest.json"
            write_json(manifest, {"runs": [{"video": str(root / "clip.MOV"), "qa_events_path": str(qa_events)}]})

            records = build_review_batch.collect_records(manifest)
            gap_records = [
                record
                for record in records
                if record.get("source") == "active_learning" and record.get("drop_source") == "gap_without_floor_reset"
            ]
            self.assertEqual(len(gap_records), 1)
            record = gap_records[0]
            self.assertEqual(record["kind"], "drop_floor")
            self.assertEqual(record["proposed_status"], "missing")
            self.assertIn("likely_missing_floor_reset", record["review_tags"])
            self.assertIn("strict_best_rally_blocker", record["review_tags"])


if __name__ == "__main__":
    unittest.main()
