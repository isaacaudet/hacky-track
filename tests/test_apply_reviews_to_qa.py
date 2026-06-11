from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import apply_reviews_to_qa
import strict_rally_audit


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class ApplyReviewsToQaTests(unittest.TestCase):
    def test_missing_drop_review_is_inserted_and_resegments_rallies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_events = root / "qa" / "clip" / "qa_events.json"
            events = [
                {
                    "type": "touch",
                    "time_sec": float(idx),
                    "qa_rally_id": 1,
                    "qa_touch_index": idx + 1,
                    "qa_ball_accuracy": "high",
                    "contact_type": "foot",
                    "contact_side": "right",
                }
                for idx in range(6)
            ]
            write_json(
                qa_events,
                {
                    "source_video": "clip.MOV",
                    "events": events,
                    "rallies": [
                        {
                            "id": 1,
                            "start_sec": 0.0,
                            "end_sec": 5.0,
                            "touches": 6,
                            "quality_score": 88.0,
                            "ended_by_gap_without_floor_reset": True,
                            "next_contact_gap_sec": 3.0,
                        }
                    ],
                },
            )
            manifest = root / "qa" / "qa_manifest.json"
            write_json(manifest, {"runs": [{"video": "clip.MOV", "qa_events_path": str(qa_events)}]})
            reviews_dir = root / "reviews"
            write_json(
                reviews_dir / "clip.review.json",
                {
                    "source_video": "clip.MOV",
                    "items": [
                        {
                            "id": "miss-r001-dropgap-0006150",
                            "source": "manual",
                            "active_learning_candidate": True,
                            "kind": "drop_floor",
                            "status": "missing",
                            "time_sec": 6.15,
                            "x": 320,
                            "y": 880,
                            "contact_side": "unknown",
                            "contact_type": "ground",
                            "drop_source": "gap_without_floor_reset",
                            "review_tags": ["active_learning", "gap_without_floor_reset"],
                        }
                    ],
                },
            )

            out_root = root / "qa_reviewed"
            result = apply_reviews_to_qa.run_apply(manifest, reviews_dir, out_root)
            reviewed_manifest = result["manifest_path"]
            reviewed_doc = json.loads((out_root / "clip" / "qa_events.json").read_text(encoding="utf-8"))
            self.assertEqual(result["summary"]["manual_missing_inserted"], 1)
            self.assertEqual(sum(1 for event in reviewed_doc["events"] if event["type"] == "drop_floor"), 1)
            self.assertEqual(reviewed_doc["review_application"]["manual_missing_inserted"], 1)

            audit = strict_rally_audit.audit_manifest(reviewed_manifest)
            row = audit["rallies"][0]
            self.assertTrue(row["strict_complete"])
            self.assertEqual(row["rejection_reasons"], [])
            self.assertNotIn("ended_by_gap_without_floor_reset", row["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
