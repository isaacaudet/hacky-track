from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_detector_label_review_batch


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class BuildDetectorLabelReviewBatchTests(unittest.TestCase):
    def test_collect_candidates_prioritizes_detector_label_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_events = root / "qa" / "clip" / "qa_events.json"
            write_json(
                qa_events,
                {
                    "events": [
                        {
                            "review_item_id": "touch-1",
                            "type": "touch",
                            "time_sec": 1.0,
                            "qa_frame_time_sec": 1.02,
                            "qa_ball_x": 100.0,
                            "qa_ball_y": 120.0,
                            "qa_ball_radius": 12.0,
                            "qa_ball_confidence": 0.72,
                            "qa_ball_accuracy": "high",
                            "qa_ball_correction_px": 18.0,
                        },
                        {
                            "review_item_id": "touch-2",
                            "type": "touch",
                            "time_sec": 2.0,
                            "qa_ball_x": 300.0,
                            "qa_ball_y": 320.0,
                            "qa_ball_confidence": 0.41,
                            "qa_ball_accuracy": "medium",
                            "qa_ball_correction_px": 180.0,
                        },
                    ]
                },
            )
            qa_manifest = root / "qa_manifest.json"
            write_json(qa_manifest, {"runs": [{"video": "clip.MOV", "qa_events_path": str(qa_events)}]})
            candidates = build_detector_label_review_batch.collect_candidates(qa_manifest)
            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[0].record["event_item_id"], "touch-2")
            self.assertEqual(candidates[0].record["suggested_detector_label"], "verify_or_correct")
            self.assertIn("large_heuristic_correction", candidates[0].record["selection_reasons"])
            self.assertEqual(candidates[1].record["suggested_detector_label"], "footbag")

    def test_reviewed_items_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_events = root / "qa_events.json"
            write_json(
                qa_events,
                {
                    "events": [
                        {
                            "review_item_id": "touch-1",
                            "type": "touch",
                            "time_sec": 1.0,
                            "qa_ball_x": 100.0,
                            "qa_ball_y": 120.0,
                            "qa_ball_confidence": 0.72,
                            "qa_ball_accuracy": "high",
                        }
                    ]
                },
            )
            qa_manifest = root / "qa_manifest.json"
            write_json(qa_manifest, {"runs": [{"video": "clip.MOV", "qa_events_path": str(qa_events)}]})
            reviews = root / "reviews"
            write_json(
                reviews / "clip.review.json",
                {"source_video": "clip.MOV", "items": [{"id": "touch-1", "status": "approved"}]},
            )
            candidates = build_detector_label_review_batch.collect_candidates(qa_manifest, reviews_dir=reviews)
            self.assertEqual(candidates, [])

    def test_dry_run_builds_manifest_without_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_events = root / "qa_events.json"
            write_json(
                qa_events,
                {
                    "events": [
                        {
                            "review_item_id": "touch-1",
                            "type": "touch",
                            "time_sec": 1.0,
                            "qa_ball_x": 100.0,
                            "qa_ball_y": 120.0,
                            "qa_ball_confidence": 0.72,
                            "qa_ball_accuracy": "high",
                        }
                    ]
                },
            )
            qa_manifest = root / "qa_manifest.json"
            write_json(qa_manifest, {"runs": [{"video": "clip.MOV", "qa_events_path": str(qa_events)}]})
            summary = build_detector_label_review_batch.build_review_batch(qa_manifest, root / "out", dry_run=True)
            self.assertEqual(summary["total_candidates"], 1)
            self.assertEqual(summary["selected_items"], 1)
            self.assertIsNone(summary["contact_sheet_path"])
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
