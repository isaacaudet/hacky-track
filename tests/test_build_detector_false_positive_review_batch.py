from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_detector_false_positive_review_batch


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class BuildDetectorFalsePositiveReviewBatchTests(unittest.TestCase):
    def test_collects_false_positives_far_from_qa_ball(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_events = root / "qa" / "clip" / "qa_events.json"
            write_json(
                qa_events,
                {
                    "events": [
                        {"time_sec": 1.0, "qa_frame_time_sec": 1.0, "qa_ball_x": 100.0, "qa_ball_y": 100.0}
                    ]
                },
            )
            qa_manifest = root / "qa_manifest.json"
            write_json(qa_manifest, {"runs": [{"video": "clip.MOV", "qa_events_path": str(qa_events)}]})
            inference_dir = root / "inference" / "clip"
            write_json(
                inference_dir / "detector_inference_manifest.json",
                {
                    "video": "clip.MOV",
                    "video_info": {"width": 688, "height": 912, "fps": 30.0},
                },
            )
            write_jsonl(
                inference_dir / "raw_model_detections.jsonl",
                [
                    {"frame_index": 30, "time_sec": 1.0, "center": [102, 102], "bbox": [90, 90, 114, 114], "confidence": 0.02},
                    {"frame_index": 30, "time_sec": 1.0, "center": [430, 500], "bbox": [410, 480, 450, 520], "confidence": 0.03},
                ],
            )
            candidates = build_detector_false_positive_review_batch.collect_false_positive_candidates(
                qa_manifest=qa_manifest,
                inference_root=root / "inference",
            )
            self.assertEqual(len(candidates), 1)
            record = candidates[0].record
            self.assertEqual(record["suggested_detector_label"], "verify_or_correct")
            self.assertEqual(record["x"], 430.0)
            self.assertEqual(record["center_x"], 430.0)
            self.assertEqual(record["center_y"], 500.0)
            self.assertEqual(record["detector_confidence"], 0.03)
            self.assertEqual(record["time_s"], 1.0)
            self.assertEqual(record["video_id"], "clip")
            self.assertIn("object_label_review_candidate", record["selection_reasons"])

    def test_dry_run_summary_does_not_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_manifest = root / "qa_manifest.json"
            write_json(qa_manifest, {"runs": []})
            summary = build_detector_false_positive_review_batch.build_false_positive_review_batch(
                qa_manifest,
                root / "out",
                inference_root=root / "inference",
                dry_run=True,
            )
            self.assertEqual(summary["total_candidates"], 0)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
