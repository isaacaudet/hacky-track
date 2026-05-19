from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import summarize_detector_batch


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_manifest(path: Path, *, video: str, scanned: int, raw: int, track: int, model: int, predicted: int) -> None:
    write_json(
        path,
        {
            "video": video,
            "detector": "ultralytics-yolo-footbag",
            "source": "model",
            "parameters": {
                "confidence_threshold": 0.012,
                "confidence_threshold_source": "calibration_metrics",
                "tracker_mode": "greedy",
            },
            "video_info": {"scanned_frames": scanned},
            "counts": {
                "raw_detections": raw,
                "track_points": track,
                "model_detection_points": model,
                "predicted_track_points": predicted,
            },
        },
    )


class SummarizeDetectorBatchTests(unittest.TestCase):
    def test_summarizes_tracks_root_and_flags_coverage_risks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks = root / "tracks"
            write_manifest(
                tracks / "good" / "detector_inference_manifest.json",
                video="good.MOV",
                scanned=100,
                raw=80,
                track=90,
                model=80,
                predicted=10,
            )
            write_manifest(
                tracks / "weak" / "detector_inference_manifest.json",
                video="weak.MOV",
                scanned=100,
                raw=10,
                track=80,
                model=10,
                predicted=70,
            )

            summary = summarize_detector_batch.summarize_detector_batch(
                tracks_root=tracks,
                out_dir=root / "summary",
                high_prediction_share=0.35,
                low_model_coverage=0.20,
            )

            self.assertEqual(summary["counts"]["videos"], 2)
            self.assertEqual(summary["counts"]["total_raw_detections"], 90)
            self.assertEqual(summary["rates"]["model_coverage"], 0.45)
            self.assertEqual(summary["counts"]["flagged_videos"], 1)
            self.assertEqual(summary["flags"][0]["video"], "weak.MOV")
            self.assertIn("high_interpolation_share", summary["flags"][0]["reasons"])
            self.assertTrue((root / "summary" / "detector_batch_summary.json").exists())
            self.assertTrue((root / "summary" / "detector_batch_summary.csv").exists())
            self.assertTrue((root / "summary" / "detector_batch_summary.md").exists())

    def test_dry_run_does_not_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks = root / "tracks"
            write_manifest(
                tracks / "clip" / "detector_inference_manifest.json",
                video="clip.MOV",
                scanned=100,
                raw=0,
                track=0,
                model=0,
                predicted=0,
            )
            summary = summarize_detector_batch.summarize_detector_batch(
                tracks_root=tracks,
                out_dir=root / "summary",
                dry_run=True,
            )
            self.assertEqual(summary["counts"]["videos_with_no_raw_detections"], 1)
            self.assertFalse((root / "summary").exists())


if __name__ == "__main__":
    unittest.main()
