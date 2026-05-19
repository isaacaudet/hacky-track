from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import evaluate_dense_trajectory


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class EvaluateDenseTrajectoryTests(unittest.TestCase):
    def test_evaluates_visible_and_no_target_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            predictions = root / "predictions.jsonl"
            write_jsonl(
                labels,
                [
                    {
                        "clip_id": "clip-a",
                        "source_video": "clip.mp4",
                        "split": "train",
                        "training_use": "train_or_calibration",
                        "frame_index": 1,
                        "time_sec": 0.1,
                        "x": 100,
                        "y": 200,
                        "radius": 10,
                        "visibility": "visible",
                        "quality": "reviewed",
                    },
                    {
                        "clip_id": "clip-a",
                        "source_video": "clip.mp4",
                        "split": "train",
                        "training_use": "train_or_calibration",
                        "frame_index": 2,
                        "time_sec": 0.2,
                        "x": 100,
                        "y": 200,
                        "radius": 10,
                        "visibility": "visible",
                        "quality": "reviewed",
                    },
                    {
                        "clip_id": "clip-a",
                        "source_video": "clip.mp4",
                        "split": "train",
                        "training_use": "train_or_calibration",
                        "frame_index": 3,
                        "time_sec": 0.3,
                        "x": None,
                        "y": None,
                        "radius": 10,
                        "visibility": "out_of_frame",
                        "quality": "reviewed",
                    },
                ],
            )
            write_jsonl(
                predictions,
                [
                    {"clip_id": "clip-a", "source_video": "clip.mp4", "frame_index": 1, "x": 105, "y": 198, "confidence": 0.9},
                    {"clip_id": "clip-a", "source_video": "clip.mp4", "frame_index": 2, "x": 180, "y": 198, "confidence": 0.9},
                    {"clip_id": "clip-a", "source_video": "clip.mp4", "frame_index": 3, "x": 50, "y": 50, "confidence": 0.9},
                ],
            )

            summary = evaluate_dense_trajectory.evaluate_dense_trajectory(
                labels_jsonl=labels,
                predictions_jsonl=predictions,
                out_dir=root / "metrics",
                tolerance_px=12,
            )

            self.assertEqual(summary["overall"]["visible_frames"], 2)
            self.assertEqual(summary["overall"]["passed_visible_frames"], 1)
            self.assertEqual(summary["overall"]["failed_visible_frames"], 1)
            self.assertEqual(summary["overall"]["no_target_frames"], 1)
            self.assertEqual(summary["overall"]["false_positive_no_target_frames"], 1)
            self.assertTrue((root / "metrics" / "dense_trajectory_metrics.json").exists())

    def test_dry_run_ignores_pending_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            write_jsonl(
                labels,
                [
                    {
                        "clip_id": "clip-a",
                        "source_video": "clip.mp4",
                        "split": "train",
                        "frame_index": 1,
                        "time_sec": 0.1,
                        "visibility": "unlabeled",
                        "quality": "pending",
                    }
                ],
            )

            summary = evaluate_dense_trajectory.evaluate_dense_trajectory(
                labels_jsonl=labels,
                out_dir=root / "metrics",
                dry_run=True,
            )

            self.assertEqual(summary["overall"]["visible_frames"], 0)
            self.assertFalse((root / "metrics").exists())


if __name__ == "__main__":
    unittest.main()
