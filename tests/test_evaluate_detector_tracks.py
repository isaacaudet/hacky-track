from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import evaluate_detector_tracks


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class EvaluateDetectorTracksTests(unittest.TestCase):
    def test_evaluates_positive_labels_and_hard_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "dataset" / "reviewed_detector_labels.jsonl"
            hard_negatives = root / "dataset" / "hard_negatives" / "points.jsonl"
            write_jsonl(
                labels,
                [
                    {"item_id": "ok", "source_video": "clip.MOV", "split": "test", "time_sec": 1.0, "x": 100.0, "y": 100.0, "radius": 10.0},
                    {"item_id": "bad", "source_video": "clip.MOV", "split": "test", "time_sec": 2.0, "x": 200.0, "y": 200.0, "radius": 10.0},
                ],
            )
            write_jsonl(
                hard_negatives,
                [{"item_id": "hn", "source_video": "clip.MOV", "split": "test", "time_sec": 3.0, "x": 400.0, "y": 400.0, "radius": 10.0}],
            )
            track_dir = root / "tracks" / "clip"
            track_dir.mkdir(parents=True)
            (track_dir / "detector_track.json").write_text(
                json.dumps(
                    {
                        "track": [
                            {"frame_index": 30, "time_sec": 1.0, "source": "model_detection", "center": [104.0, 103.0], "confidence": 0.9},
                            {"frame_index": 60, "time_sec": 2.0, "source": "model_detection", "center": [250.0, 250.0], "confidence": 0.8},
                            {"frame_index": 90, "time_sec": 3.0, "source": "model_detection", "center": [401.0, 402.0], "confidence": 0.7},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            summary = evaluate_detector_tracks.evaluate_detector_tracks(
                labels_jsonl=labels,
                hard_negatives_jsonl=hard_negatives,
                tracks_root=root / "tracks",
                out_dir=root / "metrics",
                tolerance_px=12.0,
                radius_multiplier=1.0,
            )
            self.assertEqual(summary["overall"]["positive_labels"], 2)
            self.assertEqual(summary["overall"]["passed_positive_labels"], 1)
            self.assertEqual(summary["overall"]["failed_positive_labels"], 1)
            self.assertEqual(summary["overall"]["false_positive_hard_negatives"], 1)
            self.assertTrue((root / "metrics" / "detector_track_metrics.json").exists())
            self.assertTrue((root / "metrics" / "detector_track_metrics.csv").exists())
            self.assertTrue((root / "metrics" / "detector_track_metrics.md").exists())

    def test_missing_track_file_counts_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            write_jsonl(labels, [{"item_id": "missing", "source_video": "clip.MOV", "split": "validation", "time_sec": 1.0, "x": 100.0, "y": 100.0}])
            summary = evaluate_detector_tracks.evaluate_detector_tracks(
                labels_jsonl=labels,
                hard_negatives_jsonl=None,
                tracks_root=root / "tracks",
                out_dir=root / "metrics",
            )
            self.assertEqual(summary["overall"]["missing_track_files"], 1)
            self.assertEqual(summary["by_split"]["validation"]["pass_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
