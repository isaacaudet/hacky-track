import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import attach_touch_l2_features as attach


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class AttachTouchL2FeaturesTests(unittest.TestCase):
    def test_load_detection_tracks_supports_event_and_prediction_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detections.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "source_video": "video-a.MOV",
                        "frame_index": 1,
                        "time_sec": 0.1,
                        "detections": [{"score": 0.19, "x": 1, "y": 1}, {"score": 0.7, "x": 2, "y": 3}],
                    },
                    {
                        "source_video": "video-b.MOV",
                        "frame_index": 2,
                        "time_sec": 0.2,
                        "x": 4,
                        "y": 5,
                        "confidence": 0.8,
                    },
                ],
            )

            tracks = attach.load_detection_tracks([path], threshold=0.2)

            self.assertEqual(sorted(tracks), ["video-a.MOV", "video-b.MOV"])
            self.assertEqual(tracks["video-a.MOV"][0].x, 2)
            self.assertEqual(tracks["video-b.MOV"][0].confidence, 0.8)

    def test_attach_row_features_uses_nearby_breakpoints_and_local_reversal(self) -> None:
        row = {
            "video_name": "video-a.MOV",
            "candidate_time_sec": 1.0,
            "trajectory_feature_status": "missing_l2_features",
        }
        track = attach.TrackFeatures(
            status="ok",
            breakpoints=[
                {"time_sec": 0.94, "lambda": 1200, "dvy": 120.0},
                {"time_sec": 1.04, "lambda": 3000, "dvy": -10.0},
                {"time_sec": 2.0, "lambda": 1200, "dvy": 500.0},
            ],
            ts=np.asarray([0.88, 0.94, 1.06, 1.12]),
            xs=np.asarray([0.0, 1.0, 2.0, 3.0]),
            ys=np.asarray([0.0, 10.0, 8.0, 0.0]),
            confidences=np.asarray([0.3, 0.4, 0.7, 0.2]),
        )

        out = attach.attach_row_features(row, track, break_tolerance_sec=0.11)

        self.assertEqual(out["trajectory_feature_status"], "ok")
        self.assertEqual(out["trajectory_break_support"], 2)
        self.assertAlmostEqual(out["trajectory_nearest_break_delta_sec"], 0.04)
        self.assertEqual(out["trajectory_max_positive_dvy"], 120.0)
        self.assertTrue(out["height_reversal"])
        self.assertEqual(out["detector_confidence_near_candidate"], 0.7)

    def test_dataset_attachment_preserves_frozen_test_separation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            detections = root / "detections.jsonl"
            write_jsonl(
                dataset / "touch_training_candidates.jsonl",
                [
                    {
                        "video_id": "train",
                        "video_name": "video-train.MOV",
                        "split": "train",
                        "candidate_time_sec": 0.5,
                        "label_is_touch": True,
                    }
                ],
            )
            write_jsonl(
                dataset / "touch_training_test_frozen.jsonl",
                [
                    {
                        "video_id": "test",
                        "video_name": "video-test.MOV",
                        "split": "test_frozen",
                        "candidate_time_sec": 0.5,
                        "label_is_touch": True,
                    }
                ],
            )
            rows = []
            for video in ["video-train.MOV", "video-test.MOV"]:
                for frame in range(20):
                    rows.append(
                        {
                            "source_video": video,
                            "frame_index": frame,
                            "time_sec": frame / 30.0,
                            "x": float(frame),
                            "y": float(frame * frame),
                            "confidence": 0.9,
                        }
                    )
            write_jsonl(detections, rows)
            args = argparse.Namespace(
                dataset_dir=dataset,
                out_dir=root / "out",
                detections_jsonl=[detections],
                detections_dir=[],
                threshold=0.2,
                break_tolerance_sec=0.11,
                max_track_gap_sec=0.25,
                min_points=12,
            )

            manifest = attach.attach_dataset(args)
            train_rows = read_jsonl(root / "out" / "touch_training_candidates.jsonl")
            test_rows = read_jsonl(root / "out" / "touch_training_test_frozen.jsonl")

            self.assertEqual(manifest["train_val"]["rows"], 1)
            self.assertEqual(manifest["test_frozen"]["rows"], 1)
            self.assertFalse(any(row["split"] == "test_frozen" for row in train_rows))
            self.assertEqual(test_rows[0]["split"], "test_frozen")

    def test_discontinuous_windows_are_not_fit_as_one_global_track(self) -> None:
        first = [
            attach.TrackPoint(time_sec=frame / 30.0, x=float(frame), y=float(frame * frame), confidence=0.9)
            for frame in range(10)
        ]
        second = [
            attach.TrackPoint(time_sec=10.0 + frame / 30.0, x=float(frame), y=float(frame * frame), confidence=0.9)
            for frame in range(10)
        ]

        features = attach.compute_track_features(first + second, min_points=12, max_gap_sec=0.25)

        self.assertEqual(features.status, "too_few_segment_track_points")
        self.assertEqual(features.segments, 2)

    def test_continuous_segment_can_provide_features_even_with_distant_extra_window(self) -> None:
        first = [
            attach.TrackPoint(time_sec=frame / 30.0, x=float(frame), y=float(frame * frame), confidence=0.9)
            for frame in range(14)
        ]
        second = [
            attach.TrackPoint(time_sec=10.0 + frame / 30.0, x=float(frame), y=float(frame * frame), confidence=0.9)
            for frame in range(6)
        ]

        features = attach.compute_track_features(first + second, min_points=12, max_gap_sec=0.25)

        self.assertEqual(features.status, "ok")
        self.assertEqual(features.segments, 1)
        self.assertGreaterEqual(len(features.ts), 12)


if __name__ == "__main__":
    unittest.main()
