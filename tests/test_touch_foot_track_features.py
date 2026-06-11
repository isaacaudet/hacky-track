from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import attach_touch_foot_track_features as foot_track


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class TouchFootTrackFeatureTests(unittest.TestCase):
    def test_attach_dataset_adds_temporal_and_manual_calibration_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            labels_dir = root / "labels"
            rows = [
                {
                    "video_id": "video-a",
                    "video_name": "video-a.MOV",
                    "candidate_time_sec": 1.0,
                    "pose_feature_status": "ok",
                    "pose_geometry_nearest_side": "left",
                    "pose_left_foot_min_dist_px": 10.0,
                    "pose_right_foot_min_dist_px": 30.0,
                    "pose_geometry_side_margin_px": 20.0,
                    "pose_left_foot_min_dist_norm_shank": 0.2,
                    "pose_left_edge_surface_guess": "outer",
                    "pose_left_edge_surface_margin_px": 4.0,
                },
                {
                    "video_id": "video-a",
                    "video_name": "video-a.MOV",
                    "candidate_time_sec": 2.0,
                    "pose_feature_status": "ok",
                    "pose_geometry_nearest_side": "left",
                    "pose_left_foot_min_dist_px": 12.0,
                    "pose_geometry_side_margin_px": 22.0,
                    "pose_left_foot_min_dist_norm_shank": 0.24,
                },
                {
                    "video_id": "video-a",
                    "video_name": "video-a.MOV",
                    "candidate_time_sec": 3.0,
                    "pose_feature_status": "ok",
                    "pose_geometry_nearest_side": "right",
                    "pose_left_foot_min_dist_px": 40.0,
                    "pose_right_foot_min_dist_px": 8.0,
                    "pose_geometry_side_margin_px": 32.0,
                    "pose_right_foot_min_dist_norm_shank": 0.16,
                },
            ]
            write_jsonl(dataset_dir / "touch_training_candidates.jsonl", rows)
            write_jsonl(dataset_dir / "touch_training_test_frozen.jsonl", [])
            write_json(
                labels_dir / "video-a.events.json",
                {
                    "source_video": "video-a.MOV",
                    "rallies": [
                        {
                            "events": [
                                {"time_sec": 1.0, "type": "touch", "review_status": "approved", "trick_label": "left_kick"},
                                {"time_sec": 3.0, "type": "touch", "review_status": "approved", "trick_label": "right_kick"},
                            ]
                        }
                    ],
                },
            )

            manifest = foot_track.attach_dataset(
                type(
                    "Args",
                    (),
                    {
                        "dataset_dir": dataset_dir,
                        "out_dir": dataset_dir,
                        "labels_dir": labels_dir,
                        "label_match_tolerance_sec": 0.08,
                        "window_sec": 2.0,
                    },
                )()
            )

            out_rows = foot_track.read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
            self.assertEqual(manifest["status"], "features_attached")
            self.assertEqual(out_rows[0]["foot_track_feature_status"], "ok")
            self.assertEqual(out_rows[0]["foot_track_nearest_pose_side"], "left")
            self.assertEqual(out_rows[0]["foot_track_next_pose_side"], "left")
            self.assertEqual(out_rows[0]["foot_track_nearest_surface_guess"], "outer")
            self.assertIsNone(out_rows[0]["foot_track_right_dist_velocity_after_px_s"])
            self.assertEqual(out_rows[0]["manual_foot_calibrated_side"], "left")
            self.assertEqual(out_rows[2]["manual_foot_calibrated_side"], "right")


if __name__ == "__main__":
    unittest.main()
