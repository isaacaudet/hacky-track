from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import attach_touch_local_mask_features as local_mask


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_video(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (96, 96))
    try:
        for _ in range(6):
            frame = np.full((96, 96, 3), 160, dtype=np.uint8)
            cv2.circle(frame, (48, 48), 5, (0, 0, 220), -1)
            cv2.rectangle(frame, (24, 42), (40, 58), (25, 25, 25), -1)
            writer.write(frame)
    finally:
        writer.release()


class TouchLocalMaskFeatureTests(unittest.TestCase):
    def test_extract_features_finds_nearest_local_component(self) -> None:
        frame = np.full((96, 96, 3), 160, dtype=np.uint8)
        cv2.circle(frame, (48, 48), 5, (0, 0, 220), -1)
        cv2.rectangle(frame, (24, 42), (40, 58), (20, 20, 20), -1)
        ball = local_mask.BallPoint(x=48.0, y=48.0, frame_index=0, score=0.9)

        features = local_mask.extract_features(frame, ball, crop_size_px=72, ball_radius_px=6.0)

        self.assertEqual(features["local_mask_feature_status"], "ok")
        self.assertTrue(features["local_mask_component_present"])
        self.assertGreater(features["local_mask_ring_texture_frac"], 0)
        self.assertLess(features["local_mask_component_centroid_dx_norm"], 0)
        self.assertLess(features["local_mask_component_angle_cos"], 0)
        self.assertIn("local_mask_sector_left_texture_frac", features)

    def test_attach_dataset_processes_only_contact_labeled_rows_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            labels_dir = root / "labels"
            video_path = root / "video-a.MOV"
            write_video(video_path)
            write_json(
                root / "manifest.json",
                {
                    "items": [
                        {
                            "video_id": "video-a",
                            "video_name": "video-a.MOV",
                            "video_path": str(video_path),
                        }
                    ]
                },
            )
            write_jsonl(
                dataset_dir / "touch_training_candidates.jsonl",
                [
                    {
                        "video_id": "video-a",
                        "video_name": "video-a.MOV",
                        "split": "train",
                        "candidate_time_sec": 0.2,
                        "visual_crop_frame_index": 2,
                        "visual_crop_ball_x": 48.0,
                        "visual_crop_ball_y": 48.0,
                    },
                    {
                        "video_id": "video-a",
                        "video_name": "video-a.MOV",
                        "split": "train",
                        "candidate_time_sec": 0.5,
                        "visual_crop_frame_index": 5,
                        "visual_crop_ball_x": 48.0,
                        "visual_crop_ball_y": 48.0,
                    },
                ],
            )
            write_jsonl(dataset_dir / "touch_training_test_frozen.jsonl", [])
            write_json(
                labels_dir / "video-a.events.json",
                {
                    "source_video": "video-a.MOV",
                    "rallies": [
                        {
                            "events": [
                                {
                                    "type": "touch",
                                    "time_sec": 0.2,
                                    "review_status": "approved",
                                    "trick_label": "left_kick",
                                }
                            ]
                        }
                    ],
                },
            )

            manifest = local_mask.attach_dataset(
                argparse.Namespace(
                    dataset_dir=dataset_dir,
                    out_dir=dataset_dir,
                    labels_dir=labels_dir,
                    review_manifest=root / "manifest.json",
                    label_match_tolerance_sec=0.08,
                    contact_labeled_only=True,
                    crop_size_px=72,
                    ball_radius_px=6.0,
                )
            )

            rows = local_mask.read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
            self.assertEqual(manifest["status"], "features_attached")
            self.assertEqual(manifest["train_val"]["requested_rows"], 1)
            self.assertEqual(rows[0]["local_mask_feature_status"], "ok")
            self.assertEqual(rows[1]["local_mask_feature_status"], "skipped_not_requested")


if __name__ == "__main__":
    unittest.main()
