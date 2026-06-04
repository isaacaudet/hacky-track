from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import attach_touch_cotracker_features as cotracker


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_video(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
    try:
        for _ in range(8):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:] = (30, 30, 30)
            cv2.circle(frame, (24, 32), 4, (0, 0, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


class FakePoseModel:
    def det_model(self, frame):
        return np.array([[0, 0, frame.shape[1], frame.shape[0]]], dtype=float)

    def pose_model(self, frame, bboxes):
        keypoints = np.zeros((1, 23, 2), dtype=float)
        scores = np.zeros((1, 23), dtype=float)
        keypoints[0, 13] = [20, 15]
        keypoints[0, 15] = [20, 30]
        keypoints[0, 17] = [24, 36]
        keypoints[0, 18] = [16, 36]
        keypoints[0, 19] = [20, 28]
        keypoints[0, 14] = [52, 15]
        keypoints[0, 16] = [52, 30]
        keypoints[0, 20] = [56, 36]
        keypoints[0, 21] = [48, 36]
        keypoints[0, 22] = [52, 28]
        scores[:] = 0.0
        for idx in (13, 15, 17, 18, 19, 14, 16, 20, 21, 22):
            scores[0, idx] = 1.0
        return keypoints, scores


class FakeRunner:
    def __init__(self, *, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

    def track(self, frames_rgb, queries, query_frame, *, scale_x, scale_y):
        tracks = np.zeros((frames_rgb.shape[0], len(queries), 2), dtype=float)
        visibility = np.ones((frames_rgb.shape[0], len(queries)), dtype=float)
        for t in range(frames_rgb.shape[0]):
            for i, query in enumerate(queries):
                tracks[t, i] = [query.x + 0.5 * t, query.y]
        return cotracker.TrackResult(tracks=tracks, visibility=visibility)


class TouchCoTrackerFeatureTests(unittest.TestCase):
    def test_attach_dataset_adds_cotracker_features_from_seeded_foot_tracks(self) -> None:
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
                        "candidate_time_sec": 0.3,
                        "visual_crop_frame_index": 3,
                        "visual_crop_ball_x": 24.0,
                        "visual_crop_ball_y": 34.0,
                    }
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
                                    "time_sec": 0.3,
                                    "review_status": "approved",
                                    "trick_label": "left_kick",
                                }
                            ]
                        }
                    ],
                },
            )
            original_loader = cotracker.load_pose_model
            cotracker.load_pose_model = lambda _args: FakePoseModel()
            try:
                manifest = cotracker.attach_dataset(
                    argparse.Namespace(
                        dataset_dir=dataset_dir,
                        out_dir=dataset_dir,
                        labels_dir=labels_dir,
                        review_manifest=root / "manifest.json",
                        label_match_tolerance_sec=0.08,
                        contact_labeled_only=True,
                        row_limit=None,
                        dry_run=False,
                        pose_mode="performance",
                        pose_device="cpu",
                        cotracker_model="fake",
                        cotracker_device="cpu",
                        window_sec=0.4,
                        process_width=64,
                        keypoint_threshold=0.25,
                        visibility_threshold=0.5,
                    ),
                    runner_factory=FakeRunner,
                )
            finally:
                cotracker.load_pose_model = original_loader

            rows = cotracker.read_jsonl(dataset_dir / "touch_training_candidates.jsonl")
            self.assertEqual(manifest["status"], "features_attached")
            self.assertEqual(rows[0]["cotracker_feature_status"], "ok")
            self.assertEqual(rows[0]["cotracker_nearest_track_side"], "left")
            self.assertGreater(rows[0]["cotracker_side_confidence"], 0)
            self.assertEqual(rows[0]["cotracker_query_count"], 8)


if __name__ == "__main__":
    unittest.main()
