import argparse
import json
import tempfile
import unittest
from pathlib import Path

import attach_touch_pose_features as pose


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class AttachTouchPoseFeaturesTests(unittest.TestCase):
    def test_cache_only_attachment_preserves_splits_and_adds_pose_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            review_manifest = root / "touch_review_manifest.json"
            detections = root / "detections.jsonl"
            cache = root / "pose_cache.jsonl"
            write_json(
                review_manifest,
                {
                    "items": [
                        {"video_id": "train", "video_name": "train.MOV", "video_path": "/tmp/train.MOV", "split": "train"},
                        {"video_id": "test", "video_name": "test.MOV", "video_path": "/tmp/test.MOV", "split": "test_frozen"},
                    ]
                },
            )
            write_jsonl(
                dataset / "touch_training_candidates.jsonl",
                [
                    {
                        "video_id": "train",
                        "video_name": "train.MOV",
                        "split": "train",
                        "candidate_time_sec": 1.0,
                        "label_is_touch": True,
                    }
                ],
            )
            write_jsonl(
                dataset / "touch_training_test_frozen.jsonl",
                [
                    {
                        "video_id": "test",
                        "video_name": "test.MOV",
                        "split": "test_frozen",
                        "candidate_time_sec": 1.0,
                        "label_is_touch": False,
                    }
                ],
            )
            write_jsonl(
                detections,
                [
                    {
                        "source_video": name,
                        "time_sec": 1.0,
                        "frame_index": 30,
                        "detections": [{"score": 0.9, "x": 10.0, "y": 20.0}],
                    }
                    for name in ["train.MOV", "test.MOV"]
                ],
            )
            ball = pose.BallPoint(time_sec=1.0, frame_index=30, x=10.0, y=20.0, score=0.9)
            rows = []
            for name, dist in [("train.MOV", 12.0), ("test.MOV", 80.0)]:
                rows.append(
                    {
                        "cache_key": pose.cache_key(
                            video_name=name,
                            frame_index=30,
                            mode="performance",
                            kpt_thr=0.25,
                            ball=ball,
                        ),
                        "pose_feature_status": "ok",
                        "pose_frame_index": 30,
                        "pose_ball_x": 10.0,
                        "pose_ball_y": 20.0,
                        "pose_ball_score": 0.9,
                        "pose_ball_missing": False,
                        "pose_missing": False,
                        "pose_present": True,
                        "pose_person_boxes": 1,
                        "pose_lower_body_present": True,
                        "pose_foot_present": True,
                        "pose_shank_length_px": 60.0,
                        "pose_nearest_foot_part": "left_big_toe",
                        "pose_nearest_foot_conf": 0.8,
                        "pose_nearest_foot_dist_px": dist,
                        "pose_nearest_foot_dist_norm_shank": dist / 60.0,
                        "pose_nearest_lower_part": "left_big_toe",
                        "pose_nearest_lower_conf": 0.8,
                        "pose_nearest_lower_dist_px": dist,
                        "pose_nearest_lower_dist_norm_shank": dist / 60.0,
                    }
                )
            write_jsonl(cache, rows)

            manifest = pose.attach_dataset(
                argparse.Namespace(
                    dataset_dir=dataset,
                    out_dir=root / "out",
                    review_manifest=review_manifest,
                    detections_jsonl=[detections],
                    detections_dir=[],
                    threshold=0.2,
                    ball_tolerance_sec=0.08,
                    pose_mode="performance",
                    pose_device="cpu",
                    keypoint_threshold=0.25,
                    cache_path=cache,
                    cache_only=True,
                )
            )

            train_rows = read_jsonl(root / "out" / "touch_training_candidates.jsonl")
            test_rows = read_jsonl(root / "out" / "touch_training_test_frozen.jsonl")
            self.assertEqual(manifest["status"], "features_attached")
            self.assertEqual(manifest["train_val"]["pose_present_rows"], 1)
            self.assertEqual(manifest["test_frozen"]["pose_present_rows"], 1)
            self.assertFalse(any(row["split"] == "test_frozen" for row in train_rows))
            self.assertEqual(test_rows[0]["split"], "test_frozen")
            self.assertEqual(train_rows[0]["pose_nearest_foot_part"], "left_big_toe")
            self.assertAlmostEqual(train_rows[0]["pose_nearest_foot_dist_px"], 12.0)


if __name__ == "__main__":
    unittest.main()
