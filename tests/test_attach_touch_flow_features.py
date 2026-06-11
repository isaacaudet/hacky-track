import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import attach_touch_flow_features as flow


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class AttachTouchFlowFeaturesTests(unittest.TestCase):
    def make_video(self, path: Path, *, dx_per_frame: float) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (96, 96))
        if not writer.isOpened():
            self.skipTest("OpenCV VideoWriter could not open mp4v output")
        for frame in range(40):
            image = np.zeros((96, 96, 3), dtype=np.uint8)
            x = int(round(30 + frame * dx_per_frame))
            cv2.circle(image, (x, 48), 6, (255, 255, 255), -1)
            writer.write(image)
        writer.release()

    def test_flow_attachment_preserves_splits_and_adds_motion_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            train_video = root / "train.mp4"
            test_video = root / "test.mp4"
            self.make_video(train_video, dx_per_frame=0.5)
            self.make_video(test_video, dx_per_frame=0.5)
            write_json(
                root / "touch_review_manifest.json",
                {
                    "items": [
                        {"video_id": "train", "video_name": "train.mp4", "video_path": str(train_video), "split": "train"},
                        {"video_id": "test", "video_name": "test.mp4", "video_path": str(test_video), "split": "test_frozen"},
                    ]
                },
            )
            write_jsonl(
                dataset / "touch_training_candidates.jsonl",
                [
                    {
                        "video_id": "train",
                        "video_name": "train.mp4",
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
                        "video_name": "test.mp4",
                        "split": "test_frozen",
                        "candidate_time_sec": 0.5,
                        "label_is_touch": False,
                    }
                ],
            )
            detections = root / "detections.jsonl"
            rows = []
            for video in ["train.mp4", "test.mp4"]:
                for frame in range(40):
                    rows.append(
                        {
                            "source_video": video,
                            "frame_index": frame,
                            "time_sec": frame / 30.0,
                            "x": 30 + frame * 0.5,
                            "y": 48.0,
                            "confidence": 0.9,
                        }
                    )
            write_jsonl(detections, rows)

            manifest = flow.attach_dataset(
                argparse.Namespace(
                    dataset_dir=dataset,
                    out_dir=root / "out",
                    review_manifest=root / "touch_review_manifest.json",
                    detections_jsonl=[detections],
                    detections_dir=[],
                    threshold=0.2,
                    ball_tolerance_sec=0.08,
                    frame_step=2,
                    patch_radius_px=14,
                    grid_step_px=4,
                    cache_path=None,
                    force=False,
                )
            )

            train_rows = read_jsonl(root / "out" / "touch_training_candidates.jsonl")
            test_rows = read_jsonl(root / "out" / "touch_training_test_frozen.jsonl")
            self.assertEqual(manifest["status"], "features_attached")
            self.assertEqual(manifest["train_val"]["ok_rows"], 1)
            self.assertEqual(manifest["test_frozen"]["ok_rows"], 1)
            self.assertFalse(any(row["split"] == "test_frozen" for row in train_rows))
            self.assertEqual(test_rows[0]["split"], "test_frozen")
            self.assertEqual(train_rows[0]["flow_feature_status"], "ok")
            self.assertFalse(train_rows[0]["flow_missing"])
            self.assertIsNotNone(train_rows[0]["flow_after_mag"])
            self.assertTrue((root / "out" / "touch_flow_frame_features_cache.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
