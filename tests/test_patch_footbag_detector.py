from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import patch_footbag_detector


def write_image(path: Path, *, with_ball: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    image[:, :] = (70, 125, 70)
    if with_ball:
        cv2.circle(image, (80, 96), 13, (20, 20, 20), -1)
        cv2.circle(image, (75, 91), 5, (40, 40, 220), -1)
        cv2.circle(image, (86, 100), 4, (55, 55, 235), -1)
    cv2.imwrite(str(path), image)


def write_label(path: Path, *, with_ball: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if with_ball:
        path.write_text("0 0.500000 0.600000 0.180000 0.180000\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")


class PatchFootbagDetectorTests(unittest.TestCase):
    def test_red_object_proposals_find_synthetic_ball(self) -> None:
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        image[:, :] = (70, 125, 70)
        cv2.circle(image, (60, 70), 10, (30, 30, 220), -1)
        proposals = patch_footbag_detector.red_object_proposals(image, max_candidates=10)
        self.assertTrue(any(abs(x - 60) < 8 and abs(y - 70) < 8 for x, y in proposals))

    def test_train_and_evaluate_patch_detector_on_synthetic_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            for split in ["train", "validation", "test"]:
                for idx in range(2):
                    write_image(dataset / "images" / split / f"pos_{idx}.jpg", with_ball=True)
                    write_label(dataset / "labels" / split / f"pos_{idx}.txt", with_ball=True)
                write_image(dataset / "images" / split / "neg.jpg", with_ball=False)
                write_label(dataset / "labels" / split / "neg.txt", with_ball=False)
            hard_negative_crop = dataset / "hard_negatives" / "crops" / "shoe_like_false_positive.jpg"
            write_image(hard_negative_crop, with_ball=False)
            points_jsonl = dataset / "hard_negatives" / "points.jsonl"
            points_jsonl.write_text(
                json.dumps(
                    {
                        "split": "train",
                        "crop": "hard_negatives/crops/shoe_like_false_positive.jpg",
                        "source_video": "synthetic.mov",
                        "item_id": "reviewed-fp-001",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            train_summary = patch_footbag_detector.train_patch_detector(
                dataset=dataset,
                out_dir=root / "model",
                crop_size=56,
                jitter=1,
                negatives_per_image=4,
                seed=42,
            )
            self.assertTrue((root / "model" / "patch_footbag_detector.joblib").exists())
            self.assertEqual(train_summary["hard_negative_points"]["train"], 1)
            self.assertGreaterEqual(train_summary["sample_metrics"]["validation"]["recall"], 0.5)
            with (root / "model" / "patch_training_samples.csv").open(encoding="utf-8", newline="") as handle:
                sample_rows = list(csv.DictReader(handle))
            hard_negative_rows = [row for row in sample_rows if row["sample"] == "hard_negative_point"]
            self.assertEqual(len(hard_negative_rows), 1)
            self.assertEqual(hard_negative_rows[0]["item_id"], "reviewed-fp-001")

            eval_summary = patch_footbag_detector.evaluate_patch_detector(
                dataset=dataset,
                model_path=root / "model" / "patch_footbag_detector.joblib",
                out_dir=root / "eval",
                tolerance_px=12.0,
                max_candidates=30,
                max_detections=10,
            )
            self.assertGreaterEqual(eval_summary["overall"]["pass_rate"], 0.8)
            self.assertTrue((root / "eval" / "patch_detector_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
