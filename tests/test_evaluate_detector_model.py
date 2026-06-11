from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import evaluate_detector_model


class EvaluateDetectorModelTests(unittest.TestCase):
    def test_match_label_distinguishes_low_confidence_from_center_failure(self) -> None:
        label = {
            "center": [50.0, 50.0],
            "width": 20.0,
            "height": 20.0,
        }
        low_conf = evaluate_detector_model.match_label_to_predictions(
            label=label,
            detections=[{"center": [52.0, 51.0], "confidence": 0.04, "bbox": [45.0, 45.0, 58.0, 58.0]}],
            tolerance_px=8.0,
            box_tolerance_multiplier=0.75,
            release_confidence_threshold=0.25,
        )
        self.assertEqual(low_conf["result"], "low_confidence_near_label")
        self.assertTrue(low_conf["candidate_center_pass"])

        far = evaluate_detector_model.match_label_to_predictions(
            label=label,
            detections=[{"center": [90.0, 90.0], "confidence": 0.9, "bbox": [85.0, 85.0, 95.0, 95.0]}],
            tolerance_px=8.0,
            box_tolerance_multiplier=0.75,
            release_confidence_threshold=0.25,
        )
        self.assertEqual(far["result"], "fail_center")
        self.assertFalse(far["candidate_center_pass"])

    def test_evaluate_detector_model_uses_prediction_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            (dataset / "images" / "train").mkdir(parents=True)
            (dataset / "labels" / "train").mkdir(parents=True)
            (dataset / "images" / "validation").mkdir(parents=True)
            (dataset / "labels" / "validation").mkdir(parents=True)
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(dataset / "images" / "train" / "positive.jpg"), image)
            cv2.imwrite(str(dataset / "images" / "validation" / "negative.jpg"), image)
            (dataset / "labels" / "train" / "positive.txt").write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
            (dataset / "labels" / "validation" / "negative.txt").write_text("", encoding="utf-8")
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "image": "images/train/positive.jpg",
                                "detections": [{"bbox": [45, 45, 55, 55], "confidence": 0.8}],
                            }
                        ),
                        json.dumps(
                            {
                                "image": "images/validation/negative.jpg",
                                "detections": [{"bbox": [10, 10, 20, 20], "confidence": 0.8}],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = evaluate_detector_model.evaluate_detector_model(
                dataset=dataset,
                model=None,
                predictions_jsonl=predictions,
                out_dir=root / "metrics",
                tolerance_px=8.0,
                release_confidence_threshold=0.25,
            )

            self.assertEqual(summary["overall"]["positive_labels"], 1)
            self.assertEqual(summary["overall"]["pass_rate"], 1.0)
            self.assertEqual(summary["overall"]["hard_negative_images"], 1)
            self.assertEqual(summary["overall"]["hard_negative_false_positive_rate"], 1.0)
            self.assertIn("threshold_recommendation", summary)
            self.assertTrue((root / "metrics" / "detector_model_metrics.json").exists())

    def test_threshold_recommendation_uses_validation_hard_negative_gate(self) -> None:
        rows = [
            {
                "kind": "positive_label",
                "split": "validation",
                "candidate_center_pass": True,
                "confidence": 0.05,
                "top_confidence": 0.05,
            },
            {
                "kind": "positive_label",
                "split": "validation",
                "candidate_center_pass": True,
                "confidence": 0.02,
                "top_confidence": 0.02,
            },
            {
                "kind": "hard_negative_image",
                "split": "validation",
                "top_confidence": 0.03,
            },
        ]

        recommendation = evaluate_detector_model.recommended_threshold(
            rows,
            calibration_split="validation",
            max_hard_negative_false_positive_rate=0.0,
        )

        self.assertGreater(recommendation["recommended_threshold"], 0.03)
        self.assertEqual(recommendation["metrics"]["hard_negative_false_positives"], 0)
        self.assertEqual(recommendation["metrics"]["true_positives"], 1)


if __name__ == "__main__":
    unittest.main()
