from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import train_footbag_detector


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class TrainFootbagDetectorTests(unittest.TestCase):
    def test_dry_run_builds_versioned_training_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            (dataset / "images" / "train").mkdir(parents=True)
            (dataset / "labels" / "train").mkdir(parents=True)
            (dataset / "data.yaml").write_text(
                "path: .\ntrain: images/train\nval: images/validation\ntest: images/test\nnames:\n  0: footbag\n",
                encoding="utf-8",
            )
            write_json(
                dataset / "manifest.json",
                {
                    "positive_labels": 2,
                    "hard_negative_points": 1,
                    "written_images": 3,
                    "written_hard_negative_yolo_images": 1,
                    "splits": {"clip.MOV": "test"},
                },
            )
            out = root / "models"
            summary = train_footbag_detector.train_detector(dataset, out, dry_run=True, epochs=3, base_model="yolo11n.pt")
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["trainer"], "ultralytics-yolo")
            self.assertEqual(summary["dataset_counts"]["positive_labels"], 2)
            self.assertEqual(summary["dataset_counts"]["written_hard_negative_yolo_images"], 1)
            self.assertIsNone(summary["model_path"])
            self.assertFalse((out / "training_manifest.json").exists())

    def test_runtime_data_yaml_uses_absolute_dataset_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            data_yaml = dataset / "data.yaml"
            data_yaml.write_text("path: .\n", encoding="utf-8")
            runtime = train_footbag_detector.write_runtime_data_yaml(data_yaml, root / "models")
            text = runtime.read_text(encoding="utf-8")
            self.assertIn(f"path: {dataset.resolve()}", text)
            self.assertIn("train: images/train", text)

    def test_training_output_dir_is_normalized_for_ultralytics_project(self) -> None:
        out_dir = train_footbag_detector.normalized_output_dir(Path("runs/demo_detector"))
        self.assertTrue(out_dir.is_absolute())
        self.assertEqual(out_dir.name, "demo_detector")

    def test_parse_results_csv_returns_final_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.csv"
            results.write_text(
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B)\n"
                "1,0.1,0.2,0.3\n"
                "2,0.4,0.5,0.6\n",
                encoding="utf-8",
            )
            metrics = train_footbag_detector.parse_results_csv(results)
            self.assertEqual(metrics["epoch"], 2.0)
            self.assertEqual(metrics["metrics/precision(B)"], 0.4)
            self.assertEqual(metrics["metrics/mAP50(B)"], 0.6)

    def test_parse_best_results_csv_returns_best_map_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results.csv"
            results.write_text(
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B)\n"
                "1,0.9,0.1,0.2\n"
                "2,0.4,0.5,0.7\n"
                "3,0.8,0.2,0.6\n",
                encoding="utf-8",
            )
            metrics = train_footbag_detector.parse_best_results_csv(results)
            self.assertEqual(metrics["epoch"], 2.0)
            self.assertEqual(metrics["metrics/mAP50(B)"], 0.7)

    def test_recover_existing_run_writes_manifest_and_model_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "data.yaml").write_text(
                "path: .\ntrain: images/train\nval: images/validation\ntest: images/test\nnames:\n  0: footbag\n",
                encoding="utf-8",
            )
            out = root / "models"
            run = out / "footbag-detector" / "weights"
            run.mkdir(parents=True)
            (run / "best.pt").write_bytes(b"fake weights")
            (run.parent / "results.csv").write_text(
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B)\n"
                "11,0.12,0.34,0.56\n",
                encoding="utf-8",
            )
            summary = train_footbag_detector.train_detector(
                dataset,
                out,
                dry_run=False,
                recover_existing=True,
                epochs=25,
                run_name="footbag-detector",
            )
            self.assertTrue(summary["recovered_existing_run"])
            self.assertEqual(summary["training_report"]["final_metrics"]["epoch"], 11.0)
            self.assertEqual(summary["training_report"]["final_metrics"]["metrics/recall(B)"], 0.34)
            self.assertEqual(summary["training_report"]["best_metrics"]["metrics/mAP50(B)"], 0.56)
            self.assertTrue((out / "footbag_detector_best.pt").exists())
            self.assertTrue((out / "training_manifest.json").exists())

    def test_missing_data_yaml_fails_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                train_footbag_detector.train_detector(Path(tmp) / "missing", Path(tmp) / "models", dry_run=True)


if __name__ == "__main__":
    unittest.main()
