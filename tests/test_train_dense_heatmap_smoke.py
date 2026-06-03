import json
import tempfile
import unittest
from pathlib import Path

import train_dense_heatmap_smoke


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def base_record(clip_id: str, *, training_use: str = "train_or_calibration") -> dict:
    return {
        "image_path": "unused.jpg",
        "clip_id": clip_id,
        "source_video": f"{clip_id}.mov",
        "frame_index": 1,
        "visibility": "visible",
        "split": "train",
        "training_use": training_use,
        "is_target": True,
        "center_x": 10.0,
        "center_y": 20.0,
        "sigma_px": 10.0,
    }


class DenseHeatmapSmokeTests(unittest.TestCase):
    def test_dataset_discipline_accepts_sealed_clip_disjoint_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "train.jsonl", [base_record("clip-train")])
            validation = base_record("clip-val")
            validation["split"] = "validation"
            write_jsonl(root / "val.jsonl", [validation])
            audit = base_record("clip-audit", training_use="audit_only")
            audit["split"] = "test"
            write_jsonl(root / "test_audit.jsonl", [audit])

            summary = train_dense_heatmap_smoke.validate_dataset_discipline(root)

            self.assertEqual(summary["train_records"], 1)
            self.assertEqual(summary["validation_records"], 1)
            self.assertEqual(summary["test_audit_records_sealed"], 1)

    def test_dataset_discipline_rejects_audit_leak_and_clip_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leaked = base_record("clip-train", training_use="audit_only")
            write_jsonl(root / "train.jsonl", [leaked])
            validation = base_record("clip-val")
            validation["split"] = "validation"
            write_jsonl(root / "val.jsonl", [validation])
            audit = base_record("clip-audit", training_use="audit_only")
            audit["split"] = "test"
            write_jsonl(root / "test_audit.jsonl", [audit])

            with self.assertRaises(AssertionError):
                train_dense_heatmap_smoke.validate_dataset_discipline(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "train.jsonl", [base_record("clip-shared")])
            validation = base_record("clip-shared")
            validation["split"] = "validation"
            write_jsonl(root / "val.jsonl", [validation])
            audit = base_record("clip-audit", training_use="audit_only")
            audit["split"] = "test"
            write_jsonl(root / "test_audit.jsonl", [audit])

            with self.assertRaises(AssertionError):
                train_dense_heatmap_smoke.validate_dataset_discipline(root)

    def test_summarize_predictions_counts_visible_and_no_target_failures(self) -> None:
        rows = [
            {"is_target": True, "confidence": 0.9, "center_error_px": 6.0},
            {"is_target": True, "confidence": 0.8, "center_error_px": 30.0},
            {"is_target": True, "confidence": 0.1, "center_error_px": 3.0},
            {"is_target": False, "confidence": 0.7, "center_error_px": None},
            {"is_target": False, "confidence": 0.05, "center_error_px": None},
        ]

        summary = train_dense_heatmap_smoke.summarize_predictions(rows, threshold=0.5)

        self.assertEqual(summary["visible_frames"], 3)
        self.assertEqual(summary["passed_visible_frames"], 1)
        self.assertEqual(summary["failed_visible_frames"], 1)
        self.assertEqual(summary["missing_visible_frames"], 1)
        self.assertEqual(summary["no_target_frames"], 2)
        self.assertEqual(summary["false_positive_no_target_frames"], 1)
        self.assertAlmostEqual(summary["visible_pass_rate"], 1 / 3, places=6)
        self.assertEqual(summary["no_target_false_positive_rate"], 0.5)

    def test_compare_with_baselines_keeps_visible_and_fp_claims_separate(self) -> None:
        comparison = train_dense_heatmap_smoke.compare_with_baselines(
            [
                {"threshold": 0.1, "visible_pass_rate": 0.4, "no_target_false_positive_rate": 0.2},
                {"threshold": 0.5, "visible_pass_rate": 0.45, "no_target_false_positive_rate": 0.01},
            ],
            {
                "v10": {"visible_pass_rate": 0.35, "no_target_false_positive_rate": 0.4},
                "v11": {"visible_pass_rate": 0.5, "no_target_false_positive_rate": 0.1},
            },
        )

        self.assertEqual(comparison["best_baseline_visible"]["name"], "v11")
        self.assertFalse(comparison["beats_best_baseline_visible"])
        self.assertEqual(comparison["best_baseline_no_target_fp"]["name"], "v11")
        self.assertTrue(comparison["beats_best_baseline_no_target_fp"])


if __name__ == "__main__":
    unittest.main()
