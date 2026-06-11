from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_detector_error_review_batch


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class BuildDetectorErrorReviewBatchTests(unittest.TestCase):
    def test_collects_track_failures_and_keeps_test_as_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_manifest = root / "qa_manifest.json"
            write_json(qa_manifest, {"runs": [{"video": "clip.MOV"}, {"video": "heldout.MOV"}]})
            track_metrics = root / "track_metrics.json"
            write_json(
                track_metrics,
                {
                    "rows": [
                        {
                            "kind": "positive",
                            "result": "pass",
                            "split": "train",
                            "video": "clip.MOV",
                            "item_id": "ok",
                            "time_sec": 1.0,
                            "expected_x": 100,
                            "expected_y": 200,
                        },
                        {
                            "kind": "positive",
                            "result": "fail",
                            "split": "train",
                            "video": "clip.MOV",
                            "item_id": "bad-center",
                            "time_sec": 1.2,
                            "expected_x": 110,
                            "expected_y": 210,
                            "track_x": 180,
                            "track_y": 260,
                            "center_error_px": 86.0,
                            "confidence": 0.4,
                        },
                        {
                            "kind": "hard_negative",
                            "result": "not_applicable",
                            "hard_negative_result": "false_positive_near_bad_point",
                            "split": "test",
                            "video": "heldout.MOV",
                            "item_id": "hard-neg",
                            "time_sec": 2.0,
                            "expected_x": 300,
                            "expected_y": 400,
                            "track_x": 304,
                            "track_y": 398,
                            "confidence": 0.05,
                        },
                    ]
                },
            )

            candidates = build_detector_error_review_batch.collect_track_error_candidates(
                track_metrics=track_metrics,
                qa_manifest=qa_manifest,
            )

            self.assertEqual(len(candidates), 2)
            by_reason = {candidate.record["selection_reason"]: candidate.record for candidate in candidates}
            self.assertEqual(by_reason["track_center_fail"]["suggested_detector_label"], "verify_or_correct")
            self.assertEqual(by_reason["track_center_fail"]["training_use"], "train_or_calibration")
            self.assertEqual(by_reason["hard_negative_false_positive"]["suggested_detector_label"], "not_footbag")
            self.assertEqual(by_reason["hard_negative_false_positive"]["training_use"], "audit_only")

    def test_collects_model_failures_from_dataset_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_manifest = root / "qa_manifest.json"
            write_json(qa_manifest, {"runs": [{"video": "clip.MOV"}]})
            dataset = root / "dataset"
            write_jsonl(
                dataset / "reviewed_detector_labels.jsonl",
                [
                    {
                        "image": "images/train/clip__item-a.jpg",
                        "label": "labels/train/clip__item-a.txt",
                        "source_video": "clip.MOV",
                        "item_id": "item-a",
                        "split": "train",
                        "time_sec": 1.25,
                        "x": 122,
                        "y": 456,
                        "radius": 20,
                    },
                    {
                        "image": "images/train/clip__item-b.jpg",
                        "label": "labels/train/clip__item-b.txt",
                        "source_video": "clip.MOV",
                        "item_id": "item-b",
                        "split": "train",
                        "time_sec": 1.5,
                        "x": 222,
                        "y": 556,
                        "radius": 22,
                    },
                ],
            )
            model_metrics = root / "model_metrics.json"
            write_json(
                model_metrics,
                {
                    "rows": [
                        {
                            "kind": "positive_label",
                            "result": "fail_center",
                            "split": "train",
                            "image": "images/train/clip__item-a.jpg",
                            "expected_x": 122,
                            "expected_y": 456,
                            "prediction_x": 160,
                            "prediction_y": 460,
                            "confidence": 0.004,
                            "center_error_px": 38,
                        },
                        {
                            "kind": "positive_label",
                            "result": "low_confidence_near_label",
                            "split": "train",
                            "image": "images/train/clip__item-b.jpg",
                            "expected_x": 222,
                            "expected_y": 556,
                            "prediction_x": 224,
                            "prediction_y": 558,
                            "confidence": 0.003,
                            "center_error_px": 3,
                        },
                    ]
                },
            )

            candidates = build_detector_error_review_batch.collect_model_error_candidates(
                model_metrics=model_metrics,
                dataset=dataset,
                qa_manifest=qa_manifest,
                low_confidence_per_split=1,
            )

            self.assertEqual({candidate.record["selection_reason"] for candidate in candidates}, {"model_center_fail", "low_confidence_near_label"})
            self.assertTrue(all(candidate.record["source"] == "v10_detector_model_evaluation" for candidate in candidates))

    def test_dry_run_summary_does_not_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_manifest = root / "qa_manifest.json"
            track_metrics = root / "track_metrics.json"
            write_json(qa_manifest, {"runs": []})
            write_json(track_metrics, {"rows": []})

            summary = build_detector_error_review_batch.build_error_review_batch(
                qa_manifest=qa_manifest,
                track_metrics=track_metrics,
                out_dir=root / "out",
                dry_run=True,
            )

            self.assertEqual(summary["selected_items"], 0)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
