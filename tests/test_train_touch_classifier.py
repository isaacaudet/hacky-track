import argparse
import json
import tempfile
import unittest
from pathlib import Path

import train_touch_classifier as trainer


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


class TrainTouchClassifierTests(unittest.TestCase):
    def make_args(self, root: Path, **overrides) -> argparse.Namespace:
        values = {
            "dataset_dir": root,
            "out_dir": root / "model",
            "threshold": 0.5,
            "min_videos": 2,
            "allow_small": False,
            "allow_missing_trajectory": False,
            "allow_missing_frozen_test": False,
            "audio_only": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def row(self, video_id: str, time_sec: float, label: bool, split: str = "train", trajectory: bool = True) -> dict:
        row = {
            "video_id": video_id,
            "video_name": f"{video_id}.MOV",
            "split": split,
            "candidate_time_sec": time_sec,
            "label_is_touch": label,
            "has_audio": True,
            "audio_strength": 8.0 if label else 1.0,
            "has_existing_hint": False,
            "time_since_prev_candidate_sec": 0.4,
            "time_to_next_candidate_sec": 0.4,
            "nearest_drop_delta_sec": None,
            "nearest_stall_delta_sec": None,
            "in_stall_window": False,
            "trajectory_break_support": None,
            "trajectory_nearest_break_delta_sec": None,
            "trajectory_max_positive_dvy": None,
            "height_reversal": None,
            "detector_confidence_near_candidate": None,
        }
        if trajectory:
            row.update(
                {
                    "trajectory_break_support": 3 if label else 0,
                    "trajectory_nearest_break_delta_sec": 0.02 if label else 0.6,
                    "trajectory_max_positive_dvy": 200.0 if label else 0.0,
                    "height_reversal": label,
                    "detector_confidence_near_candidate": 0.4,
                }
            )
        return row

    def test_refuses_missing_trajectory_features_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [self.row("a", 1.0, True, trajectory=False), self.row("a", 2.0, False, trajectory=False)]
            write_jsonl(root / "touch_training_candidates.jsonl", rows)
            write_jsonl(root / "touch_training_test_frozen.jsonl", [])

            summary = trainer.train_classifier(self.make_args(root, min_videos=1, allow_small=True))

            self.assertEqual(summary["status"], "not_ready")
            self.assertTrue(any("trajectory" in reason for reason in summary["reasons"]))

    def test_trains_with_clip_disjoint_cv_when_features_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self.row("a", 1.0, True),
                self.row("a", 2.0, False),
                self.row("b", 1.0, True),
                self.row("b", 2.0, False),
            ]
            write_jsonl(root / "touch_training_candidates.jsonl", rows)
            write_jsonl(
                root / "touch_training_test_frozen.jsonl",
                [
                    self.row("heldout", 1.0, True, split="test_frozen"),
                    self.row("heldout", 2.0, False, split="test_frozen"),
                ],
            )

            summary = trainer.train_classifier(self.make_args(root))

            self.assertEqual(summary["status"], "trained")
            self.assertTrue((root / "model" / "touch_classifier.joblib").exists())
            self.assertEqual(summary["test_frozen_videos"], ["heldout"])
            self.assertEqual(len(summary["leave_one_video_out"]["folds"]), 2)
            self.assertIn("ablation_leave_one_video_out", summary)
            self.assertIn("fused_audio_trajectory", summary["ablation_leave_one_video_out"])
            self.assertIsNotNone(summary["frozen_test"])
            self.assertEqual(summary["frozen_test"]["rows"], 2)
            self.assertIn("passes", summary["frozen_test_gate"])
            self.assertEqual(summary["candidate_touch_precision_gate"], 0.90)
            self.assertEqual(summary["candidate_touch_recall_gate"], 0.85)
            self.assertTrue(summary["strict_release_mode"])
            self.assertTrue((root / "model" / "touch_classifier_errors.jsonl").exists())
            self.assertTrue((root / "model" / "touch_classifier_errors.csv").exists())
            self.assertIn("out_of_fold_errors", summary)
            self.assertIn("false_negative", summary["out_of_fold_errors"])

    def test_ablation_includes_frozen_test_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self.row("a", 1.0, True),
                self.row("a", 2.0, False),
                self.row("b", 1.0, True),
                self.row("b", 2.0, False),
            ]
            write_jsonl(root / "touch_training_candidates.jsonl", rows)
            write_jsonl(
                root / "touch_training_test_frozen.jsonl",
                [
                    self.row("heldout", 1.0, True, split="test_frozen"),
                    self.row("heldout", 2.0, False, split="test_frozen"),
                ],
            )

            summary = trainer.train_classifier(self.make_args(root))

            self.assertIn("ablation_frozen_test", summary)
            self.assertIn("fused_audio_trajectory", summary["ablation_frozen_test"])
            aggregate = summary["ablation_frozen_test"]["fused_audio_trajectory"]["aggregate"]
            self.assertIsNotNone(aggregate)
            for key in ("precision", "recall", "f1"):
                self.assertIn(key, aggregate)

    def test_candidate_precision_gate_vetoes_audio_fire_without_trajectory(self) -> None:
        prediction = {
            **self.row("heldout", 1.0, False, split="test_frozen"),
            "predicted_is_touch": True,
            "touch_score": 0.99,
            "error_type": "false_positive",
            "trajectory_break_support": 0,
            "trajectory_nearest_break_delta_sec": 0.8,
        }

        gated = trainer.apply_candidate_precision_gate([prediction])[0]

        self.assertFalse(gated["predicted_is_touch"])
        self.assertTrue(gated["candidate_precision_gate_vetoed"])
        self.assertEqual(gated["candidate_precision_gate_reason"], "no_trajectory_corroboration")
        self.assertEqual(gated["error_type"], "true_negative")

    def test_candidate_precision_gate_keeps_near_break_candidate(self) -> None:
        prediction = {
            **self.row("heldout", 1.0, True, split="test_frozen"),
            "predicted_is_touch": True,
            "touch_score": 0.99,
            "error_type": "true_positive",
            "trajectory_break_support": 1,
            "trajectory_nearest_break_delta_sec": 0.08,
        }

        gated = trainer.apply_candidate_precision_gate([prediction])[0]

        self.assertTrue(gated["predicted_is_touch"])
        self.assertFalse(gated["candidate_precision_gate_vetoed"])
        self.assertIsNone(gated["candidate_precision_gate_reason"])

    def test_candidate_precision_gate_vetoes_weak_audio_weak_trajectory(self) -> None:
        prediction = {
            **self.row("heldout", 1.0, False, split="test_frozen"),
            "predicted_is_touch": True,
            "touch_score": 0.99,
            "error_type": "false_positive",
            "audio_strength": 3.5,
            "trajectory_break_support": 4,
            "trajectory_nearest_break_delta_sec": 0.02,
        }

        gated = trainer.apply_candidate_precision_gate([prediction])[0]

        self.assertFalse(gated["predicted_is_touch"])
        self.assertTrue(gated["candidate_precision_gate_vetoed"])
        self.assertEqual(gated["candidate_precision_gate_reason"], "weak_audio_weak_trajectory")
        self.assertEqual(gated["error_type"], "true_negative")

    def test_candidate_recall_rescue_adds_high_impulse_audio_trajectory_candidate(self) -> None:
        prediction = {
            **self.row("heldout", 1.0, True, split="test_frozen"),
            "predicted_is_touch": False,
            "touch_score": 0.01,
            "error_type": "false_negative",
            "audio_strength": 12.0,
            "trajectory_break_support": 1,
            "trajectory_nearest_break_delta_sec": 0.04,
            "trajectory_impulse_score": 1200.0,
        }

        gated = trainer.apply_candidate_precision_gate([prediction])[0]

        self.assertTrue(gated["predicted_is_touch"])
        self.assertTrue(gated["candidate_recall_rescued"])
        self.assertEqual(gated["candidate_recall_rescue_reason"], "high_impulse_audio_trajectory_rescue")
        self.assertEqual(gated["error_type"], "true_positive")

    def test_candidate_recall_rescue_requires_nearby_trajectory_break(self) -> None:
        prediction = {
            **self.row("heldout", 1.0, True, split="test_frozen"),
            "predicted_is_touch": False,
            "touch_score": 0.01,
            "error_type": "false_negative",
            "audio_strength": 20.0,
            "trajectory_break_support": 0,
            "trajectory_nearest_break_delta_sec": 0.40,
            "trajectory_impulse_score": None,
        }

        gated = trainer.apply_candidate_precision_gate([prediction])[0]

        self.assertFalse(gated["predicted_is_touch"])
        self.assertFalse(gated["candidate_recall_rescued"])
        self.assertIsNone(gated["candidate_recall_rescue_reason"])

    def test_event_metrics_merge_duplicate_predicted_candidates(self) -> None:
        predictions = [
            {
                **self.row("heldout", 1.0, True, split="test_frozen"),
                "label_touch_time_sec": 1.0,
                "predicted_is_touch": True,
                "touch_score": 0.80,
            },
            {
                **self.row("heldout", 1.10, False, split="test_frozen"),
                "label_touch_time_sec": None,
                "predicted_is_touch": True,
                "touch_score": 0.95,
            },
            {
                **self.row("heldout", 3.0, False, split="test_frozen"),
                "label_touch_time_sec": None,
                "predicted_is_touch": False,
                "touch_score": 0.10,
            },
        ]

        candidate_metrics = trainer.metrics_from_predictions(predictions)
        event_metrics = trainer.event_level_metrics_from_predictions(predictions)
        event_rows = trainer.event_rows_from_predictions(predictions)

        self.assertEqual(candidate_metrics["false_positive"], 1)
        self.assertEqual(event_metrics["false_positive"], 0)
        self.assertEqual(event_metrics["false_negative"], 0)
        self.assertEqual(event_metrics["precision"], 1.0)
        self.assertEqual(event_metrics["recall"], 1.0)
        self.assertEqual(len(event_rows), 1)
        self.assertEqual(event_rows[0]["event_match_type"], "true_positive")
        self.assertEqual(event_rows[0]["candidate_count"], 2)
        self.assertEqual(event_rows[0]["suppressed_candidate_times_sec"], [1.0])

    def test_event_metrics_use_approved_visual_labels_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels_dir = Path(tmp)
            (labels_dir / "heldout.events.json").write_text(
                json.dumps(
                    {
                        "rallies": [
                            {
                                "events": [
                                    {
                                        "type": "touch",
                                        "time_sec": 1.0,
                                        "review_status": "approved",
                                    }
                                ]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            predictions = [
                {
                    **self.row("heldout", 1.0, False, split="test_frozen"),
                    "label_touch_time_sec": None,
                    "predicted_is_touch": True,
                    "touch_score": 0.80,
                },
            ]

            event_metrics = trainer.event_level_metrics_from_predictions(predictions, labels_dir=labels_dir)
            event_rows = trainer.event_rows_from_predictions(predictions, labels_dir=labels_dir)

            self.assertEqual(event_metrics["true_positive"], 1)
            self.assertEqual(event_metrics["false_positive"], 0)
            self.assertEqual(event_metrics["false_negative"], 0)
            self.assertEqual(event_rows[0]["event_match_type"], "true_positive")

    def test_training_summary_reports_raw_and_precision_gated_frozen_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self.row("a", 1.0, True),
                self.row("a", 2.0, False),
                self.row("b", 1.0, True),
                self.row("b", 2.0, False),
            ]
            write_jsonl(root / "touch_training_candidates.jsonl", rows)
            write_jsonl(
                root / "touch_training_test_frozen.jsonl",
                [
                    self.row("heldout", 1.0, True, split="test_frozen"),
                    self.row("heldout", 2.0, False, split="test_frozen"),
                ],
            )

            summary = trainer.train_classifier(self.make_args(root))

            self.assertTrue(summary["candidate_precision_gate_enabled"])
            self.assertIn("raw_frozen_test", summary)
            self.assertIn("raw_leave_one_video_out", summary)
            self.assertIn("candidate_precision_gate_vetoes", summary)
            self.assertTrue((root / "model" / "touch_classifier_frozen_predictions.jsonl").exists())
            self.assertIn("event_level_frozen_test", summary)
            self.assertIn("event_level_leave_one_video_out", summary)
            self.assertIn("candidate_recall_rescues", summary)
            self.assertTrue((root / "model" / "touch_classifier_frozen_events.jsonl").exists())
            self.assertTrue((root / "model" / "touch_classifier_oof_events.jsonl").exists())

    def test_report_renders_frozen_test_ablation_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self.row("a", 1.0, True),
                self.row("a", 2.0, False),
                self.row("b", 1.0, True),
                self.row("b", 2.0, False),
            ]
            write_jsonl(root / "touch_training_candidates.jsonl", rows)
            write_jsonl(
                root / "touch_training_test_frozen.jsonl",
                [
                    self.row("heldout", 1.0, True, split="test_frozen"),
                    self.row("heldout", 2.0, False, split="test_frozen"),
                ],
            )

            trainer.train_classifier(self.make_args(root))

            report = (root / "model" / "touch_classifier_report.md").read_text(encoding="utf-8")
            self.assertIn("frozen F1", report)

    def test_rejects_test_frozen_leak_in_training_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [self.row("heldout", 1.0, True, split="test_frozen")]
            write_jsonl(root / "touch_training_candidates.jsonl", rows)
            write_jsonl(root / "touch_training_test_frozen.jsonl", [])

            summary = trainer.train_classifier(self.make_args(root, min_videos=1, allow_small=True))

            self.assertEqual(summary["status"], "not_ready")
            self.assertTrue(any("test_frozen" in reason for reason in summary["reasons"]))

    def test_rejects_missing_frozen_test_rows_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self.row("a", 1.0, True),
                self.row("a", 2.0, False),
                self.row("b", 1.0, True),
                self.row("b", 2.0, False),
            ]
            write_jsonl(root / "touch_training_candidates.jsonl", rows)
            write_jsonl(root / "touch_training_test_frozen.jsonl", [])

            summary = trainer.train_classifier(self.make_args(root))

            self.assertEqual(summary["status"], "not_ready")
            self.assertTrue(any("frozen-test" in reason for reason in summary["reasons"]))

    def test_allows_missing_frozen_test_only_as_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self.row("a", 1.0, True),
                self.row("a", 2.0, False),
                self.row("b", 1.0, True),
                self.row("b", 2.0, False),
            ]
            write_jsonl(root / "touch_training_candidates.jsonl", rows)
            write_jsonl(root / "touch_training_test_frozen.jsonl", [])

            summary = trainer.train_classifier(self.make_args(root, allow_missing_frozen_test=True))

            self.assertEqual(summary["status"], "trained_smoke")
            self.assertFalse(summary["strict_release_mode"])
            self.assertTrue(summary["allow_missing_frozen_test"])
            self.assertIsNone(summary["frozen_test"])
            self.assertFalse(summary["frozen_test_gate"]["passes"])
            self.assertTrue((root / "model" / "touch_classifier_errors.jsonl").exists())

    def test_rejects_train_test_video_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self.row("a", 1.0, True),
                self.row("a", 2.0, False),
                self.row("b", 1.0, True),
                self.row("b", 2.0, False),
            ]
            write_jsonl(root / "touch_training_candidates.jsonl", rows)
            write_jsonl(
                root / "touch_training_test_frozen.jsonl",
                [
                    self.row("a", 3.0, True, split="test_frozen"),
                    self.row("a", 4.0, False, split="test_frozen"),
                ],
            )

            summary = trainer.train_classifier(self.make_args(root))

            self.assertEqual(summary["status"], "not_ready")
            self.assertTrue(any("split leakage" in reason for reason in summary["reasons"]))


if __name__ == "__main__":
    unittest.main()
