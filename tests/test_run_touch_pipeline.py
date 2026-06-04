import argparse
import unittest

from run_touch_pipeline import (
    release_feature_disable_flags,
    release_gate_status,
    should_run_diagnostic_classifier,
    training_table_effective_status,
    visual_crop_feature_summary_for_status,
    vision_embedding_feature_summary_for_status,
)


class RunTouchPipelineStatusTests(unittest.TestCase):
    def test_not_ready_when_classifier_has_not_trained(self) -> None:
        status = release_gate_status({"status": "not_ready", "reasons": ["missing labels"]})

        self.assertEqual(status["status"], "not_ready")
        self.assertFalse(status["passes"])
        self.assertFalse(status["strict_trained"])

    def test_trained_classifier_still_fails_when_any_release_gate_fails(self) -> None:
        status = release_gate_status(
            {
                "status": "trained",
                "cv_gate": {"passes": True},
                "frozen_test_gate": {"passes": False},
            }
        )

        self.assertEqual(status["status"], "trained_gate_failed")
        self.assertFalse(status["passes"])
        self.assertTrue(status["strict_trained"])
        self.assertTrue(status["cv_gate_passes"])
        self.assertFalse(status["frozen_test_gate_passes"])

    def test_release_gate_passes_only_when_strict_trained_and_both_gates_pass(self) -> None:
        status = release_gate_status(
            {
                "status": "trained",
                "cv_gate": {"passes": True},
                "frozen_test_gate": {"passes": True},
            }
        )

        self.assertEqual(status["status"], "release_gate_passed")
        self.assertTrue(status["passes"])
        self.assertTrue(status["strict_trained"])
        self.assertTrue(status["cv_gate_passes"])
        self.assertTrue(status["frozen_test_gate_passes"])

    def test_release_excludes_flow_and_pose_by_default(self) -> None:
        flags = release_feature_disable_flags(
            argparse.Namespace(include_flow_features=False, include_pose_features=False)
        )

        self.assertTrue(flags["disable_flow_features"])
        self.assertTrue(flags["disable_pose_features"])

    def test_release_can_opt_into_flow_and_pose(self) -> None:
        flags = release_feature_disable_flags(
            argparse.Namespace(include_flow_features=True, include_pose_features=True)
        )

        self.assertFalse(flags["disable_flow_features"])
        self.assertFalse(flags["disable_pose_features"])

    def test_training_table_status_reflects_attached_l2_features(self) -> None:
        status = training_table_effective_status(
            {"status": "waiting_for_l2_features", "candidate_rows": 5},
            {"train_val_ok_rows": 5, "test_frozen_ok_rows": 0},
        )

        self.assertEqual(status, "l2_features_attached")

    def test_training_table_status_reflects_partial_l2_features(self) -> None:
        status = training_table_effective_status(
            {"status": "waiting_for_l2_features", "candidate_rows": 5},
            {"train_val_ok_rows": 2, "test_frozen_ok_rows": 0},
        )

        self.assertEqual(status, "partial_l2_features_attached")

    def test_visual_crop_summary_reports_attachment_counts(self) -> None:
        summary = visual_crop_feature_summary_for_status(
            {
                "status": "features_attached",
                "train_val": {"rows": 4, "ok_rows": 3},
                "test_frozen": {"rows": 2, "ok_rows": 1},
            }
        )

        self.assertEqual(summary["status"], "features_attached")
        self.assertEqual(summary["train_val_rows"], 4)
        self.assertEqual(summary["train_val_ok_rows"], 3)
        self.assertEqual(summary["test_frozen_rows"], 2)
        self.assertEqual(summary["test_frozen_ok_rows"], 1)

    def test_vision_embedding_summary_reports_requested_and_ok_counts(self) -> None:
        summary = vision_embedding_feature_summary_for_status(
            {
                "status": "features_attached",
                "model_name": "owlv2",
                "contact_labeled_only": True,
                "train_val": {"rows": 5, "requested_rows": 2, "ok_rows": 2},
                "test_frozen": {"rows": 7, "requested_rows": 3, "ok_rows": 1},
            }
        )

        self.assertEqual(summary["status"], "features_attached")
        self.assertEqual(summary["model_name"], "owlv2")
        self.assertTrue(summary["contact_labeled_only"])
        self.assertEqual(summary["train_val_requested_rows"], 2)
        self.assertEqual(summary["train_val_ok_rows"], 2)
        self.assertEqual(summary["test_frozen_requested_rows"], 3)
        self.assertEqual(summary["test_frozen_ok_rows"], 1)

    def test_runs_diagnostic_classifier_when_only_frozen_test_is_missing(self) -> None:
        self.assertTrue(
            should_run_diagnostic_classifier(
                {
                    "status": "not_ready",
                    "reasons": [
                        "no frozen-test candidate rows; release gate evaluation requires visually reviewed frozen-test labels"
                    ],
                },
                audio_only=False,
            )
        )

    def test_diagnostic_classifier_does_not_hide_other_blockers(self) -> None:
        self.assertFalse(
            should_run_diagnostic_classifier(
                {
                    "status": "not_ready",
                    "reasons": [
                        "no frozen-test candidate rows; release gate evaluation requires visually reviewed frozen-test labels",
                        "L2 trajectory features are missing; run the trajectory feature attachment step first",
                    ],
                },
                audio_only=False,
            )
        )

    def test_diagnostic_classifier_is_skipped_for_audio_only_runs(self) -> None:
        self.assertFalse(
            should_run_diagnostic_classifier(
                {
                    "status": "not_ready",
                    "reasons": [
                        "no frozen-test candidate rows; release gate evaluation requires visually reviewed frozen-test labels"
                    ],
                },
                audio_only=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
