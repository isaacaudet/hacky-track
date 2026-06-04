from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import train_release_contact_classifier as contact


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class ReleaseContactClassifierTests(unittest.TestCase):
    def test_classification_quality_reports_balanced_accuracy(self) -> None:
        labels = ["left", "left", "right", "right", "right", "right"]
        preds = ["left", "right", "right", "right", "right", "left"]

        quality = contact.classification_quality(labels, preds)

        self.assertAlmostEqual(quality["accuracy"], 4 / 6)
        self.assertAlmostEqual(quality["per_class_recall"]["left"], 0.5)
        self.assertAlmostEqual(quality["per_class_recall"]["right"], 0.75)
        self.assertAlmostEqual(quality["balanced_accuracy"], 0.625)

    def test_pose_candidate_maps_lower_body_part_to_soft_type_and_side(self) -> None:
        row = {
            "pose_nearest_lower_part": "left_big_toe",
            "pose_nearest_foot_part": "left_ankle",
        }

        out = contact.pose_candidate(row)

        self.assertEqual(out["pose_candidate_type"], "kick")
        self.assertEqual(out["pose_candidate_side"], "left")

    def test_readiness_blocks_when_pose_and_contact_labels_are_missing(self) -> None:
        rows = [{"video_id": "a", "candidate_time_sec": 1.0}]

        summary = contact.readiness_summary(rows, [], min_examples=1, min_videos=1)

        self.assertEqual(summary["status"], "not_ready")
        self.assertTrue(any("pose" in reason for reason in summary["reasons"]))
        self.assertTrue(any("contact labels" in reason for reason in summary["reasons"]))

    def test_readiness_passes_when_pose_features_and_clip_disjoint_labels_exist(self) -> None:
        rows = [
            {
                "video_id": "a",
                "contact_type": "left_kick",
                "contact_side": "left",
                "pose_nearest_foot_dist_px": 12.0,
                "pose_nearest_lower_part": "left_big_toe",
            },
            {
                "video_id": "b",
                "contact_type": "right_kick",
                "contact_side": "right",
                "pose_nearest_foot_dist_px": 14.0,
                "pose_nearest_lower_part": "right_big_toe",
            },
        ]

        summary = contact.readiness_summary(rows, [], min_examples=2, min_videos=2)

        self.assertEqual(summary["status"], "ready_for_training")
        self.assertEqual(summary["contact_side_counts_in_rows"], {"left": 1, "right": 1})
        self.assertEqual(summary["contact_type_counts_in_rows"], {"kick": 2})

    def test_readiness_distinguishes_attached_pose_columns_from_usable_distance(self) -> None:
        rows = [
            {
                "video_id": "a",
                "contact_type": "left_kick",
                "contact_side": "left",
                "pose_feature_status": "ok",
                "pose_present": True,
                "pose_nearest_foot_dist_px": None,
            }
        ]

        summary = contact.readiness_summary(rows, [], min_examples=1, min_videos=1)

        self.assertEqual(summary["rows_with_pose_columns"], 1)
        self.assertEqual(summary["rows_with_pose_present"], 1)
        self.assertEqual(summary["rows_with_pose_features"], 0)
        self.assertTrue(any("no usable ball-to-body distance" in reason for reason in summary["reasons"]))
        self.assertFalse(any("pose/body proximity columns are missing" in reason for reason in summary["reasons"]))

    def test_readiness_counts_successful_visual_and_embedding_features(self) -> None:
        rows = [
            {
                "video_id": "a",
                "contact_type": "left_kick",
                "contact_side": "left",
                "pose_nearest_foot_dist_px": 12.0,
                "visual_crop_feature_status": "ok",
                "vision_embedding_present": True,
            },
            {
                "video_id": "b",
                "contact_type": "right_kick",
                "contact_side": "right",
                "pose_nearest_foot_dist_px": 14.0,
                "visual_crop_feature_status": "missing_ball",
                "vision_embedding_present": False,
                "vision_embedding_000": 0.0,
            },
        ]

        summary = contact.readiness_summary(rows, [], min_examples=2, min_videos=2)

        self.assertEqual(summary["rows_with_visual_crop_features"], 1)
        self.assertEqual(summary["rows_with_vision_embedding_features"], 1)

    def test_matches_event_file_contact_labels_to_candidate_rows(self) -> None:
        rows = [
            {
                "video_id": "video-a",
                "video_name": "video-a.MOV",
                "candidate_time_sec": 2.005,
                "pose_nearest_foot_dist_px": 12.0,
            },
            {
                "video_id": "video-a",
                "video_name": "video-a.MOV",
                "candidate_time_sec": 4.0,
                "pose_nearest_foot_dist_px": 40.0,
            },
        ]
        examples = [
            {
                "video_id": "video-a",
                "source_video": "video-a.MOV",
                "time_sec": 2.0,
                "event_type": "stall",
                "contact_type": "stall",
                "contact_side": None,
            }
        ]

        matched, match_summary = contact.attach_event_file_contact_labels(rows, examples, tolerance_sec=0.08)
        labeled = contact.rows_with_contact_labels(matched)

        self.assertEqual(match_summary["matched_examples"], 1)
        self.assertEqual(labeled[0]["contact_type"], "stall")
        self.assertEqual(labeled[0]["contact_label_source"], "visual_event_file")
        self.assertAlmostEqual(labeled[0]["contact_label_match_delta_sec"], 0.005)

    def test_loads_inner_outer_surface_labels_from_event_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "video-a.events.json",
                {
                    "source_video": "video-a.MOV",
                    "rallies": [
                        {
                            "id": 1,
                            "events": [
                                {
                                    "type": "touch",
                                    "time_sec": 2.0,
                                    "review_status": "approved",
                                    "contact_side": "left",
                                    "contact_type": "kick",
                                    "contact_surface": "outer",
                                    "trick_label": "left_outer_kick",
                                },
                                {
                                    "type": "touch",
                                    "time_sec": 3.0,
                                    "review_status": "approved",
                                    "contact_side": "right",
                                    "trick_label": "right_inner_knee",
                                },
                            ],
                        }
                    ],
                },
            )

            examples = contact.load_label_contact_examples(root)

            self.assertEqual(examples[0]["contact_type"], "kick")
            self.assertEqual(examples[0]["contact_side"], "left")
            self.assertEqual(examples[0]["contact_surface"], "outer")
            self.assertEqual(examples[1]["contact_type"], "knee")
            self.assertEqual(examples[1]["contact_side"], "right")
            self.assertEqual(examples[1]["contact_surface"], "inner")

    def test_loads_generic_knee_and_surface_stall_labels_from_event_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "video-a.events.json",
                {
                    "source_video": "video-a.MOV",
                    "rallies": [
                        {
                            "id": 1,
                            "events": [
                                {
                                    "type": "touch",
                                    "time_sec": 2.0,
                                    "review_status": "approved",
                                    "trick_label": "left_knee",
                                },
                                {
                                    "type": "stall",
                                    "time_sec": 3.0,
                                    "review_status": "approved",
                                    "trick_label": "right_outer_stall",
                                },
                            ],
                        }
                    ],
                },
            )

            examples = contact.load_label_contact_examples(root)

            self.assertEqual(examples[0]["contact_type"], "knee")
            self.assertEqual(examples[0]["contact_side"], "left")
            self.assertIsNone(examples[0]["contact_surface"])
            self.assertEqual(examples[1]["contact_type"], "stall")
            self.assertEqual(examples[1]["contact_side"], "right")
            self.assertEqual(examples[1]["contact_surface"], "outer")

    def test_ignores_non_wearer_side_labels_for_side_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "video-a.events.json",
                {
                    "source_video": "video-a.MOV",
                    "rallies": [
                        {
                            "id": 1,
                            "events": [
                                {
                                    "type": "touch",
                                    "time_sec": 2.0,
                                    "review_status": "approved",
                                    "contact_side": "left",
                                    "contact_side_basis": "screen_position",
                                    "contact_type": "kick",
                                },
                                {
                                    "type": "touch",
                                    "time_sec": 3.0,
                                    "review_status": "approved",
                                    "contact_side": "right",
                                    "contact_side_basis": "wearer_limb",
                                    "contact_type": "kick",
                                },
                            ],
                        }
                    ],
                },
            )

            examples = contact.load_label_contact_examples(root)

            self.assertIsNone(examples[0]["contact_side"])
            self.assertEqual(examples[0]["contact_side_basis"], "screen_position")
            self.assertEqual(examples[1]["contact_side"], "right")
            self.assertEqual(examples[1]["contact_side_basis"], "wearer_limb")

    def test_side_specific_trick_label_implies_wearer_limb_basis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "video-a.events.json",
                {
                    "source_video": "video-a.MOV",
                    "rallies": [
                        {
                            "id": 1,
                            "events": [
                                {
                                    "type": "touch",
                                    "time_sec": 2.0,
                                    "review_status": "approved",
                                    "contact_side": "left",
                                    "contact_type": "kick",
                                    "trick_label": "left_outer_kick",
                                }
                            ],
                        }
                    ],
                },
            )

            examples = contact.load_label_contact_examples(root)

            self.assertEqual(examples[0]["contact_side"], "left")
            self.assertEqual(examples[0]["contact_side_basis"], "wearer_limb")

    def test_loads_stall_contact_examples_from_reviewed_event_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "video-a.events.json",
                {
                    "source_video": "video-a.MOV",
                    "rallies": [
                        {
                            "id": 1,
                            "events": [
                                {"type": "stall", "time_sec": 2.0, "review_status": "approved"},
                                {"type": "touch", "time_sec": 3.0, "review_status": "approved"},
                            ],
                        }
                    ],
                },
            )

            examples = contact.load_label_contact_examples(root)

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0]["contact_type"], "stall")

    def test_trains_clip_disjoint_contact_side_model_when_labels_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            rows = []
            for video_id in ("video-a", "video-b", "video-c"):
                rows.extend(
                    [
                        {
                            "video_id": video_id,
                            "candidate_time_sec": 1.0,
                            "contact_side": "left",
                            "pose_nearest_lower_part": "left_big_toe",
                            "pose_nearest_foot_part": "left_big_toe",
                            "pose_nearest_lower_dist_px": 8.0,
                            "pose_nearest_foot_dist_px": 8.0,
                            "pose_nearest_lower_conf": 0.9,
                            "pose_nearest_foot_conf": 0.9,
                            "trajectory_impulse_score": 10.0,
                        },
                        {
                            "video_id": video_id,
                            "candidate_time_sec": 2.0,
                            "contact_side": "right",
                            "pose_nearest_lower_part": "right_big_toe",
                            "pose_nearest_foot_part": "right_big_toe",
                            "pose_nearest_lower_dist_px": 8.0,
                            "pose_nearest_foot_dist_px": 8.0,
                            "pose_nearest_lower_conf": 0.9,
                            "pose_nearest_foot_conf": 0.9,
                            "trajectory_impulse_score": 10.0,
                        },
                    ]
                )

            result = contact.train_contact_models(
                rows,
                out_dir=out_dir,
                min_examples=2,
                min_videos=2,
                targets=("contact_side",),
            )

            side = result["targets"]["contact_side"]
            self.assertEqual(side["status"], "trained")
            self.assertEqual(side["accuracy"], 1.0)
            self.assertEqual(side["rows"], 6)
            self.assertEqual(side["gate"], "fail")
            self.assertEqual(side["side_basis_counts"], {"legacy_unspecified": 6})
            self.assertIn("explicit wearer_limb side labels", side["gate_blockers"][0])
            self.assertTrue((out_dir / "release_contact_classifier.joblib").exists())

    def test_side_gate_passes_when_explicit_wearer_limb_labels_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            rows = []
            for video_id in ("video-a", "video-b", "video-c"):
                rows.extend(
                    [
                        {
                            "video_id": video_id,
                            "candidate_time_sec": 1.0,
                            "contact_side": "left",
                            "contact_side_basis": "wearer_limb",
                            "pose_nearest_lower_part": "left_big_toe",
                            "pose_nearest_foot_part": "left_big_toe",
                            "trajectory_impulse_score": 10.0,
                        },
                        {
                            "video_id": video_id,
                            "candidate_time_sec": 2.0,
                            "contact_side": "right",
                            "contact_side_basis": "wearer_limb",
                            "pose_nearest_lower_part": "right_big_toe",
                            "pose_nearest_foot_part": "right_big_toe",
                            "trajectory_impulse_score": 10.0,
                        },
                    ]
                )

            result = contact.train_contact_models(
                rows,
                out_dir=out_dir,
                min_examples=2,
                min_videos=2,
                targets=("contact_side",),
            )

            side = result["targets"]["contact_side"]
            self.assertEqual(side["status"], "trained")
            self.assertEqual(side["gate"], "pass")
            self.assertEqual(side["side_basis_counts"], {"wearer_limb": 6})
            self.assertEqual(side["gate_blockers"], [])

    def test_feature_dict_can_disable_visual_crop_features(self) -> None:
        row = {
            "pose_nearest_foot_dist_px": 12.0,
            "visual_ball_x_norm": 0.3,
            "visual_crop_grid_0_0_sat_mean": 0.8,
            "visual_crop_feature_status": "ok",
            "vision_embedding_000": 0.1,
            "vision_embedding_feature_status": "ok",
        }

        features = contact.contact_feature_dict(row, disabled_prefixes=("visual_",))

        self.assertIn("pose_nearest_foot_dist_px", features)
        self.assertNotIn("visual_ball_x_norm", features)
        self.assertNotIn("visual_crop_grid_0_0_sat_mean", features)
        self.assertNotIn("visual_crop_feature_status", features)
        self.assertIn("vision_embedding_000", features)
        self.assertIn("vision_embedding_feature_status", features)

    def test_feature_dict_can_disable_all_visual_features(self) -> None:
        row = {
            "pose_nearest_foot_dist_px": 12.0,
            "visual_ball_x_norm": 0.3,
            "vision_embedding_000": 0.1,
            "vision_embedding_feature_status": "ok",
        }

        features = contact.contact_feature_dict(row, disabled_prefixes=("visual_", "vision_"))

        self.assertIn("pose_nearest_foot_dist_px", features)
        self.assertNotIn("visual_ball_x_norm", features)
        self.assertNotIn("vision_embedding_000", features)
        self.assertNotIn("vision_embedding_feature_status", features)

    def test_ridge_classifier_is_available_as_bounded_contact_model_family(self) -> None:
        model = contact.build_model("ridge_classifier")

        self.assertIn("ridge_classifier", contact.CONTACT_MODEL_FAMILIES)
        self.assertIsNotNone(model)

    def test_ridge_classifier_confidence_uses_decision_margin(self) -> None:
        model = contact.build_model("ridge_classifier")
        features = [{"x": -2.0}, {"x": -1.0}, {"x": 1.0}, {"x": 2.0}]
        labels = ["left", "left", "right", "right"]
        model.fit(features, labels)

        confidences = contact.prediction_confidences(model, features)

        self.assertEqual(len(confidences), len(features))
        self.assertTrue(all(0.5 < confidence < 1.0 for confidence in confidences))
        self.assertGreater(len({round(confidence, 4) for confidence in confidences}), 1)

    def test_release_class_coverage_flags_missing_contact_type_classes(self) -> None:
        coverage = contact.release_class_coverage(
            {"kick": 25, "stall": 7},
            "contact_type",
            min_examples_per_class=20,
        )

        self.assertFalse(coverage["passes"])
        self.assertEqual(
            coverage["class_counts"],
            {"kick": 25, "stall": 7, "knee": 0, "drop_floor": 0},
        )
        self.assertIn("`knee` labels", " ".join(coverage["blockers"]))
        self.assertIn("`drop_floor` labels", " ".join(coverage["blockers"]))

    def test_contact_type_accuracy_pass_can_still_fail_release_scope_on_class_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            rows = []
            for video_id in ("video-a", "video-b", "video-c"):
                rows.extend(
                    [
                        {
                            "video_id": video_id,
                            "candidate_time_sec": 1.0,
                            "contact_type": "kick",
                            "pose_nearest_lower_part": "left_big_toe",
                            "pose_nearest_foot_dist_px": 8.0,
                            "trajectory_impulse_score": 10.0,
                        },
                        {
                            "video_id": video_id,
                            "candidate_time_sec": 2.0,
                            "contact_type": "stall",
                            "pose_nearest_lower_part": "left_big_toe",
                            "pose_nearest_foot_dist_px": 8.0,
                            "trajectory_impulse_score": 1.0,
                            "in_stall_window": True,
                        },
                    ]
                )

            result = contact.train_contact_models(
                rows,
                out_dir=out_dir,
                min_examples=2,
                min_videos=2,
                targets=("contact_type",),
            )

            target = result["targets"]["contact_type"]
            self.assertEqual(target["gate"], "pass")
            self.assertEqual(target["release_scope_gate"], "fail")
            blockers = " ".join(target["release_scope_blockers"])
            self.assertIn("`knee` labels", blockers)
            self.assertIn("`drop_floor` labels", blockers)

    def test_selective_accuracy_reports_coverage_and_accuracy(self) -> None:
        rows = [
            {"correct": True, "confidence": 0.9},
            {"correct": False, "confidence": 0.8},
            {"correct": True, "confidence": 0.4},
        ]

        result = contact.selective_accuracy_rows(rows, thresholds=(0.5, 0.85))

        self.assertEqual(result[0]["kept"], 2)
        self.assertAlmostEqual(result[0]["coverage"], 2 / 3)
        self.assertAlmostEqual(result[0]["accuracy"], 0.5)
        self.assertEqual(result[1]["kept"], 1)
        self.assertAlmostEqual(result[1]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
