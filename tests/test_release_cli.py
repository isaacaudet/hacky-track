from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import hackytrack


class ReleaseCliTests(unittest.TestCase):
    def test_create_run_dir_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = hackytrack.create_run_dir(root, "demo", overwrite=False)
            self.assertTrue(first.exists())
            with self.assertRaises(RuntimeError):
                hackytrack.create_run_dir(root, "demo", overwrite=False)
            second = hackytrack.create_run_dir(root, "demo", overwrite=True)
            self.assertEqual(first, second)

    def test_qa_totals_reads_release_manifest_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "qa_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "touch_candidates": 3,
                                "ground_hit_candidates": 1,
                                "stall_candidates": 1,
                                "around_the_world_candidates": 0,
                                "rallies": 1,
                                "suppressed_touch_candidates": 2,
                                "suppressed_drop_candidates": 1,
                                "large_ball_corrections": 4,
                            },
                            {
                                "touch_candidates": 2,
                                "ground_hit_candidates": 0,
                                "stall_candidates": 0,
                                "around_the_world_candidates": 1,
                                "rallies": 1,
                                "suppressed_touch_candidates": 0,
                                "suppressed_drop_candidates": 0,
                                "large_ball_corrections": 1,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            totals = hackytrack.qa_totals(manifest)
            self.assertEqual(totals["videos"], 2)
            self.assertEqual(totals["touches"], 5)
            self.assertEqual(totals["drops"], 1)
            self.assertEqual(totals["stalls"], 1)
            self.assertEqual(totals["atw"], 1)
            self.assertEqual(totals["rallies"], 2)
            self.assertEqual(totals["large_ball_corrections"], 5)

    def test_export_run_data_writes_top_level_release_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_dir = root / "qa" / "clip"
            qa_dir.mkdir(parents=True)
            qa_events = qa_dir / "qa_events.json"
            qa_events.write_text(
                json.dumps(
                    {
                        "source_video": "/private/path/clip.MOV",
                        "events": [
                            {
                                "type": "touch",
                                "time_sec": 1.23,
                                "qa_rally_id": 1,
                                "contact_side": "right",
                                "contact_type": "foot",
                            }
                        ],
                        "rallies": [{"id": 1, "touches": 1, "duration_sec": 0.5}],
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "qa" / "qa_manifest.json"
            manifest.write_text(json.dumps({"runs": [{"qa_events_path": str(qa_events)}]}), encoding="utf-8")
            artifacts = hackytrack.export_run_data(root, manifest)
            self.assertTrue(artifacts["events_json"].exists())
            self.assertTrue(artifacts["events_csv"].exists())
            self.assertTrue(artifacts["rallies_json"].exists())
            events = json.loads(artifacts["events_json"].read_text(encoding="utf-8"))["events"]
            self.assertEqual(events[0]["source_video"], "clip.MOV")
            csv_text = artifacts["events_csv"].read_text(encoding="utf-8")
            self.assertIn("source_video,video_index,type,time_sec", csv_text)
            self.assertIn("clip.MOV,0,touch,1.23", csv_text)

    def test_make_run_outputs_portable_removes_user_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            video = Path.home() / "Downloads" / "clip.MOV"
            summary = run_dir / "summary.md"
            summary.write_text(
                f"run {hackytrack.ROOT / 'runs/demo'} video {video}\n",
                encoding="utf-8",
            )
            audit = hackytrack.make_run_outputs_portable(run_dir, [video])
            self.assertTrue(audit["passed"])
            self.assertNotIn("/Users/", summary.read_text(encoding="utf-8"))
            self.assertIn("clip.MOV", summary.read_text(encoding="utf-8"))

    def test_resolve_run_manifest_path_handles_root_relative_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "demo"
            run_dir.mkdir(parents=True)
            resolved = hackytrack.resolve_run_manifest_path(run_dir, "reviews")
            self.assertEqual(resolved, hackytrack.ROOT / "reviews")
            training = hackytrack.resolve_run_manifest_path(run_dir, "training/full_training_manifest.json")
            self.assertEqual(training, run_dir / "training" / "full_training_manifest.json")

    def test_detect_footbag_command_is_public(self) -> None:
        parser = hackytrack.build_parser()
        args = parser.parse_args(
            [
                "detect-footbag",
                "--detections-jsonl",
                "detections.jsonl",
                "--out-dir",
                "detector_out",
                "--patch-model",
                "patch.joblib",
                "--tracker-mode",
                "temporal",
                "--process-width",
                "688",
                "--process-height",
                "912",
                "--dry-run",
            ]
        )
        self.assertEqual(args.func, hackytrack.detect_footbag)
        self.assertEqual(args.max_gap_frames, 6)
        self.assertEqual(args.patch_model, Path("patch.joblib"))
        self.assertEqual(args.tracker_mode, "temporal")
        self.assertEqual(args.process_width, 688)

    def test_apply_detector_track_command_is_public(self) -> None:
        parser = hackytrack.build_parser()
        args = parser.parse_args(
            [
                "apply-detector-track",
                "--qa-events",
                "qa_events.json",
                "--track-json",
                "detector_track.json",
                "--out-events",
                "out.json",
            ]
        )
        self.assertEqual(args.func, hackytrack.apply_detector_track)
        self.assertFalse(args.promote_model)

    def test_evaluate_detector_command_is_public(self) -> None:
        parser = hackytrack.build_parser()
        args = parser.parse_args(
            [
                "evaluate-detector",
                "--labels-jsonl",
                "labels.jsonl",
                "--tracks-root",
                "tracks",
                "--out-dir",
                "metrics",
            ]
        )
        self.assertEqual(args.func, hackytrack.evaluate_detector)
        self.assertEqual(args.tolerance_px, 24.0)

    def test_detector_batch_collects_videos_from_qa_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_root = root / "videos"
            video_root.mkdir()
            clip_a = video_root / "clip-a.MOV"
            clip_b = video_root / "clip-b.MOV"
            clip_a.write_bytes(b"a")
            clip_b.write_bytes(b"b")
            manifest = root / "qa_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "runs": [
                            {"video": "clip-a.MOV"},
                            {"video": "clip-b.MOV"},
                            {"video": "clip-a.MOV"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            videos = hackytrack.detector_batch_videos(
                explicit_videos=[],
                qa_manifest=manifest,
                video_root=video_root,
                run_dir=None,
            )
            self.assertEqual(videos, [clip_a.resolve(), clip_b.resolve()])
            self.assertEqual(hackytrack.portable_path_ref(clip_a.resolve()), "clip-a.MOV")

    def test_detect_footbag_batch_command_is_public(self) -> None:
        parser = hackytrack.build_parser()
        args = parser.parse_args(
            [
                "detect-footbag-batch",
                "--qa-manifest",
                "qa_manifest.json",
                "--detections-root",
                "detections",
                "--out-root",
                "tracks",
                "--patch-max-detections",
                "32",
                "--calibration-metrics",
                "detector_model_metrics.json",
                "--dry-run",
            ]
        )
        self.assertEqual(args.func, hackytrack.detect_footbag_batch)
        self.assertEqual(args.every_nth_frame, 1)
        self.assertEqual(args.patch_max_detections, 32)
        self.assertEqual(str(args.calibration_metrics), "detector_model_metrics.json")

    def test_detector_label_review_command_is_public(self) -> None:
        parser = hackytrack.build_parser()
        args = parser.parse_args(
            [
                "detector-label-review",
                "--qa-manifest",
                "qa_manifest.json",
                "--out-dir",
                "detector_review",
                "--max-items",
                "24",
            ]
        )
        self.assertEqual(args.func, hackytrack.detector_label_review)
        self.assertEqual(args.max_items, 24)

    def test_export_detector_dataset_accepts_multiple_review_pairs(self) -> None:
        parser = hackytrack.build_parser()
        args = parser.parse_args(
            [
                "export-detector-dataset",
                "--qa-manifest",
                "qa_manifest.json",
                "--reviews-dir",
                "reviews",
                "--out-dir",
                "dataset",
                "--detector-label-review-pair",
                "review_a.json:decisions_a.json",
                "--detector-label-review-pair",
                "review_b.json:decisions_b.json",
            ]
        )
        self.assertEqual(args.func, hackytrack.export_detector_dataset)
        self.assertEqual(args.detector_label_review_pair, ["review_a.json:decisions_a.json", "review_b.json:decisions_b.json"])

    def test_assist_detector_labels_command_is_public(self) -> None:
        parser = hackytrack.build_parser()
        args = parser.parse_args(
            [
                "assist-detector-labels",
                "--review-manifest",
                "detector_label_review_manifest.json",
                "--out",
                "assisted.json",
            ]
        )
        self.assertEqual(args.func, hackytrack.assist_detector_labels)
        self.assertEqual(args.min_confidence, 0.70)

    def test_patch_detector_commands_are_public(self) -> None:
        parser = hackytrack.build_parser()
        train_args = parser.parse_args(
            [
                "train-patch-detector",
                "--dataset",
                "detector_dataset",
                "--out-dir",
                "patch_detector",
                "--model-kind",
                "extra-trees",
                "--dry-run",
            ]
        )
        self.assertEqual(train_args.func, hackytrack.train_patch_detector)
        self.assertEqual(train_args.crop_size, 96)
        self.assertEqual(train_args.model_kind, "extra-trees")

        eval_args = parser.parse_args(
            [
                "evaluate-patch-detector",
                "--dataset",
                "detector_dataset",
                "--model",
                "patch_detector.joblib",
                "--out-dir",
                "patch_eval",
                "--max-detections",
                "32",
            ]
        )
        self.assertEqual(eval_args.func, hackytrack.evaluate_patch_detector)
        self.assertEqual(eval_args.max_detections, 32)

    def test_detector_false_positive_review_command_is_public(self) -> None:
        parser = hackytrack.build_parser()
        args = parser.parse_args(
            [
                "detector-false-positive-review",
                "--qa-manifest",
                "qa_manifest.json",
                "--inference-root",
                "detector_inference",
                "--out-dir",
                "false_positive_review",
                "--min-confidence",
                "0.5",
                "--max-confidence",
                "1.0",
            ]
        )
        self.assertEqual(args.func, hackytrack.detector_false_positive_review)
        self.assertEqual(args.per_video, 24)
        self.assertEqual(args.min_confidence, 0.5)
        self.assertEqual(args.max_confidence, 1.0)

    def test_detector_error_review_command_is_public(self) -> None:
        parser = hackytrack.build_parser()
        args = parser.parse_args(
            [
                "detector-error-review",
                "--qa-manifest",
                "qa_manifest.json",
                "--track-metrics",
                "track_metrics.json",
                "--dataset",
                "detector_dataset",
                "--model-metrics",
                "model_metrics.json",
                "--out-dir",
                "error_review",
                "--low-confidence-per-split",
                "3",
                "--exclude-audit-only",
            ]
        )
        self.assertEqual(args.func, hackytrack.detector_error_review)
        self.assertEqual(args.per_video, 12)
        self.assertEqual(args.low_confidence_per_split, 3)
        self.assertTrue(args.exclude_audit_only)

    def test_dense_trajectory_commands_are_public(self) -> None:
        parser = hackytrack.build_parser()
        review_args = parser.parse_args(
            [
                "dense-trajectory-review",
                "--qa-manifest",
                "qa_manifest.json",
                "--track-metrics",
                "v10_metrics.json",
                "--batch-summary",
                "v11_summary.json",
                "--target-video",
                "video-352_singular_display 2.MOV",
                "--track-root",
                "v10:tracks",
                "--out-dir",
                "dense_review",
                "--max-clips",
                "6",
                "--dry-run",
            ]
        )
        self.assertEqual(review_args.func, hackytrack.dense_trajectory_review)
        self.assertEqual(review_args.max_clips, 6)
        self.assertEqual(review_args.target_video, ["video-352_singular_display 2.MOV"])
        self.assertEqual(review_args.track_root, ["v10:tracks"])

        eval_args = parser.parse_args(
            [
                "evaluate-dense-trajectory",
                "--labels-jsonl",
                "labels.jsonl",
                "--predictions-jsonl",
                "predictions.jsonl",
                "--out-dir",
                "dense_eval",
            ]
        )
        self.assertEqual(eval_args.func, hackytrack.evaluate_dense_trajectory)
        self.assertEqual(eval_args.tolerance_px, 12.0)

    def test_evaluate_detector_model_exposes_calibration_options(self) -> None:
        parser = hackytrack.build_parser()
        args = parser.parse_args(
            [
                "evaluate-detector-model",
                "--dataset",
                "detector_dataset",
                "--model",
                "model.pt",
                "--out-dir",
                "model_eval",
                "--calibration-split",
                "test",
                "--max-hard-negative-false-positive-rate",
                "0.05",
            ]
        )
        self.assertEqual(args.func, hackytrack.evaluate_detector_model)
        self.assertEqual(args.calibration_split, "test")
        self.assertEqual(args.max_hard_negative_false_positive_rate, 0.05)

    def test_summarize_detector_batch_command_is_public(self) -> None:
        parser = hackytrack.build_parser()
        args = parser.parse_args(
            [
                "summarize-detector-batch",
                "--tracks-root",
                "detector_inference",
                "--out-dir",
                "detector_summary",
                "--high-prediction-share",
                "0.4",
            ]
        )
        self.assertEqual(args.func, hackytrack.summarize_detector_batch)
        self.assertEqual(str(args.tracks_root), "detector_inference")
        self.assertEqual(args.high_prediction_share, 0.4)


if __name__ == "__main__":
    unittest.main()
