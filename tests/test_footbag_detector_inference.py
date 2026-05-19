from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import footbag_detector_inference


def detection(frame: int, x: float, y: float, confidence: float = 0.9) -> dict:
    return {
        "frame_index": frame,
        "bbox": [x - 5.0, y - 5.0, x + 5.0, y + 5.0],
        "center": [x, y],
        "confidence": confidence,
        "class_name": "footbag",
    }


class FootbagDetectorInferenceTests(unittest.TestCase):
    def test_bbox_filter_rejects_implausible_footbag_boxes(self) -> None:
        self.assertIsNone(
            footbag_detector_inference.bbox_filter_reason(
                [100, 100, 140, 140],
                frame_width=688,
                frame_height=912,
            )
        )
        self.assertEqual(
            footbag_detector_inference.bbox_filter_reason(
                [0, 0, 300, 320],
                frame_width=688,
                frame_height=912,
            ),
            "box_too_large_for_footbag",
        )
        self.assertEqual(
            footbag_detector_inference.bbox_filter_reason(
                [100, 100, 180, 108],
                frame_width=688,
                frame_height=912,
            ),
            "box_aspect_ratio_unlikely",
        )

    def test_tracker_keeps_stable_id_and_predicts_missing_frame(self) -> None:
        track = footbag_detector_inference.track_detections(
            [
                detection(0, 100, 100),
                detection(1, 111, 99),
                detection(3, 130, 102),
            ],
            fps=30.0,
            smoothing_alpha=1.0,
            max_gap_frames=3,
            max_jump_px=80.0,
        )
        self.assertEqual([item["frame_index"] for item in track], [0, 1, 2, 3])
        self.assertEqual({item["track_id"] for item in track}, {1})
        predicted = track[2]
        self.assertEqual(predicted["source"], "track_predicted")
        self.assertIn("missing_detection", predicted["uncertainty_reasons"])
        self.assertEqual(predicted["center"], [122.0, 98.0])

    def test_tracker_starts_new_track_for_unreasonable_jump(self) -> None:
        track = footbag_detector_inference.track_detections(
            [
                detection(0, 10, 10),
                detection(1, 12, 10),
                detection(2, 500, 500),
            ],
            smoothing_alpha=1.0,
            max_gap_frames=2,
            max_jump_px=50.0,
        )
        self.assertEqual([item["track_id"] for item in track], [1, 1, 2])
        self.assertIn("new_track_large_jump", track[-1]["uncertainty_reasons"])

    def test_temporal_tracker_prefers_consistent_path_over_isolated_high_confidence(self) -> None:
        detections = [
            detection(0, 100, 100, 0.70),
            detection(0, 500, 500, 0.99),
            detection(1, 110, 100, 0.70),
            detection(1, 20, 500, 0.99),
            detection(2, 120, 100, 0.70),
            detection(2, 500, 20, 0.99),
        ]
        track = footbag_detector_inference.track_detections(
            detections,
            smoothing_alpha=1.0,
            max_gap_frames=2,
            max_jump_px=50.0,
            tracker_mode="temporal",
        )
        self.assertEqual([item["frame_index"] for item in track], [0, 1, 2])
        self.assertEqual([item["center"] for item in track], [[100.0, 100.0], [110.0, 100.0], [120.0, 100.0]])
        self.assertTrue(all("temporal_path" in item["uncertainty_reasons"] for item in track))

    def test_fixture_inference_writes_track_manifest_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "detections.jsonl"
            fixture.write_text(
                "\n".join(
                    [
                        json.dumps(detection(0, 100, 100, 0.92)),
                        json.dumps(detection(2, 120, 102, 0.88)),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            out = root / "out"
            summary = footbag_detector_inference.run_detector_inference(
                video=None,
                model=None,
                detections_jsonl=fixture,
                out_dir=out,
                smoothing_alpha=1.0,
                max_gap_frames=3,
                fps_override=30.0,
            )
            self.assertEqual(summary["source"], "detections_jsonl")
            self.assertEqual(summary["counts"]["raw_detections"], 2)
            self.assertEqual(summary["counts"]["track_points"], 3)
            self.assertEqual(summary["counts"]["predicted_track_points"], 1)
            self.assertTrue((out / "detector_track.json").exists())
            self.assertTrue((out / "detector_track.csv").exists())
            self.assertTrue((out / "detector_inference_manifest.json").exists())
            track_doc = json.loads((out / "detector_track.json").read_text(encoding="utf-8"))
            self.assertEqual(track_doc["track"][1]["source"], "track_predicted")
            self.assertEqual(track_doc["track"][1]["time_sec"], 0.033333)
            csv_text = (out / "detector_track.csv").read_text(encoding="utf-8")
            self.assertIn("frame_index,time_sec,track_id,source", csv_text)

    def test_dry_run_does_not_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            summary = footbag_detector_inference.run_detector_inference(
                video=None,
                model=None,
                out_dir=out,
                detections_jsonl=Path(tmp) / "missing.jsonl",
                dry_run=True,
            )
            self.assertTrue(summary["dry_run"])
            self.assertFalse(out.exists())

    def test_dry_run_records_processed_coordinate_space(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            summary = footbag_detector_inference.run_detector_inference(
                video=Path(tmp) / "clip.MOV",
                model=None,
                patch_model=Path(tmp) / "patch.joblib",
                out_dir=out,
                process_width=688,
                process_height=912,
                dry_run=True,
            )
            self.assertEqual(summary["patch_model"], "patch.joblib")
            self.assertEqual(summary["parameters"]["process_width"], 688)
            self.assertEqual(summary["parameters"]["process_height"], 912)

    def test_calibration_metrics_override_confidence_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "detector_model_metrics.json"
            metrics.write_text(
                json.dumps({"threshold_recommendation": {"recommended_threshold": 0.0125}}),
                encoding="utf-8",
            )
            summary = footbag_detector_inference.run_detector_inference(
                video=None,
                model=None,
                detections_jsonl=root / "missing.jsonl",
                out_dir=root / "out",
                confidence_threshold=0.25,
                calibration_metrics=metrics,
                dry_run=True,
            )
            self.assertEqual(summary["parameters"]["confidence_threshold"], 0.0125)
            self.assertEqual(summary["parameters"]["confidence_threshold_source"], "calibration_metrics")
            self.assertEqual(summary["parameters"]["calibration_metrics"], "detector_model_metrics.json")

    def test_fps_override_allows_fixture_with_placeholder_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "detections.jsonl"
            fixture.write_text(json.dumps(detection(0, 100, 100, 0.92)) + "\n", encoding="utf-8")
            placeholder_video = root / "clip.MOV"
            placeholder_video.write_bytes(b"not a real movie")
            summary = footbag_detector_inference.run_detector_inference(
                video=placeholder_video,
                model=None,
                detections_jsonl=fixture,
                out_dir=root / "out",
                fps_override=30.0,
            )
            self.assertEqual(summary["video_info"]["source"], "fps_override")
            self.assertEqual(summary["counts"]["track_points"], 1)


if __name__ == "__main__":
    unittest.main()
