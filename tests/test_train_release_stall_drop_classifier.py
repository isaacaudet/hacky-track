from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import train_release_stall_drop_classifier as stall_drop


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


class ReleaseStallDropClassifierTests(unittest.TestCase):
    def test_default_detection_inputs_include_stall_drop_supplemental_cache(self) -> None:
        names = [path.as_posix() for path in stall_drop.DEFAULT_DETECTIONS_JSONL]

        self.assertTrue(any("owlv2_stall_drop_missing_detections_v1" in name for name in names))

    def test_review_rows_drop_conflicts_and_merge_same_status_duplicates(self) -> None:
        rows = [
            {
                "kind": "drop_floor",
                "status": "approved",
                "source_video": "video-a_singular_display.MOV",
                "time_sec": 1.000,
            },
            {
                "kind": "drop_floor",
                "status": "rejected",
                "source_video": "video-a_singular_display.MOV",
                "time_sec": 1.018,
            },
            {
                "kind": "stall",
                "status": "rejected",
                "source_video": "video-b_singular_display.MOV",
                "time_sec": 2.000,
            },
            {
                "kind": "stall",
                "status": "rejected",
                "source_video": "video-b_singular_display.MOV",
                "time_sec": 2.021,
                "confidence": 0.9,
            },
        ]

        cleaned, summary = stall_drop.prepare_review_rows(rows, bin_sec=0.05)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["kind"], "stall")
        self.assertAlmostEqual(cleaned[0]["time_sec"], 2.021)
        self.assertEqual(summary["excluded_conflict_rows"], 2)
        self.assertEqual(summary["merged_duplicate_rows"], 1)

    def test_stream_compact_tracks_keeps_only_thresholded_top_point(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detections = root / "detections.jsonl"
            cache = root / "compact.jsonl"
            write_jsonl(
                detections,
                [
                    {
                        "source_video": "video-a.MOV",
                        "video_id": "video-a",
                        "frame_index": 1,
                        "time_sec": 0.1,
                        "top_score": 0.7,
                        "detections": [
                            {"score": 0.3, "x": 1, "y": 1},
                            {"score": 0.7, "x": 4, "y": 5},
                        ],
                    },
                    {
                        "source_video": "video-a.MOV",
                        "video_id": "video-a",
                        "frame_index": 2,
                        "time_sec": 0.2,
                        "top_score": 0.19,
                        "detections": [{"score": 0.19, "x": 9, "y": 9}],
                    },
                    {
                        "source_video": "video-b.MOV",
                        "video_id": "video-b",
                        "frame_index": 1,
                        "time_sec": 0.1,
                        "x": 8,
                        "y": 9,
                        "confidence": 0.4,
                    },
                ],
            )

            manifest = stall_drop.build_compact_track_cache([detections], cache, threshold=0.2)
            tracks = stall_drop.load_compact_tracks(cache)

            self.assertEqual(manifest["input_rows"], 3)
            self.assertEqual(manifest["kept_points"], 2)
            self.assertEqual(sorted(tracks), ["video-a.MOV", "video-b.MOV"])
            self.assertEqual(tracks["video-a.MOV"][0].x, 4)
            self.assertEqual(tracks["video-a.MOV"][0].confidence, 0.7)

    def test_target_training_reports_stall_not_ready_when_positive_floor_is_missing(self) -> None:
        rows = [
            {"kind": "stall", "label": 1, "video_id": "video-a", "candidate_time_sec": 1.0, "trajectory_impulse_score": 10.0},
            {"kind": "stall", "label": 0, "video_id": "video-b", "candidate_time_sec": 2.0, "trajectory_impulse_score": 20.0},
            {"kind": "stall", "label": 0, "video_id": "video-c", "candidate_time_sec": 3.0, "trajectory_impulse_score": 30.0},
        ]

        result, model = stall_drop.train_target(rows, "stall", min_examples=2, min_videos=2, min_positive_examples=2)

        self.assertIsNone(model)
        self.assertEqual(result["status"], "not_ready")
        self.assertTrue(any("approved" in reason for reason in result["reasons"]))

    def test_sequence_window_features_measure_low_screen_track_context(self) -> None:
        row = {
            "kind": "drop_floor",
            "label": 1,
            "video_id": "video-a",
            "video_name": "video-a.MOV",
            "candidate_time_sec": 1.0,
        }
        points = [
            stall_drop.TrackPoint(time_sec=0.90, x=100, y=910, confidence=0.7, frame_index=27),
            stall_drop.TrackPoint(time_sec=1.00, x=105, y=920, confidence=0.8, frame_index=30),
            stall_drop.TrackPoint(time_sec=1.10, x=110, y=930, confidence=0.9, frame_index=33),
            stall_drop.TrackPoint(time_sec=2.00, x=300, y=200, confidence=0.9, frame_index=60),
        ]

        out = stall_drop.add_sequence_window_features(
            row,
            points,
            video_path=None,
            width=1000,
            height=1000,
            window_sec=0.5,
        )

        self.assertAlmostEqual(out["candidate_y_ratio"], 0.92)
        self.assertEqual(out["sequence_track_count_window"], 3)
        self.assertEqual(out["sequence_pre_track_count"], 1)
        self.assertEqual(out["sequence_post_track_count"], 1)
        self.assertGreater(out["sequence_track_coverage_ratio"], 0.0)
        self.assertEqual(out["sequence_low_screen_ratio_window"], 1.0)
        self.assertGreater(out["sequence_mean_speed_px_sec"], 0.0)

    def test_l2_only_feature_mode_excludes_sequence_and_floor_context(self) -> None:
        row = {
            "trajectory_impulse_score": 4.0,
            "sequence_track_count_window": 12,
            "floor_context_score": 0.5,
            "candidate_y_ratio": 0.9,
            "event_next_touch_gap_sec": 1.2,
        }

        features = stall_drop.feature_dict(row, disabled_prefixes=stall_drop.FEATURE_MODES["l2_only"])

        self.assertIn("trajectory_impulse_score", features)
        self.assertNotIn("sequence_track_count_window", features)
        self.assertNotIn("floor_context_score", features)
        self.assertNotIn("candidate_y_ratio", features)
        self.assertNotIn("event_next_touch_gap_sec", features)

    def test_touch_event_stream_features_use_normalized_video_lookup(self) -> None:
        row = {
            "video_id": "video-230_singular_display-2",
            "video_name": "video-230_singular_display 2.MOV",
            "candidate_time_sec": 10.0,
        }
        streams = {
            "video-230_singular_display-2": [8.5, 11.25],
        }

        out = stall_drop.add_touch_event_stream_features(row, streams)

        self.assertEqual(out["event_touch_stream_present"], 1.0)
        self.assertAlmostEqual(out["event_prev_touch_gap_sec"], 1.5)
        self.assertAlmostEqual(out["event_next_touch_gap_sec"], 1.25)
        self.assertEqual(out["event_post_gap_gt_1s"], 1.0)
        self.assertEqual(out["event_pre_gap_gt_1s"], 1.0)

    def test_load_touch_event_streams_indexes_video_name_and_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            write_jsonl(
                events,
                [
                    {
                        "event_type": "touch",
                        "video_id": "video-a_singular_display",
                        "video_name": "video-a_singular_display.MOV",
                        "time_sec": 1.25,
                    }
                ],
            )

            streams = stall_drop.load_touch_event_streams([events])

            self.assertEqual(streams["video-a_singular_display"], [1.25])
            self.assertEqual(streams["video-a_singular_display.MOV"], [1.25])

    def test_score_threshold_sweep_reports_precision_recall_tradeoff(self) -> None:
        predictions = [
            {"label": "approved", "positive_score": 0.9},
            {"label": "approved", "positive_score": 0.4},
            {"label": "rejected", "positive_score": 0.8},
            {"label": "rejected", "positive_score": 0.3},
        ]

        rows = stall_drop.score_threshold_sweep(predictions, thresholds=(0.35, 0.85))
        best = stall_drop.best_threshold_row(rows)

        self.assertEqual(rows[0]["true_positive"], 2)
        self.assertEqual(rows[0]["false_positive"], 1)
        self.assertAlmostEqual(rows[0]["recall"], 1.0)
        self.assertEqual(rows[1]["true_positive"], 1)
        self.assertEqual(rows[1]["false_positive"], 0)
        self.assertAlmostEqual(rows[1]["precision"], 1.0)
        self.assertEqual(best["threshold"], 0.35)


if __name__ == "__main__":
    unittest.main()
