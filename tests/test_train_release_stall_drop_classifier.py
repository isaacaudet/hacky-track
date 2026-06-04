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


if __name__ == "__main__":
    unittest.main()
