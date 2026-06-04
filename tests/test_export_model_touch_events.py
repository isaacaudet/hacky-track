from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import export_model_touch_events as exporter


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


class ExportModelTouchEventsTests(unittest.TestCase):
    def test_existing_stream_and_reset_ids_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            resets = root / "resets.jsonl"
            write_jsonl(events, [{"video_name": "video-230_singular_display 2.MOV", "event_type": "touch", "time_sec": 1.0}])
            write_jsonl(resets, [{"video_id": "video-230_singular_display-2", "kind": "drop_floor"}, {"video_id": "video-498_singular_display", "kind": "stall"}])

            self.assertEqual(exporter.event_stream_video_ids([events]), {"video-230_singular_display-2"})
            self.assertEqual(exporter.reset_review_video_ids(resets), {"video-230_singular_display-2", "video-498_singular_display"})

    def test_build_unreviewed_candidate_rows_preserves_model_only_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates_dir = root / "candidates"
            write_json(
                candidates_dir / "video-a.touch_candidates.json",
                {
                    "audio_candidates": [
                        {"time_sec": 1.0, "strength": 2.5, "source": "loose_audio_onset"},
                        {"time_sec": 1.08, "strength": 3.5, "source": "loose_audio_onset"},
                    ],
                    "existing_event_hints": [{"time_sec": 2.0, "event_type": "touch", "source": "legacy_hint"}],
                },
            )

            train, test, videos = exporter.build_unreviewed_candidate_rows(
                manifest_items=[{"video_id": "video-a", "video_name": "video-a.MOV", "split": "train"}],
                selected_video_ids={"video-a"},
                candidates_dir=candidates_dir,
                cluster_gap_sec=0.18,
            )

            self.assertEqual(len(train), 2)
            self.assertEqual(test, [])
            self.assertEqual(videos[0]["status"], "selected")
            self.assertTrue(all(row["model_only_stream"] for row in train))
            self.assertTrue(all(row["label_is_touch"] is False for row in train))
            self.assertTrue(all(row["candidate_review_source"] == "model_only_unreviewed" for row in train))
            self.assertAlmostEqual(train[0]["candidate_time_sec"], 1.08)

    def test_retag_model_only_events_removes_release_truth_claim(self) -> None:
        rows = [
            {
                "video_id": "video-a",
                "event_type": "touch",
                "time_sec": 1.0,
                "event_match_type": "false_positive",
                "matched_truth_time_sec": 1.1,
                "matched_truth_delta_sec": 0.1,
            }
        ]

        out = exporter.retag_model_only_events(rows)

        self.assertEqual(out[0]["event_match_type"], "model_only_unreviewed")
        self.assertIsNone(out[0]["matched_truth_time_sec"])
        self.assertIsNone(out[0]["matched_truth_delta_sec"])
        self.assertEqual(out[0]["truth_source"], "none_unreviewed_model_stream")
        self.assertTrue(out[0]["model_only_stream"])


if __name__ == "__main__":
    unittest.main()
