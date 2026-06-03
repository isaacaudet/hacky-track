from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import hackytrack


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class HackyTrackTouchReleaseTests(unittest.TestCase):
    def test_ordered_jsonl_video_ids_preserves_first_seen_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            write_jsonl(
                path,
                [
                    {"video_id": "video-b"},
                    {"video_id": "video-a"},
                    {"video_id": "video-b"},
                    {"video_id": ""},
                ],
            )

            self.assertEqual(hackytrack.ordered_jsonl_video_ids(path), ["video-b", "video-a"])

    def test_release_gate_passed_requires_touch_and_hud_gates(self) -> None:
        summary = {
            "touch_classifier": {"cv_gate_passed": True, "frozen_gate_passed": True},
            "hud_model_only": {"status": "passed"},
            "hud_visual_corrected": {"status": "passed"},
        }

        self.assertTrue(hackytrack.release_gate_passed(summary))

        summary["hud_visual_corrected"]["status"] = "failed"
        self.assertFalse(hackytrack.release_gate_passed(summary))

    def test_write_touch_release_report_marks_contact_type_as_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "touch_release_readiness.md"
            summary = {
                "status": "release_ready_v0_1",
                "scope": "v0.1_touch_hud_release",
                "created_at": "now",
                "run_dir": "runs/touch-release-test",
                "commands": [{"name": "model-only release HUD", "command": "python3 render_touch_release_hud.py"}],
                "touch_classifier": {
                    "status": "trained",
                    "feature_mode": "fused_audio_trajectory",
                    "cv_gate_passed": True,
                    "frozen_gate_passed": True,
                    "cv_event": {"precision": 0.91, "recall": 0.89, "f1": 0.90, "false_positive": 7, "false_negative": 9},
                    "frozen_event": {"precision": 0.99, "recall": 0.99, "f1": 0.99, "false_positive": 2, "false_negative": 1},
                },
                "hud_model_only": {
                    "status": "passed",
                    "videos": 6,
                    "preview_sheet": "preview.jpg",
                    "report": "hud.md",
                    "analytics_report": "analytics.md",
                    "analytics": {"precision": 1.0, "recall": 0.99, "f1": 0.996, "false_positive": 0, "false_negative": 1},
                },
                "hud_visual_corrected": {},
                "contact_classifier": {
                    "status": "not_ready",
                    "rows_with_pose_features": 0,
                    "rows_with_contact_labels": 0,
                    "report": "contact.md",
                    "reasons": ["need reviewed contact labels"],
                },
                "limitations": ["generic touches only"],
            }

            hackytrack.write_touch_release_report(path, summary)

            text = path.read_text(encoding="utf-8")
            self.assertIn("release_ready_v0_1", text)
            self.assertIn("need reviewed contact labels", text)
            self.assertIn("generic touches only", text)


if __name__ == "__main__":
    unittest.main()
