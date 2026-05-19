from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import apply_detector_track_to_qa


class ApplyDetectorTrackToQaTests(unittest.TestCase):
    def test_apply_track_attaches_model_evidence_and_promotes_matches(self) -> None:
        qa_doc = {
            "source_video": "/videos/clip.MOV",
            "events": [
                {
                    "type": "touch",
                    "time_sec": 1.0,
                    "qa_frame_time_sec": 1.0,
                    "x": 90.0,
                    "y": 90.0,
                    "qa_ball_x": 100.0,
                    "qa_ball_y": 100.0,
                    "qa_ball_confidence": 0.5,
                    "qa_ball_source": "local_red_snap",
                    "qa_ball_accuracy": "medium",
                },
                {
                    "type": "touch",
                    "time_sec": 2.0,
                    "qa_ball_x": 10.0,
                    "qa_ball_y": 10.0,
                    "qa_ball_confidence": 0.6,
                    "qa_ball_source": "local_red_snap",
                },
            ],
        }
        track_doc = {
            "track": [
                {"frame_index": 30, "time_sec": 1.02, "source": "model_detection", "center": [105.0, 102.0], "confidence": 0.91, "uncertainty_reasons": []},
                {"frame_index": 60, "time_sec": 2.01, "source": "model_detection", "center": [300.0, 300.0], "confidence": 0.87, "uncertainty_reasons": []},
            ]
        }
        out_doc, counts = apply_detector_track_to_qa.apply_track_to_doc(
            qa_doc,
            track_doc,
            max_time_delta_sec=0.05,
            max_center_delta_px=30.0,
            promote_model=True,
        )
        self.assertEqual(counts["matched"], 2)
        self.assertEqual(counts["promoted"], 1)
        self.assertEqual(counts["mismatched"], 1)
        first = out_doc["events"][0]
        self.assertEqual(first["qa_ball_source"], "model_detector")
        self.assertEqual(first["qa_ball_x"], 105.0)
        self.assertEqual(first["heuristic_ball_evidence"]["qa_ball_source"], "local_red_snap")
        self.assertEqual(first["model_ball_agreement"], "match")
        second = out_doc["events"][1]
        self.assertEqual(second["qa_ball_source"], "local_red_snap")
        self.assertEqual(second["model_ball_agreement"], "mismatch")
        self.assertEqual(second["model_ball_review_reason"], "model_heuristic_center_mismatch")

    def test_manifest_mode_writes_updated_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_dir = root / "qa" / "clip"
            qa_dir.mkdir(parents=True)
            qa_events = qa_dir / "qa_events.json"
            qa_events.write_text(
                json.dumps(
                    {
                        "source_video": "/videos/clip.MOV",
                        "events": [{"type": "touch", "time_sec": 1.0, "qa_ball_x": 100.0, "qa_ball_y": 100.0}],
                    }
                ),
                encoding="utf-8",
            )
            qa_manifest = root / "qa" / "qa_manifest.json"
            qa_manifest.write_text(json.dumps({"runs": [{"qa_events_path": str(qa_events)}]}), encoding="utf-8")
            track_dir = root / "tracks" / "clip"
            track_dir.mkdir(parents=True)
            (track_dir / "detector_track.json").write_text(
                json.dumps({"track": [{"frame_index": 30, "time_sec": 1.0, "source": "model_detection", "center": [101.0, 99.0], "confidence": 0.9}]}),
                encoding="utf-8",
            )
            out = root / "qa_model"
            summary = apply_detector_track_to_qa.apply_manifest(
                qa_manifest=qa_manifest,
                tracks_root=root / "tracks",
                out_root=out,
                max_time_delta_sec=0.08,
                max_center_delta_px=36.0,
                promote_model=False,
                allow_predicted=False,
            )
            self.assertEqual(summary["counts"]["matched"], 1)
            self.assertTrue((out / "qa_manifest.json").exists())
            run = summary["runs"][0]
            self.assertEqual(run["detector_track_status"], "applied")
            updated = json.loads((out / "clip" / "qa_events.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["events"][0]["model_ball_source"], "model_detector")


if __name__ == "__main__":
    unittest.main()
