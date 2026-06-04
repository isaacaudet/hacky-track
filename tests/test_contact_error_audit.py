from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import contact_error_audit


class ContactErrorAuditTests(unittest.TestCase):
    def test_report_includes_sequence_smoothed_side_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.md"
            summary = {
                "status": "complete",
                "strip_dir": "/tmp/raw-strips",
            }
            raw_errors = [
                {
                    "target": "contact_side",
                    "video_id": "video-a",
                    "candidate_time_sec": 1.0,
                    "label": "left",
                    "prediction": "right",
                    "failure_bucket": "pose_side_disagreement",
                }
            ]
            sequence_errors = [
                {
                    "video_id": "video-a",
                    "candidate_time_sec": 2.0,
                    "label": "right",
                    "prediction": "left",
                    "raw_prediction": "right",
                    "failure_bucket": "side_visual_ambiguity",
                    "strip_path": "/tmp/sequence-strip.jpg",
                }
            ]

            contact_error_audit.write_report(report_path, summary, raw_errors, sequence_errors)

            text = report_path.read_text(encoding="utf-8")
            self.assertIn("Sequence-Smoothed Side Errors", text)
            self.assertIn("diagnostic_only", text)
            self.assertIn("side_visual_ambiguity", text)
            self.assertIn("raw pred", text)


if __name__ == "__main__":
    unittest.main()
