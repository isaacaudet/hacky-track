from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import strict_rally_audit


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class StrictRallyAuditTests(unittest.TestCase):
    def test_terminal_drop_boundary_is_not_a_rejection_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_events = root / "qa" / "clip" / "qa_events.json"
            events = [
                {
                    "type": "touch",
                    "time_sec": float(idx),
                    "qa_rally_id": 1,
                    "contact_type": "foot",
                    "qa_ball_accuracy": "high",
                }
                for idx in range(6)
            ]
            events.append(
                {
                    "type": "drop_floor",
                    "time_sec": 6.3,
                    "qa_rally_id": 1,
                    "contact_type": "ground",
                    "qa_ball_accuracy": "high",
                }
            )
            write_json(
                qa_events,
                {
                    "source_video": "clip.MOV",
                    "events": events,
                    "rallies": [
                        {
                            "id": 1,
                            "touches": 6,
                            "quality_score": 88.0,
                            "end_reason": "drop_floor",
                        }
                    ],
                },
            )
            manifest = root / "qa_manifest.json"
            write_json(manifest, {"runs": [{"video": "clip.MOV", "qa_events_path": str(qa_events)}]})

            audit = strict_rally_audit.audit_manifest(manifest)
            self.assertEqual(audit["summary"]["strict_complete_rallies"], 1)
            self.assertEqual(audit["top_strict_candidates"][0]["rejection_reasons"], [])

    def test_gap_without_floor_reset_rejects_otherwise_strict_rally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_events = root / "qa" / "clip" / "qa_events.json"
            events = [
                {
                    "type": "touch",
                    "time_sec": float(idx),
                    "qa_rally_id": 1,
                    "contact_type": "foot",
                    "qa_ball_accuracy": "high",
                }
                for idx in range(6)
            ]
            write_json(
                qa_events,
                {
                    "source_video": "clip.MOV",
                    "events": events,
                    "rallies": [
                        {
                            "id": 1,
                            "touches": 6,
                            "quality_score": 88.0,
                            "ended_by_gap_without_floor_reset": True,
                            "next_contact_gap_sec": 3.4,
                        }
                    ],
                },
            )
            manifest = root / "qa_manifest.json"
            write_json(manifest, {"runs": [{"video": "clip.MOV", "qa_events_path": str(qa_events)}]})

            audit = strict_rally_audit.audit_manifest(manifest)
            self.assertEqual(audit["summary"]["strict_complete_rallies"], 0)
            self.assertEqual(audit["summary"]["rejection_counts"]["ended_by_gap_without_floor_reset"], 1)
            self.assertIn("ended_by_gap_without_floor_reset", audit["top_rejected_candidates"][0]["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
