from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import release_candidate_report


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class ReleaseCandidateReportTests(unittest.TestCase):
    def test_strict_audit_blocks_hud_when_no_strict_rally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_json(run_dir / "run_manifest.json", {"artifacts": {}})
            for path in [
                "summary.md",
                "events.json",
                "events.csv",
                "rallies.json",
                "training/full_training_manifest.json",
                "review_batches/latest_review_batch.json",
                "ball_tracking_audit/audit_metrics.json",
                "validation/validation_metrics.json",
                "release_evaluation/release_metrics.json",
                "portable_paths_audit.json",
            ]:
                full = run_dir / path
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text("{}" if path.endswith(".json") else "", encoding="utf-8")
            (run_dir / "models").mkdir()
            (run_dir / "models" / "footbag_patch_hgb.joblib").write_text("model", encoding="utf-8")
            write_json(run_dir / "models" / "training_report.json", {})
            qa_events = run_dir / "qa" / "video-352_singular_display 2" / "qa_events.json"
            write_json(
                qa_events,
                {
                    "source_video": "video-352_singular_display 2.MOV",
                    "events": [{"type": "drop_floor", "time_sec": 22.9, "qa_rally_id": 1}],
                    "rallies": [],
                },
            )
            write_json(
                run_dir / "qa" / "qa_manifest.json",
                {
                    "runs": [
                        {
                            "video": "video-352_singular_display 2.MOV",
                            "qa_events_path": str(qa_events),
                        }
                        for _ in range(27)
                    ]
                },
            )
            write_json(
                run_dir / "strict_rally_audit" / "strict_rally_audit.json",
                {"summary": {"rallies": 10, "strict_complete_rallies": 0}, "top_rejected_candidates": []},
            )
            write_json(
                run_dir / "review_batches" / "latest_review_batch.json",
                {"summary": {"source_counts": {"candidate": 1, "active_learning": 1}, "tag_counts": {"gap_without_floor_reset": 1}}},
            )
            write_json(run_dir / "portable_paths_audit.json", {"passed": True, "sanitized_files": []})
            write_json(run_dir / "release_evaluation" / "release_metrics.json", {"target_gates": []})

            doc = release_candidate_report.build_report(run_dir, tests_passed=True, test_evidence="unit test")
            hud_gate = next(gate for gate in doc["gates"] if gate["name"] == "HUD MP4 technical verification")
            strict_gate = next(gate for gate in doc["gates"] if gate["name"] == "Strict best-rally eligibility")
            self.assertEqual(strict_gate["status"], "blocked")
            self.assertEqual(hud_gate["status"], "blocked")
            self.assertFalse(doc["goal_complete"])

    def test_candidate_only_release_guard_is_guarded_not_blocking(self) -> None:
        metrics = {
            "target_gates": [
                {
                    "name": "knee release guard",
                    "status": "candidate_only",
                    "value": {
                        "reviewed_examples": 0,
                        "precision": None,
                        "release_state": "candidate_only",
                    },
                    "target": {"min_examples": 20, "precision": 0.8},
                }
            ]
        }
        gates = release_candidate_report.release_metric_gates(metrics)
        self.assertEqual(gates[0].status, "guarded")
        self.assertIn("Candidate-only guard is active", gates[0].next_step)

    def test_strict_gate_blocks_when_better_review_required_rally_outranks_strict_pick(self) -> None:
        strict_audit = {
            "summary": {"rallies": 3, "strict_complete_rallies": 1},
            "top_strict_candidates": [
                {"source_video": "small.MOV", "rally_id": 1, "touches": 8, "quality_score": 78.0}
            ],
            "top_rejected_candidates": [
                {
                    "source_video": "better.MOV",
                    "rally_id": 2,
                    "touches": 11,
                    "quality_score": 90.0,
                    "rejection_reasons": ["ended_by_gap_without_floor_reset"],
                }
            ],
        }
        gate = release_candidate_report.strict_rally_gate(strict_audit)
        self.assertEqual(gate.status, "blocked")
        self.assertIn("better.MOV", gate.evidence)
        self.assertIn("review-required", gate.evidence)

        hud_gate = release_candidate_report.hud_gate({"streams": ["0|video|", "1|audio|"]}, strict_audit)
        self.assertEqual(hud_gate.status, "blocked")
        self.assertIn("best-rally selection is unresolved", hud_gate.evidence)

    def test_report_uses_reviewed_strict_audit_for_selection_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_json(run_dir / "run_manifest.json", {"artifacts": {}})
            for path in [
                "summary.md",
                "events.json",
                "events.csv",
                "rallies.json",
                "training/full_training_manifest.json",
                "review_batches/latest_review_batch.json",
                "ball_tracking_audit/audit_metrics.json",
                "validation/validation_metrics.json",
                "release_evaluation/release_metrics.json",
                "portable_paths_audit.json",
            ]:
                full = run_dir / path
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text("{}" if path.endswith(".json") else "", encoding="utf-8")
            (run_dir / "models").mkdir()
            (run_dir / "models" / "footbag_patch_hgb.joblib").write_text("model", encoding="utf-8")
            write_json(run_dir / "models" / "training_report.json", {})
            qa_events = run_dir / "qa" / "video-352_singular_display 2" / "qa_events.json"
            write_json(
                qa_events,
                {
                    "source_video": "video-352_singular_display 2.MOV",
                    "events": [{"type": "drop_floor", "time_sec": 22.9, "qa_rally_id": 1}],
                    "rallies": [],
                },
            )
            write_json(
                run_dir / "qa" / "qa_manifest.json",
                {
                    "runs": [
                        {
                            "video": "video-352_singular_display 2.MOV",
                            "qa_events_path": str(qa_events),
                        }
                        for _ in range(27)
                    ]
                },
            )
            write_json(
                run_dir / "strict_rally_audit" / "strict_rally_audit.json",
                {
                    "summary": {"rallies": 2, "strict_complete_rallies": 1},
                    "top_strict_candidates": [{"source_video": "small.MOV", "rally_id": 1, "touches": 8, "quality_score": 78.0}],
                    "top_rejected_candidates": [
                        {
                            "source_video": "better.MOV",
                            "rally_id": 2,
                            "touches": 11,
                            "quality_score": 90.0,
                            "rejection_reasons": ["ended_by_gap_without_floor_reset"],
                        }
                    ],
                },
            )
            write_json(
                run_dir / "qa_reviewed" / "qa_manifest.json",
                {"summary": {"manual_missing_inserted": 1}},
            )
            write_json(
                run_dir / "strict_rally_audit_reviewed" / "strict_rally_audit.json",
                {
                    "summary": {"rallies": 2, "strict_complete_rallies": 2},
                    "top_strict_candidates": [{"source_video": "better.MOV", "rally_id": 2, "touches": 11, "quality_score": 90.5}],
                    "top_rejected_candidates": [],
                },
            )
            write_json(
                run_dir / "review_batches" / "latest_review_batch.json",
                {"summary": {"source_counts": {"candidate": 1, "active_learning": 1}, "tag_counts": {"gap_without_floor_reset": 1}}},
            )
            write_json(run_dir / "portable_paths_audit.json", {"passed": True, "sanitized_files": []})
            write_json(run_dir / "release_evaluation" / "release_metrics.json", {"target_gates": []})
            write_json(run_dir / "hud" / "hud_verification.json", {"streams": ["mpeg4|video|", "aac|audio|"], "sampled_frame_means": [[0, 10.0]]})

            doc = release_candidate_report.build_report(run_dir, tests_passed=True, test_evidence="unit test")
            strict_gate = next(gate for gate in doc["gates"] if gate["name"] == "Strict best-rally eligibility")
            hud_gate = next(gate for gate in doc["gates"] if gate["name"] == "HUD MP4 technical verification")
            self.assertEqual(strict_gate["status"], "pass")
            self.assertEqual(hud_gate["status"], "pass")


if __name__ == "__main__":
    unittest.main()
