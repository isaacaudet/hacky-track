import argparse
import json
import tempfile
import unittest
from pathlib import Path

import touch_label_readiness as readiness


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class TouchLabelReadinessTests(unittest.TestCase):
    def make_args(self, root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            review_manifest=root / "touch_review_manifest.json",
            candidates_dir=root / "audio_candidates",
            labels_dir=root / "visual_touch_labels",
            out_dir=root,
            review_match_tolerance_sec=0.05,
            candidate_cluster_gap_sec=0.04,
            min_non_test_videos=2,
            min_frozen_test_videos=1,
        )

    def write_manifest(self, root: Path) -> None:
        write_json(
            root / "touch_review_manifest.json",
            {
                "items": [
                    {"video_id": "train-a", "video_name": "train-a.MOV", "video_path": "/tmp/train-a.MOV", "split": "train"},
                    {"video_id": "train-b", "video_name": "train-b.MOV", "video_path": "/tmp/train-b.MOV", "split": "train"},
                    {"video_id": "test-a", "video_name": "test-a.MOV", "video_path": "/tmp/test-a.MOV", "split": "test_frozen"},
                ]
            },
        )

    def write_candidates(self, root: Path, video_id: str, times: list[float]) -> None:
        write_json(
            root / "audio_candidates" / f"{video_id}.touch_candidates.json",
            {
                "audio_candidates": [{"time_sec": time_sec, "strength": 1.0} for time_sec in times],
                "existing_event_hints": [],
            },
        )

    def write_label(
        self,
        root: Path,
        video_id: str,
        video_name: str,
        split: str,
        *,
        complete: bool,
        reviews: list[dict],
        touches: list[float] | None = None,
    ) -> None:
        write_json(
            root / "visual_touch_labels" / f"{video_id}.events.json",
            {
                "schema_version": 1,
                "source_video": video_name,
                "split": split,
                "annotation_method": "muted_visual_touch_review",
                "audio_muted_during_review_required": True,
                "candidate_review_complete": complete,
                "candidate_reviews": reviews,
                "rallies": [
                    {
                        "id": 1,
                        "start_sec": 0.0,
                        "end_sec": None,
                        "events": [
                            {"type": "touch", "time_sec": time_sec, "review_status": "approved"}
                            for time_sec in touches or []
                        ],
                    }
                ],
            },
        )

    def setup_basic(self, root: Path) -> None:
        self.write_manifest(root)
        self.write_candidates(root, "train-a", [1.0, 2.0])
        self.write_candidates(root, "train-b", [1.0])
        self.write_candidates(root, "test-a", [1.0])

    def test_classifies_missing_draft_unchecked_and_ready_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_basic(root)
            self.write_label(root, "train-a", "train-a.MOV", "train", complete=False, reviews=[{"time_sec": 1.0, "decision": "touch"}])
            self.write_label(root, "train-b", "train-b.MOV", "train", complete=True, reviews=[{"time_sec": 1.0, "decision": "no_touch"}])

            manifest = readiness.build_readiness_report(self.make_args(root))
            by_id = {row["video_id"]: row for row in manifest["videos"]}

            self.assertEqual(by_id["test-a"]["status"], "missing_label_file")
            self.assertEqual(by_id["test-a"]["unchecked_hint_count"], 1)
            self.assertEqual(by_id["test-a"]["audio_only_unchecked_hint_count"], 1)
            self.assertEqual(by_id["train-a"]["status"], "draft_incomplete")
            self.assertEqual(by_id["train-a"]["unchecked_hint_count"], 1)
            self.assertEqual(by_id["train-b"]["status"], "complete_ready")
            self.assertTrue(by_id["train-b"]["ready_for_training_table"])
            self.assertEqual(manifest["summary"]["complete_ready_videos"], 1)

    def test_complete_file_with_unchecked_hint_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_basic(root)
            self.write_label(root, "train-a", "train-a.MOV", "train", complete=True, reviews=[{"time_sec": 1.0, "decision": "touch"}])

            manifest = readiness.build_readiness_report(self.make_args(root))
            by_id = {row["video_id"]: row for row in manifest["videos"]}

            self.assertEqual(by_id["train-a"]["status"], "complete_but_unchecked_hints")
            self.assertFalse(by_id["train-a"]["ready_for_training_table"])
            self.assertEqual(by_id["train-a"]["checked_hint_count"], 1)
            self.assertEqual(by_id["train-a"]["unchecked_hint_count"], 1)

    def test_readiness_counts_clustered_candidate_moments_not_raw_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_candidates(root, "train-a", [1.0, 1.02, 2.0])
            self.write_candidates(root, "train-b", [1.0])
            self.write_candidates(root, "test-a", [1.0])
            self.write_label(
                root,
                "train-a",
                "train-a.MOV",
                "train",
                complete=True,
                reviews=[
                    {"time_sec": 1.02, "decision": "touch"},
                    {"time_sec": 2.0, "decision": "no_touch"},
                ],
            )

            manifest = readiness.build_readiness_report(self.make_args(root))
            by_id = {row["video_id"]: row for row in manifest["videos"]}

            self.assertEqual(by_id["train-a"]["status"], "complete_ready")
            self.assertEqual(by_id["train-a"]["audio_candidate_count"], 3)
            self.assertEqual(by_id["train-a"]["total_hint_count"], 2)
            self.assertEqual(by_id["train-a"]["checked_hint_count"], 2)

    def test_legacy_import_ready_status_ignores_generated_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            write_json(
                root / "audio_candidates" / "train-a.touch_candidates.json",
                {
                    "audio_candidates": [{"time_sec": 1.0, "strength": 1.0}],
                    "existing_event_hints": [],
                    "generated_event_hints": [{"time_sec": 2.0, "event_type": "touch", "source": "generated_touch"}],
                },
            )
            self.write_candidates(root, "train-b", [1.0])
            self.write_candidates(root, "test-a", [1.0])
            self.write_label(
                root,
                "train-a",
                "train-a.MOV",
                "train",
                complete=True,
                reviews=[{"time_sec": 1.0, "decision": "touch"}],
                touches=[1.0],
            )
            label_path = root / "visual_touch_labels" / "train-a.events.json"
            label_doc = json.loads(label_path.read_text())
            label_doc["legacy_import"] = {"source_events_path": "/tmp/legacy.events.json"}
            write_json(label_path, label_doc)

            manifest = readiness.build_readiness_report(self.make_args(root))
            by_id = {row["video_id"]: row for row in manifest["videos"]}

            self.assertEqual(by_id["train-a"]["status"], "complete_ready")
            self.assertEqual(by_id["train-a"]["total_hint_count"], 1)
            self.assertEqual(by_id["train-a"]["unchecked_hint_count"], 0)

    def test_readiness_reports_likely_and_audio_tail_unchecked_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            write_json(
                root / "audio_candidates" / "train-a.touch_candidates.json",
                {
                    "audio_candidates": [{"time_sec": 1.0, "strength": 1.0}, {"time_sec": 3.0, "strength": 1.0}],
                    "existing_event_hints": [],
                    "generated_event_hints": [{"time_sec": 2.0, "event_type": "touch", "source": "generated_touch"}],
                },
            )
            self.write_candidates(root, "train-b", [1.0])
            self.write_candidates(root, "test-a", [1.0])
            self.write_label(
                root,
                "train-a",
                "train-a.MOV",
                "train",
                complete=False,
                reviews=[{"time_sec": 1.0, "decision": "no_touch"}],
            )

            manifest = readiness.build_readiness_report(self.make_args(root))
            by_id = {row["video_id"]: row for row in manifest["videos"]}

            self.assertEqual(by_id["train-a"]["total_hint_count"], 3)
            self.assertEqual(by_id["train-a"]["checked_hint_count"], 1)
            self.assertEqual(by_id["train-a"]["likely_hint_count"], 1)
            self.assertEqual(by_id["train-a"]["likely_unchecked_hint_count"], 1)
            self.assertEqual(by_id["train-a"]["audio_only_hint_count"], 2)
            self.assertEqual(by_id["train-a"]["audio_only_unchecked_hint_count"], 1)
            self.assertEqual(manifest["summary"]["likely_unchecked_hints"], 1)
            self.assertEqual(manifest["summary"]["audio_only_unchecked_hints"], 3)
            self.assertIn("likely left", (root / "touch_label_readiness.md").read_text(encoding="utf-8"))

    def test_prioritizes_frozen_test_before_non_test_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_basic(root)
            self.write_label(root, "train-b", "train-b.MOV", "train", complete=True, reviews=[{"time_sec": 1.0, "decision": "no_touch"}])

            manifest = readiness.build_readiness_report(self.make_args(root))
            first = manifest["next_clips"][0]

            self.assertEqual(first["video_id"], "test-a")
            self.assertEqual(first["priority"], 0)
            self.assertIn("frozen-test", first["reason"])

    def test_malformed_strict_label_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_basic(root)
            self.write_label(root, "train-a", "train-a.MOV", "train", complete=True, reviews=[{"time_sec": 1.0, "decision": "maybe"}])

            manifest = readiness.build_readiness_report(self.make_args(root))
            by_id = {row["video_id"]: row for row in manifest["videos"]}

            self.assertEqual(by_id["train-a"]["status"], "malformed_label_file")
            self.assertIn("unsupported candidate review decision", by_id["train-a"]["error"])


if __name__ == "__main__":
    unittest.main()
