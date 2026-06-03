import argparse
import json
import tempfile
import unittest
from pathlib import Path

import build_touch_training_table as builder


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class BuildTouchTrainingTableTests(unittest.TestCase):
    def make_args(self, root: Path, *, labels_dir: Path | None = None) -> argparse.Namespace:
        return argparse.Namespace(
            review_manifest=root / "touch_review_manifest.json",
            candidates_dir=root / "audio_candidates",
            labels_dir=labels_dir or root / "visual_touch_labels",
            out_dir=root / "touch_training_dataset",
            touch_tolerance_sec=0.20,
            cluster_gap_sec=0.04,
            allow_non_visual_labels=False,
            allow_incomplete_labels=False,
            require_labels=False,
        )

    def write_manifest(self, root: Path) -> None:
        write_json(
            root / "touch_review_manifest.json",
            {
                "items": [
                    {
                        "video_id": "video-train",
                        "video_name": "video-train.MOV",
                        "video_path": "/tmp/video-train.MOV",
                        "split": "train",
                    },
                    {
                        "video_id": "video-test",
                        "video_name": "video-test.MOV",
                        "video_path": "/tmp/video-test.MOV",
                        "split": "test_frozen",
                    },
                ]
            },
        )

    def write_candidates(self, root: Path, video_id: str, times: list[float]) -> None:
        write_json(
            root / "audio_candidates" / f"{video_id}.touch_candidates.json",
            {
                "audio_candidates": [
                    {"time_sec": time_sec, "strength": 1.0 + index, "source": "loose_audio_onset"}
                    for index, time_sec in enumerate(times)
                ],
                "existing_event_hints": [],
            },
        )

    def write_label(
        self,
        root: Path,
        video_id: str,
        video_name: str,
        split: str,
        touches: list[float],
        *,
        method: str = "muted_visual_touch_review",
        complete: bool = True,
        candidate_reviews: list[dict] | None = None,
    ) -> None:
        write_json(
            root / "visual_touch_labels" / f"{video_id}.events.json",
            {
                "schema_version": 1,
                "source_video": video_name,
                "split": split,
                "annotation_method": method,
                "audio_muted_during_review_required": True,
                "candidate_review_complete": complete,
                "candidate_reviews": candidate_reviews or [],
                "rallies": [
                    {
                        "id": 1,
                        "start_sec": 0.0,
                        "end_sec": None,
                        "events": [
                            {
                                "type": "touch",
                                "time_sec": time_sec,
                                "review_status": "approved",
                                "source": "muted_visual_review",
                            }
                            for time_sec in touches
                        ],
                    }
                ],
            },
        )

    def test_builds_labels_and_keeps_frozen_test_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_candidates(root, "video-train", [1.0, 2.0])
            self.write_candidates(root, "video-test", [3.0])
            self.write_label(root, "video-train", "video-train.MOV", "train", [1.05])
            self.write_label(root, "video-test", "video-test.MOV", "test_frozen", [3.01])

            manifest = builder.build_training_table(self.make_args(root))

            train_rows = read_jsonl(root / "touch_training_dataset" / "touch_training_candidates.jsonl")
            test_rows = read_jsonl(root / "touch_training_dataset" / "touch_training_test_frozen.jsonl")
            self.assertEqual(manifest["labeled_videos"], 2)
            self.assertEqual(manifest["skipped_incomplete_video_count"], 0)
            self.assertEqual(len(train_rows), 2)
            self.assertEqual(len(test_rows), 1)
            self.assertTrue(all(row["candidate_reviewed"] for row in train_rows))
            self.assertTrue(any(row["label_is_touch"] for row in train_rows))
            self.assertTrue(test_rows[0]["label_is_touch"])
            self.assertFalse(any(row["split"] == "test_frozen" for row in train_rows))

    def test_rejects_audio_derived_labels_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_candidates(root, "video-train", [1.0])
            self.write_label(root, "video-train", "video-train.MOV", "train", [1.0], method="audio_transient_review")

            with self.assertRaisesRegex(ValueError, "annotation_method"):
                builder.build_training_table(self.make_args(root))

    def test_rejects_split_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_candidates(root, "video-train", [1.0])
            self.write_label(root, "video-train", "video-train.MOV", "test_frozen", [1.0])

            with self.assertRaisesRegex(ValueError, "split"):
                builder.build_training_table(self.make_args(root))

    def test_rejects_unknown_label_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            write_json(root / "visual_touch_labels" / "unknown.events.json", {})

            with self.assertRaisesRegex(ValueError, "not present in manifest"):
                builder.build_training_table(self.make_args(root))

    def test_skips_incomplete_label_files_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_candidates(root, "video-train", [1.0, 2.0])
            self.write_label(root, "video-train", "video-train.MOV", "train", [1.0], complete=False)

            manifest = builder.build_training_table(self.make_args(root))
            train_rows = read_jsonl(root / "touch_training_dataset" / "touch_training_candidates.jsonl")

            self.assertEqual(manifest["labeled_videos"], 0)
            self.assertEqual(manifest["skipped_incomplete_video_count"], 1)
            self.assertEqual(train_rows, [])

    def test_can_consume_incomplete_labels_for_smoke_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_candidates(root, "video-train", [1.0])
            self.write_label(root, "video-train", "video-train.MOV", "train", [1.0], complete=False)

            args = self.make_args(root)
            args.allow_incomplete_labels = True
            manifest = builder.build_training_table(args)

            self.assertEqual(manifest["labeled_videos"], 1)
            self.assertFalse(manifest["training_guardrails"]["complete_clip_review_required"])

    def test_skips_complete_file_with_unreviewed_negative_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_candidates(root, "video-train", [1.0, 2.0])
            self.write_label(
                root,
                "video-train",
                "video-train.MOV",
                "train",
                [1.0],
                candidate_reviews=[{"time_sec": 1.0, "decision": "touch"}],
            )

            manifest = builder.build_training_table(self.make_args(root))
            train_rows = read_jsonl(root / "touch_training_dataset" / "touch_training_candidates.jsonl")

            self.assertEqual(manifest["labeled_videos"], 0)
            self.assertEqual(manifest["skipped_incomplete_video_count"], 1)
            self.assertIn("negative candidates lack candidate_reviews", manifest["skipped_incomplete_videos"][0]["reason"])
            self.assertEqual(train_rows, [])

    def test_uses_candidate_reviews_to_audit_negative_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_candidates(root, "video-train", [1.0, 2.0])
            self.write_label(
                root,
                "video-train",
                "video-train.MOV",
                "train",
                [1.0],
                candidate_reviews=[
                    {"time_sec": 1.0, "decision": "touch"},
                    {"time_sec": 2.0, "decision": "no_touch"},
                ],
            )

            manifest = builder.build_training_table(self.make_args(root))
            train_rows = read_jsonl(root / "touch_training_dataset" / "touch_training_candidates.jsonl")

            self.assertEqual(manifest["labeled_videos"], 1)
            self.assertEqual(manifest["candidate_reviews"], 2)
            self.assertEqual(manifest["reviewed_candidate_rows"], 2)
            self.assertEqual(manifest["negative_candidate_rows_without_review"], 0)
            self.assertEqual([row["candidate_review_decision"] for row in train_rows], ["touch", "no_touch"])

    def test_candidate_review_no_touch_overrides_nearby_approved_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_candidates(root, "video-train", [1.0, 1.15])
            self.write_label(
                root,
                "video-train",
                "video-train.MOV",
                "train",
                [1.0],
                candidate_reviews=[
                    {"time_sec": 1.0, "decision": "touch"},
                    {"time_sec": 1.15, "decision": "no_touch"},
                ],
            )

            manifest = builder.build_training_table(self.make_args(root))
            train_rows = read_jsonl(root / "touch_training_dataset" / "touch_training_candidates.jsonl")

            self.assertEqual(manifest["labeled_videos"], 1)
            self.assertEqual(manifest["positive_candidate_rows"], 1)
            self.assertEqual([row["label_is_touch"] for row in train_rows], [True, False])
            self.assertEqual([row["candidate_review_source"] for row in train_rows], ["candidate_review", "candidate_review"])
            self.assertEqual([row["candidate_review_decision"] for row in train_rows], ["touch", "no_touch"])

    def test_approved_touch_without_hint_gets_training_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_candidates(root, "video-train", [2.0])
            self.write_label(
                root,
                "video-train",
                "video-train.MOV",
                "train",
                [1.0],
                candidate_reviews=[
                    {"time_sec": 2.0, "decision": "no_touch"},
                ],
            )

            manifest = builder.build_training_table(self.make_args(root))
            train_rows = read_jsonl(root / "touch_training_dataset" / "touch_training_candidates.jsonl")

            self.assertEqual(manifest["labeled_videos"], 1)
            self.assertEqual(manifest["candidate_rows"], 2)
            self.assertEqual(manifest["touches_without_candidate"], 0)
            self.assertEqual([row["candidate_time_sec"] for row in train_rows], [1.0, 2.0])
            self.assertEqual([row["label_is_touch"] for row in train_rows], [True, False])
            self.assertEqual(train_rows[0]["candidate_review_source"], "approved_manual_event")
            self.assertIn("approved_manual_touch", train_rows[0]["raw_sources"])

    def test_clusters_generated_event_hints_with_audio_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            write_json(
                root / "audio_candidates" / "video-train.touch_candidates.json",
                {
                    "audio_candidates": [{"time_sec": 1.0, "strength": 1.0, "source": "loose_audio_onset"}],
                    "existing_event_hints": [],
                    "generated_event_hints": [
                        {"time_sec": 1.01, "event_type": "touch", "source": "generated_touch"},
                        {"time_sec": 2.0, "event_type": "touch", "source": "generated_touch"},
                    ],
                },
            )
            self.write_candidates(root, "video-test", [3.0])
            self.write_label(
                root,
                "video-train",
                "video-train.MOV",
                "train",
                [1.0],
                candidate_reviews=[
                    {"time_sec": 1.0, "decision": "touch"},
                    {"time_sec": 2.0, "decision": "no_touch"},
                ],
            )

            manifest = builder.build_training_table(self.make_args(root))
            rows = read_jsonl(root / "touch_training_dataset" / "touch_training_candidates.jsonl")

            self.assertEqual(manifest["candidate_rows"], 2)
            self.assertEqual(rows[0]["existing_hint_types"], ["touch"])
            self.assertIn("generated_touch", rows[0]["raw_sources"])

    def test_legacy_import_labels_do_not_consume_generated_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            write_json(
                root / "audio_candidates" / "video-train.touch_candidates.json",
                {
                    "audio_candidates": [{"time_sec": 1.0, "strength": 1.0, "source": "loose_audio_onset"}],
                    "existing_event_hints": [],
                    "generated_event_hints": [
                        {"time_sec": 2.0, "event_type": "touch", "source": "generated_touch"},
                    ],
                },
            )
            self.write_candidates(root, "video-test", [3.0])
            self.write_label(
                root,
                "video-train",
                "video-train.MOV",
                "train",
                [1.0],
                candidate_reviews=[
                    {"time_sec": 1.0, "decision": "touch"},
                    {"time_sec": 2.0, "decision": "no_touch"},
                ],
            )
            label_path = root / "visual_touch_labels" / "video-train.events.json"
            label_doc = json.loads(label_path.read_text())
            label_doc["legacy_import"] = {"source_events_path": "/tmp/legacy.events.json"}
            write_json(label_path, label_doc)

            manifest = builder.build_training_table(self.make_args(root))
            rows = read_jsonl(root / "touch_training_dataset" / "touch_training_candidates.jsonl")

            self.assertEqual(manifest["candidate_rows"], 1)
            self.assertEqual(rows[0]["candidate_time_sec"], 1.0)


if __name__ == "__main__":
    unittest.main()
