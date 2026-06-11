import argparse
import json
import tempfile
import unittest
from pathlib import Path

from prefill_train_touch_suggestions import prefill_train_suggestions
from build_touch_training_table import candidate_reviews


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class PrefillTrainTouchSuggestionsTests(unittest.TestCase):
    def make_args(self, root: Path, *, force: bool = False) -> argparse.Namespace:
        return argparse.Namespace(
            review_manifest=root / "touch_review_manifest.json",
            candidates_dir=root / "audio_candidates",
            labels_dir=root / "visual_touch_labels",
            detections_jsonl=[],
            threshold=0.2,
            cluster_gap_sec=0.04,
            break_tolerance_sec=0.06,
            max_track_gap_sec=0.25,
            min_points=12,
            force=force,
            dry_run=False,
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
                        "video_id": "video-val",
                        "video_name": "video-val.MOV",
                        "video_path": "/tmp/video-val.MOV",
                        "split": "validation",
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

    def write_candidates(self, root: Path) -> None:
        write_json(
            root / "audio_candidates" / "video-train.touch_candidates.json",
            {
                "audio_candidates": [
                    {"time_sec": 1.0, "strength": 1.0, "source": "loose_audio_onset"},
                    {"time_sec": 2.0, "strength": 1.5, "source": "loose_audio_onset"},
                ],
                "existing_event_hints": [
                    {"time_sec": 1.0, "event_type": "touch", "source": "generated_touch_hint"},
                ],
                "generated_event_hints": [],
            },
        )
        write_json(root / "audio_candidates" / "video-val.touch_candidates.json", {"audio_candidates": []})
        write_json(root / "audio_candidates" / "video-test.touch_candidates.json", {"audio_candidates": []})

    def test_prefills_train_only_as_suggestions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_candidates(root)

            summary = prefill_train_suggestions(self.make_args(root))

            self.assertEqual(summary["prefilled"], 1)
            self.assertEqual(summary["skipped_non_train"], 2)
            self.assertFalse((root / "visual_touch_labels" / "video-val.events.json").exists())
            self.assertFalse((root / "visual_touch_labels" / "video-test.events.json").exists())

            doc = json.loads((root / "visual_touch_labels" / "video-train.events.json").read_text())
            self.assertFalse(doc["candidate_review_complete"])
            self.assertEqual(doc["review_status"], "in_progress")
            self.assertEqual(doc["suggestion_prefill"]["source"], "train_audio_trajectory_draft_prefill")
            self.assertEqual([row["review_status"] for row in doc["candidate_reviews"]], ["suggested", "suggested"])
            self.assertEqual([row["decision"] for row in doc["candidate_reviews"]], ["touch", "no_touch"])
            self.assertEqual(candidate_reviews(doc), [])

    def test_preserves_human_reviewed_rows_when_replacing_suggestions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_candidates(root)
            write_json(
                root / "visual_touch_labels" / "video-train.events.json",
                {
                    "schema_version": 1,
                    "source_video": "video-train.MOV",
                    "source_video_path": "/tmp/video-train.MOV",
                    "split": "train",
                    "annotation_method": "muted_visual_touch_review",
                    "audio_muted_during_review_required": True,
                    "candidate_review_complete": False,
                    "review_status": "in_progress",
                    "candidate_reviews": [
                        {"time_sec": 1.0, "decision": "no_touch", "review_status": "reviewed", "source": "human"},
                        {
                            "time_sec": 2.0,
                            "decision": "touch",
                            "review_status": "suggested",
                            "source": "train_audio_trajectory_draft_prefill",
                        },
                    ],
                    "rallies": [{"id": 1, "events": []}],
                },
            )

            prefill_train_suggestions(self.make_args(root, force=True))

            doc = json.loads((root / "visual_touch_labels" / "video-train.events.json").read_text())
            reviewed = [row for row in doc["candidate_reviews"] if row.get("review_status") == "reviewed"]
            suggested = [row for row in doc["candidate_reviews"] if row.get("review_status") == "suggested"]
            self.assertEqual(reviewed, [{"time_sec": 1.0, "decision": "no_touch", "review_status": "reviewed", "source": "human"}])
            self.assertEqual(len(suggested), 1)
            self.assertEqual(suggested[0]["time_sec"], 2.0)


if __name__ == "__main__":
    unittest.main()
