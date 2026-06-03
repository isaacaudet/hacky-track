import argparse
import json
import tempfile
import unittest
from pathlib import Path

from prefill_touch_labels_from_reviews import prefill_labels


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class PrefillTouchLabelsFromReviewsTests(unittest.TestCase):
    def make_args(self, root: Path, *, force: bool = False) -> argparse.Namespace:
        return argparse.Namespace(
            review_manifest=root / "touch_review_manifest.json",
            reviews_dir=root / "reviews",
            candidates_dir=root / "audio_candidates",
            labels_dir=root / "visual_touch_labels",
            split="test_frozen",
            candidate_match_tolerance_sec=0.20,
            conflict_tolerance_sec=0.05,
            cluster_gap_sec=0.04,
            write_empty=False,
            force=force,
            dry_run=False,
        )

    def write_manifest_and_candidates(self, root: Path) -> None:
        write_json(
            root / "touch_review_manifest.json",
            {
                "items": [
                    {
                        "video_id": "video-a",
                        "video_name": "video-a.MOV",
                        "video_path": str(root / "video-a.MOV"),
                        "split": "test_frozen",
                    }
                ]
            },
        )
        write_json(
            root / "audio_candidates" / "video-a.touch_candidates.json",
            {
                "audio_candidates": [
                    {"time_sec": 1.0, "strength": 1.0},
                    {"time_sec": 2.0, "strength": 1.0},
                    {"time_sec": 3.0, "strength": 1.0},
                ],
                "existing_event_hints": [],
            },
        )

    def test_prefills_draft_events_and_candidate_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest_and_candidates(root)
            write_json(
                root / "reviews" / "video-a.review.json",
                {
                    "source_video": "video-a.MOV",
                    "items": [
                        {"id": "a", "kind": "touch", "status": "approved", "time_sec": 1.02},
                        {"id": "b", "kind": "touch", "status": "rejected", "time_sec": 2.01},
                        {"id": "c", "kind": "stall", "status": "approved", "time_sec": 3.0, "duration_sec": 0.4},
                    ],
                },
            )

            summary = prefill_labels(self.make_args(root))

            self.assertEqual(summary["prefilled"], 1)
            self.assertEqual(summary["prefilled_touches"], 1)
            self.assertEqual(summary["candidate_reviews"], 3)
            label = json.loads((root / "visual_touch_labels" / "video-a.events.json").read_text())
            self.assertFalse(label["candidate_review_complete"])
            self.assertEqual(label["review_status"], "in_progress")
            self.assertEqual(label["prefill_import"]["source"], "reviews_subset")
            self.assertEqual(
                [(event["type"], event["time_sec"]) for event in label["rallies"][0]["events"]],
                [("touch", 1.02), ("stall", 3.0)],
            )
            self.assertEqual([row["decision"] for row in label["candidate_reviews"]], ["touch", "no_touch", "no_touch"])

    def test_conflicting_review_items_are_not_prefilled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest_and_candidates(root)
            write_json(
                root / "reviews" / "video-a.review.json",
                {
                    "source_video": "video-a.MOV",
                    "items": [
                        {"id": "yes", "kind": "touch", "status": "approved", "time_sec": 1.0},
                        {"id": "no", "kind": "touch", "status": "rejected", "time_sec": 1.0},
                        {"id": "clean", "kind": "touch", "status": "approved", "time_sec": 2.0},
                    ],
                },
            )

            summary = prefill_labels(self.make_args(root))

            self.assertEqual(summary["conflicted_review_items"], 2)
            label = json.loads((root / "visual_touch_labels" / "video-a.events.json").read_text())
            self.assertEqual([(event["type"], event["time_sec"]) for event in label["rallies"][0]["events"]], [("touch", 2.0)])
            self.assertEqual([row["time_sec"] for row in label["candidate_reviews"]], [2.0])

    def test_existing_label_is_not_overwritten_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest_and_candidates(root)
            write_json(root / "reviews" / "video-a.review.json", {"source_video": "video-a.MOV", "items": []})
            write_json(root / "visual_touch_labels" / "video-a.events.json", {"sentinel": True})

            summary = prefill_labels(self.make_args(root))

            self.assertEqual(summary["skipped_existing_label"], 1)
            self.assertEqual(json.loads((root / "visual_touch_labels" / "video-a.events.json").read_text()), {"sentinel": True})


if __name__ == "__main__":
    unittest.main()
