from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import release_evaluation


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class ReleaseEvaluationTests(unittest.TestCase):
    def test_video_splits_are_deterministic_and_hold_out(self) -> None:
        videos = [f"clip-{idx}.MOV" for idx in range(10)]
        first = release_evaluation.deterministic_video_splits(videos, seed=42)
        second = release_evaluation.deterministic_video_splits(list(reversed(videos)), seed=42)
        self.assertEqual(first, second)
        self.assertIn("test", set(first.values()))
        self.assertIn("validation", set(first.values()))
        self.assertIn("train", set(first.values()))

    def test_clean_batch_labels_are_separated_from_exploratory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviews = root / "reviews"
            batch = root / "latest_review_batch.json"
            write_json(
                batch,
                {
                    "batch_id": "batch-1",
                    "items": [
                        {"source_video": "clip-a.MOV", "review_item_id": "cand-1"},
                        {"source_video": "clip-b.MOV", "review_item_id": "cand-2"},
                    ],
                },
            )
            write_json(
                reviews / "clip-a.review.json",
                {
                    "source_video": "clip-a.MOV",
                    "items": [
                        {
                            "id": "cand-1",
                            "source": "candidate",
                            "kind": "touch",
                            "status": "approved",
                            "contact_side": "right",
                            "detector_contact_side": "right",
                            "contact_type": "foot",
                            "detector_contact_type": "foot",
                        },
                        {
                            "id": "old-1",
                            "source": "candidate",
                            "kind": "drop_floor",
                            "status": "rejected",
                            "reviewed_at": "2026-01-01T00:00:00Z",
                        },
                    ],
                },
            )
            write_json(
                reviews / "clip-b.review.json",
                {
                    "source_video": "clip-b.MOV",
                    "items": [
                        {
                            "id": "cand-2",
                            "source": "candidate",
                            "kind": "touch",
                            "status": "rejected",
                            "note": "duplicate",
                        }
                    ],
                },
            )
            doc = release_evaluation.evaluation_doc(
                review_paths=sorted(reviews.glob("*.review.json")),
                batch=release_evaluation.load_batch(batch),
                training_manifest=None,
                model_dir=None,
                ball_audit=None,
                seed=1,
            )
            self.assertEqual(doc["label_inventory"]["label_tiers"]["clean_current_batch"], 2)
            self.assertEqual(doc["label_inventory"]["label_tiers"]["exploratory_reviewed"], 1)
            all_clean = doc["metrics"]["all_clean_current_batch"]
            self.assertEqual(all_clean["kinds"]["touch"]["approved_candidates"], 1)
            self.assertEqual(all_clean["kinds"]["touch"]["rejected_candidates"], 1)
            self.assertEqual(all_clean["kinds"]["drop_floor"]["rejected_candidates"], 0)
            self.assertEqual(all_clean["duplicates"]["duplicate_touch_rejections"], 1)
            self.assertEqual(doc["dataset_splits"]["assignment_population"], "review_batch_videos")
            self.assertEqual(set(doc["dataset_splits"]["assignments"]), {"clip-a.MOV", "clip-b.MOV"})

    def test_video_splits_are_anchored_to_full_review_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviews = root / "reviews"
            batch = root / "latest_review_batch.json"
            batch_videos = [f"clip-{idx}.MOV" for idx in range(8)]
            write_json(
                batch,
                {
                    "batch_id": "batch-1",
                    "items": [
                        {"source_video": video, "review_item_id": f"cand-{idx}"}
                        for idx, video in enumerate(batch_videos)
                    ],
                },
            )
            write_json(
                reviews / "clip-0.review.json",
                {
                    "source_video": "clip-0.MOV",
                    "items": [
                        {
                            "id": "cand-0",
                            "source": "candidate",
                            "kind": "touch",
                            "status": "approved",
                        }
                    ],
                },
            )
            doc = release_evaluation.evaluation_doc(
                review_paths=sorted(reviews.glob("*.review.json")),
                batch=release_evaluation.load_batch(batch),
                training_manifest=None,
                model_dir=None,
                ball_audit=None,
                seed=7,
            )
            self.assertEqual(doc["dataset_splits"]["assignment_population"], "review_batch_videos")
            self.assertEqual(doc["dataset_splits"]["source_video_count"], len(batch_videos))
            self.assertEqual(set(doc["dataset_splits"]["assignments"]), set(batch_videos))

    def test_model_artifact_metadata_uses_portable_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=release_evaluation.ROOT) as tmp:
            run = Path(tmp)
            models = run / "models"
            models.mkdir()
            artifact = models / "footbag_patch_hgb.joblib"
            artifact.write_bytes(b"model")
            manifest = run / "training" / "full_training_manifest.json"
            write_json(
                manifest,
                {
                    "model_path": str(artifact),
                    "training_report": {"accuracy": 0.9},
                },
            )
            metadata = release_evaluation.model_artifacts(manifest, models)
            self.assertTrue(metadata["has_saved_model_artifact"])
            self.assertNotIn("/Users/", json.dumps(metadata))


if __name__ == "__main__":
    unittest.main()
