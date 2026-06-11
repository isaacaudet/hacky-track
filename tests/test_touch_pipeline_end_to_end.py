import argparse
import json
import tempfile
import unittest
from pathlib import Path

from attach_touch_l2_features import attach_dataset
from build_touch_training_table import build_training_table
from train_touch_classifier import train_classifier


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TouchPipelineEndToEndTests(unittest.TestCase):
    def write_manifest(self, root: Path) -> list[dict]:
        items = [
            {"video_id": "train-a", "video_name": "train-a.MOV", "video_path": "/tmp/train-a.MOV", "split": "train"},
            {"video_id": "train-b", "video_name": "train-b.MOV", "video_path": "/tmp/train-b.MOV", "split": "train"},
            {"video_id": "val-a", "video_name": "val-a.MOV", "video_path": "/tmp/val-a.MOV", "split": "validation"},
            {"video_id": "test-a", "video_name": "test-a.MOV", "video_path": "/tmp/test-a.MOV", "split": "test_frozen"},
        ]
        write_json(root / "touch_review_manifest.json", {"schema_version": 1, "items": items})
        return items

    def write_candidates_and_labels(self, root: Path, items: list[dict]) -> None:
        for item in items:
            video_id = item["video_id"]
            write_json(
                root / "audio_candidates" / f"{video_id}.touch_candidates.json",
                {
                    "schema_version": 1,
                    "video_id": video_id,
                    "video_name": item["video_name"],
                    "video_path": item["video_path"],
                    "split": item["split"],
                    "audio_candidates": [
                        {"time_sec": 1.0, "strength": 4.0, "source": "loose_audio_onset"},
                        {"time_sec": 2.0, "strength": 1.0, "source": "loose_audio_onset"},
                    ],
                    "existing_event_hints": [],
                },
            )
            write_json(
                root / "visual_touch_labels" / f"{video_id}.events.json",
                {
                    "schema_version": 1,
                    "source_video": item["video_name"],
                    "source_video_path": item["video_path"],
                    "split": item["split"],
                    "annotation_method": "muted_visual_touch_review",
                    "audio_muted_during_review_required": True,
                    "candidate_review_complete": True,
                    "review_status": "complete",
                    "candidate_reviews": [
                        {"time_sec": 1.0, "decision": "touch", "review_status": "reviewed"},
                        {"time_sec": 2.0, "decision": "no_touch", "review_status": "reviewed"},
                    ],
                    "rallies": [
                        {
                            "id": 1,
                            "start_sec": 0.0,
                            "end_sec": None,
                            "events": [
                                {
                                    "type": "touch",
                                    "time_sec": 1.0,
                                    "review_status": "approved",
                                    "source": "muted_visual_review",
                                }
                            ],
                        }
                    ],
                },
            )

    def write_detection_track(self, root: Path, items: list[dict]) -> Path:
        rows = []
        for item in items:
            for frame in range(90):
                time_sec = frame / 30.0
                rows.append(
                    {
                        "source_video": item["video_name"],
                        "frame_index": frame,
                        "time_sec": time_sec,
                        "x": 100.0 + frame * 0.5,
                        "y": 200.0 + (time_sec - 1.0) ** 2 * 50.0,
                        "confidence": 0.9,
                    }
                )
        detections = root / "detections.jsonl"
        write_jsonl(detections, rows)
        return detections

    def test_reviewed_labels_flow_to_strict_trained_classifier_without_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = self.write_manifest(root)
            self.write_candidates_and_labels(root, items)
            dataset_dir = root / "dataset"

            training_manifest = build_training_table(
                argparse.Namespace(
                    review_manifest=root / "touch_review_manifest.json",
                    candidates_dir=root / "audio_candidates",
                    labels_dir=root / "visual_touch_labels",
                    out_dir=dataset_dir,
                    touch_tolerance_sec=0.20,
                    cluster_gap_sec=0.04,
                    allow_non_visual_labels=False,
                    allow_incomplete_labels=False,
                    require_labels=True,
                )
            )

            self.assertEqual(training_manifest["labeled_videos"], 4)
            self.assertEqual(training_manifest["train_val_candidate_rows"], 6)
            self.assertEqual(training_manifest["test_frozen_candidate_rows"], 2)
            self.assertFalse(any(row["split"] == "test_frozen" for row in read_jsonl(dataset_dir / "touch_training_candidates.jsonl")))

            detections = self.write_detection_track(root, items)
            l2_manifest = attach_dataset(
                argparse.Namespace(
                    dataset_dir=dataset_dir,
                    out_dir=dataset_dir,
                    detections_jsonl=[detections],
                    detections_dir=[],
                    threshold=0.2,
                    break_tolerance_sec=0.11,
                    max_track_gap_sec=0.25,
                    min_points=12,
                )
            )

            self.assertEqual(l2_manifest["status"], "features_attached")
            self.assertEqual(l2_manifest["train_val"]["ok_rows"], 6)
            self.assertEqual(l2_manifest["test_frozen"]["ok_rows"], 2)

            classifier = train_classifier(
                argparse.Namespace(
                    dataset_dir=dataset_dir,
                    out_dir=root / "classifier",
                    threshold=0.5,
                    min_videos=3,
                    allow_small=False,
                    allow_missing_trajectory=False,
                    allow_missing_frozen_test=False,
                    audio_only=False,
                )
            )

            self.assertEqual(classifier["status"], "trained")
            self.assertTrue(classifier["strict_release_mode"])
            self.assertEqual(classifier["train_val_videos"], ["train-a", "train-b", "val-a"])
            self.assertEqual(classifier["test_frozen_videos"], ["test-a"])
            self.assertIsNotNone(classifier["frozen_test"])
            self.assertIn("audio_only", classifier["ablation_leave_one_video_out"])
            self.assertTrue((root / "classifier" / "touch_classifier.joblib").exists())


if __name__ == "__main__":
    unittest.main()
