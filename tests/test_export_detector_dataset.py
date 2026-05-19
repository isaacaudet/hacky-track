from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import export_detector_dataset


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (688, 912))
    if not writer.isOpened():
        raise RuntimeError("Could not create test video")
    for idx in range(4):
        frame = np.full((912, 688, 3), (60, 140, 70), dtype=np.uint8)
        cv2.circle(frame, (300 + idx * 5, 420), 18, (0, 0, 220), -1)
        writer.write(frame)
    writer.release()


class DetectorDatasetExportTests(unittest.TestCase):
    def test_exports_yolo_labels_and_bad_center_hard_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.avi"
            write_test_video(video)
            qa_manifest = root / "qa_manifest.json"
            write_json(qa_manifest, {"runs": [{"video": str(video)}]})
            reviews = root / "reviews"
            write_json(
                reviews / "clip.review.json",
                {
                    "source_video": "clip.avi",
                    "items": [
                        {
                            "id": "approved-touch",
                            "kind": "touch",
                            "source": "candidate",
                            "status": "approved",
                            "time_sec": 0.1,
                            "x": 300,
                            "y": 420,
                            "qa_ball_radius": 18,
                            "ball_accuracy": "reviewed",
                        },
                        {
                            "id": "bad-center",
                            "kind": "touch",
                            "source": "candidate",
                            "status": "rejected",
                            "time_sec": 0.2,
                            "x": 120,
                            "y": 700,
                            "ball_accuracy": "bad",
                            "note": "marker on the leg, no visible footbag",
                        },
                        {
                            "id": "hand-bag",
                            "kind": "touch",
                            "source": "candidate",
                            "status": "rejected",
                            "time_sec": 0.3,
                            "x": 310,
                            "y": 420,
                            "ball_accuracy": "reviewed",
                            "note": "hand-held bag is not a valid touch",
                        },
                    ],
                },
            )
            out = root / "dataset"
            summary = export_detector_dataset.export_dataset(qa_manifest, reviews, out)

            self.assertEqual(summary["positive_labels"], 2)
            self.assertEqual(summary["hard_negative_points"], 1)
            self.assertEqual(summary["written_hard_negative_yolo_images"], 1)
            self.assertTrue((out / "data.yaml").exists())
            self.assertTrue((out / "images" / "test").exists())
            labels = list((out / "labels" / "test").glob("*.txt"))
            self.assertEqual(len(labels), 3)
            positive_labels = [path for path in labels if path.read_text(encoding="utf-8").strip()]
            empty_labels = [path for path in labels if not path.read_text(encoding="utf-8").strip()]
            self.assertEqual(len(positive_labels), 2)
            self.assertEqual(len(empty_labels), 1)
            label_text = positive_labels[0].read_text(encoding="utf-8").strip()
            self.assertRegex(label_text, r"^0 0\.[0-9]+ 0\.[0-9]+ 0\.[0-9]+ 0\.[0-9]+$")
            negatives = (out / "hard_negatives" / "points.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(negatives), 1)
            self.assertEqual(json.loads(negatives[0])["item_id"], "bad-center")

    def test_dry_run_does_not_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_manifest = root / "qa_manifest.json"
            reviews = root / "reviews"
            write_json(qa_manifest, {"runs": []})
            reviews.mkdir()
            out = root / "dataset"
            summary = export_detector_dataset.export_dataset(qa_manifest, reviews, out, dry_run=True)
            self.assertEqual(summary["positive_labels"], 0)
            self.assertFalse(out.exists())

    def test_detector_label_decisions_extend_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.avi"
            write_test_video(video)
            qa_manifest = root / "qa_manifest.json"
            write_json(qa_manifest, {"runs": [{"video": str(video)}]})
            reviews = root / "reviews"
            reviews.mkdir()
            review_manifest = root / "detector_review" / "detector_label_review_manifest.json"
            write_json(
                review_manifest,
                {
                    "items": [
                        {
                            "detector_label_id": "detlbl-1",
                            "source_video": "clip.avi",
                            "event_type": "touch",
                            "time_sec": 0.1,
                            "x": 300,
                            "y": 420,
                            "radius": 18,
                            "split": "train",
                        },
                        {
                            "detector_label_id": "detlbl-2",
                            "source_video": "clip.avi",
                            "event_type": "touch",
                            "time_sec": 0.2,
                            "x": 120,
                            "y": 700,
                            "radius": 18,
                            "split": "test",
                        },
                    ]
                },
            )
            decisions = root / "detector_review" / "decisions.json"
            write_json(
                decisions,
                {
                    "decisions": [
                        {"detector_label_id": "detlbl-1", "detector_status": "corrected", "corrected_x": 305, "corrected_y": 422, "evidence": "bag center corrected"},
                        {"detector_label_id": "detlbl-2", "detector_status": "not_footbag", "evidence": "marker on leg"},
                    ]
                },
            )
            out = root / "dataset"
            summary = export_detector_dataset.export_dataset(
                qa_manifest,
                reviews,
                out,
                detector_label_review_manifest=review_manifest,
                detector_label_decisions=decisions,
            )
            self.assertEqual(summary["positive_labels"], 1)
            self.assertEqual(summary["hard_negative_points"], 1)
            self.assertEqual(summary["detector_label_decisions_used"], 2)
            labels = list((out / "labels").glob("*/*.txt"))
            self.assertEqual(len(labels), 2)
            records = [json.loads(line) for line in (out / "reviewed_detector_labels.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["x"], 305.0)
            self.assertEqual(records[0]["ball_label_source"], "detector_label_review")

    def test_multiple_detector_label_review_pairs_are_combined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.avi"
            write_test_video(video)
            qa_manifest = root / "qa_manifest.json"
            write_json(qa_manifest, {"runs": [{"video": str(video)}]})
            reviews = root / "reviews"
            reviews.mkdir()

            review_a = root / "review_a" / "manifest.json"
            decisions_a = root / "review_a" / "decisions.json"
            write_json(
                review_a,
                {"items": [{"detector_label_id": "a", "source_video": "clip.avi", "time_sec": 0.1, "x": 300, "y": 420, "split": "train"}]},
            )
            write_json(decisions_a, {"decisions": [{"detector_label_id": "a", "detector_status": "footbag"}]})

            review_b = root / "review_b" / "manifest.json"
            decisions_b = root / "review_b" / "decisions.json"
            write_json(
                review_b,
                {"items": [{"detector_label_id": "b", "source_video": "clip.avi", "time_sec": 0.2, "x": 110, "y": 700, "split": "test"}]},
            )
            write_json(decisions_b, {"decisions": [{"detector_label_id": "b", "detector_status": "not_footbag"}]})

            out = root / "dataset"
            summary = export_detector_dataset.export_dataset(
                qa_manifest,
                reviews,
                out,
                detector_label_review_pairs=[(review_a, decisions_a), (review_b, decisions_b)],
            )

            self.assertEqual(summary["positive_labels"], 1)
            self.assertEqual(summary["hard_negative_points"], 1)
            self.assertEqual(summary["detector_label_decisions_seen"], 2)
            self.assertEqual(summary["detector_label_decisions_used"], 2)
            self.assertEqual(len(summary["detector_label_review_sets"]), 2)


if __name__ == "__main__":
    unittest.main()
