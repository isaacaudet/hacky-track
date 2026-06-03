import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import export_dense_heatmap_dataset


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class DenseHeatmapDatasetExportTests(unittest.TestCase):
    def make_batch(self, root: Path) -> tuple[Path, Path]:
        batch = root / "batch"
        frames = batch / "clips" / "clip-a" / "frames"
        frames.mkdir(parents=True)
        image = np.full((64, 80, 3), 40, dtype=np.uint8)
        cv2.circle(image, (20, 30), 5, (0, 255, 0), -1)
        cv2.imwrite(str(frames / "frame_000001.jpg"), image)
        cv2.imwrite(str(frames / "frame_000002.jpg"), image)
        cv2.imwrite(str(frames / "frame_000003.jpg"), image)
        cv2.imwrite(str(frames / "frame_000004.jpg"), image)
        labels = batch / "dense_trajectory_labels.reviewed.jsonl"
        rows = [
            {
                "clip_id": "clip-a",
                "source_video": "a.mov",
                "split": "train",
                "training_use": "train_or_calibration",
                "frame_index": 1,
                "time_sec": 0.1,
                "frame_image": "clips/clip-a/frames/frame_000001.jpg",
                "x": 20.0,
                "y": 30.0,
                "radius": 10.0,
                "visibility": "visible",
                "quality": "reviewed",
            },
            {
                "clip_id": "clip-a",
                "source_video": "a.mov",
                "split": "train",
                "training_use": "train_or_calibration",
                "frame_index": 2,
                "time_sec": 0.2,
                "frame_image": "clips/clip-a/frames/frame_000002.jpg",
                "x": None,
                "y": None,
                "radius": 10.0,
                "visibility": "out_of_frame",
                "quality": "reviewed",
            },
            {
                "clip_id": "clip-b",
                "source_video": "b.mov",
                "split": "validation",
                "training_use": "train_or_calibration",
                "frame_index": 3,
                "time_sec": 0.3,
                "frame_image": "clips/clip-a/frames/frame_000003.jpg",
                "x": 21.0,
                "y": 31.0,
                "radius": 10.0,
                "visibility": "partially_occluded",
                "quality": "reviewed",
            },
            {
                "clip_id": "clip-c",
                "source_video": "c.mov",
                "split": "test",
                "training_use": "audit_only",
                "frame_index": 4,
                "time_sec": 0.4,
                "frame_image": "clips/clip-a/frames/frame_000004.jpg",
                "x": 22.0,
                "y": 32.0,
                "radius": 10.0,
                "visibility": "visible",
                "quality": "reviewed",
            },
        ]
        write_jsonl(labels, rows)
        return batch, labels

    def test_exports_center_manifests_heatmaps_and_sealed_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch, labels = self.make_batch(root)
            out = root / "out"
            manifest = export_dense_heatmap_dataset.export_dense_heatmap_dataset(
                labels_jsonl=labels,
                batch_dir=batch,
                out_dir=out,
                render_heatmaps=True,
                expected_counts={
                    "train_positive": 1,
                    "train_no_target": 1,
                    "train_clips": 1,
                    "validation_positive": 1,
                    "validation_no_target": 0,
                    "validation_clips": 1,
                    "test_audit_positive": 1,
                    "test_audit_no_target": 0,
                    "test_audit_clips": 1,
                    "total_positive": 3,
                    "total_no_target": 1,
                    "total_reviewed": 4,
                },
            )

            train = read_jsonl(out / "train.jsonl")
            val = read_jsonl(out / "val.jsonl")
            audit = read_jsonl(out / "test_audit.jsonl")
            negative_holdout = read_jsonl(out / "negative_holdout.jsonl")

            self.assertEqual(len(train), 2)
            self.assertEqual(len(val), 1)
            self.assertEqual(len(audit), 1)
            self.assertEqual(len(negative_holdout), 1)
            self.assertFalse(any(row["training_use"] == "audit_only" for row in train + val))
            self.assertEqual(audit[0]["training_use"], "audit_only")
            self.assertNotIn("bbox", train[0])
            self.assertNotIn("width", train[0])
            self.assertNotIn("height", train[0])
            self.assertEqual(train[1]["is_target"], False)
            self.assertIsNone(train[1]["center_x"])
            self.assertTrue((out / "dataset_manifest.json").exists())
            self.assertTrue((out / "qa" / "positive_heatmap_contact_sheet.jpg").exists())
            heatmap_path = root / train[0]["heatmap_path"]
            heatmap = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(heatmap)
            peak_y, peak_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
            self.assertLessEqual(abs(peak_x - 20), 1)
            self.assertLessEqual(abs(peak_y - 30), 1)
            self.assertEqual(manifest["splits"]["test_audit"]["positive"], 1)

    def test_clip_split_leak_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch, labels = self.make_batch(root)
            rows = read_jsonl(labels)
            rows[1]["split"] = "validation"
            write_jsonl(labels, rows)
            with self.assertRaises(AssertionError):
                export_dense_heatmap_dataset.export_dense_heatmap_dataset(
                    labels_jsonl=labels,
                    batch_dir=batch,
                    out_dir=root / "out",
                    expected_counts=None,
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()
