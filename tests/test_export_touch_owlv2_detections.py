import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import export_touch_owlv2_detections as exporter


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_video(path: Path, *, fps: float = 10.0, frames: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    if not writer.isOpened():
        raise RuntimeError(f"could not create test video: {path}")
    for index in range(frames):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = index % 255
        writer.write(frame)
    writer.release()


class ExportTouchOwlv2DetectionsTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        dataset = root / "dataset"
        manifest = root / "touch_review_manifest.json"
        video = root / "video-train.MOV"
        make_video(video)
        write_json(
            manifest,
            {
                "items": [
                    {
                        "video_id": "video-train",
                        "video_name": "video-train.MOV",
                        "video_path": str(video),
                        "split": "train",
                    }
                ]
            },
        )
        write_jsonl(
            dataset / "touch_training_candidates.jsonl",
            [
                {"video_id": "video-train", "video_name": "video-train.MOV", "split": "train", "candidate_time_sec": 1.0},
                {"video_id": "video-train", "video_name": "video-train.MOV", "split": "train", "candidate_time_sec": 1.05},
            ],
        )
        write_jsonl(dataset / "touch_training_test_frozen.jsonl", [])
        return dataset, manifest, video

    def make_args(self, root: Path, dataset: Path, manifest: Path, **overrides) -> argparse.Namespace:
        values = {
            "dataset_dir": dataset,
            "review_manifest": manifest,
            "out_dir": root / "out",
            "model": "owlv2",
            "threshold": 0.2,
            "device": "cpu",
            "prompts": ["a footbag"],
            "seconds_before": 0.2,
            "seconds_after": 0.2,
            "frame_stride": 1,
            "max_frames": None,
            "dry_run": False,
            "force": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_frame_plan_dedupes_overlapping_candidate_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest, _ = self.make_fixture(root)
            rows = exporter.load_training_rows(dataset)
            videos = exporter.load_manifest_videos(manifest)

            plan = exporter.frame_indexes_for_candidates(
                rows,
                videos,
                seconds_before=0.2,
                seconds_after=0.2,
                frame_stride=1,
            )

            frames = plan["video-train.MOV"]["frame_indexes"]
            self.assertEqual(frames, sorted(set(frames)))
            self.assertLess(len(frames), 10)
            self.assertEqual(plan["video-train.MOV"]["video_id"], "video-train")
            self.assertEqual(plan["video-train.MOV"]["split"], "train")

    def test_dry_run_writes_plan_without_loading_detector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest, _ = self.make_fixture(root)

            summary = exporter.export_detections(self.make_args(root, dataset, manifest, dry_run=True))

            self.assertEqual(summary["status"], "dry_run")
            self.assertGreater(summary["frames_planned"], 0)
            self.assertEqual(summary["frames_exported"], 0)
            self.assertTrue((root / "out" / "detections.jsonl").exists())
            self.assertEqual(read_jsonl(root / "out" / "detections.jsonl"), [])

    def test_detector_export_uses_threshold_and_preserves_video_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest, _ = self.make_fixture(root)

            summary = exporter.export_detections(
                self.make_args(root, dataset, manifest, max_frames=2),
                detector=lambda rgb: [{"score": 0.25, "x": 4.0, "y": 5.0}, {"score": 0.1, "x": 1.0, "y": 2.0}],
            )
            rows = read_jsonl(root / "out" / "detections.jsonl")

            self.assertEqual(summary["status"], "exported")
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]["fires"])
            self.assertEqual(rows[0]["source_video"], "video-train.MOV")
            self.assertEqual(rows[0]["video_id"], "video-train")
            self.assertEqual(rows[0]["split"], "train")

    def test_rejects_non_fixed_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, manifest, _ = self.make_fixture(root)

            with self.assertRaisesRegex(ValueError, "threshold == 0.2"):
                exporter.export_detections(self.make_args(root, dataset, manifest, threshold=0.1, dry_run=True))


if __name__ == "__main__":
    unittest.main()
