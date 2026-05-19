from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import build_dense_trajectory_review_batch


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_video(path: Path, frames: int = 24, fps: float = 12.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("test video writer did not open")
    for index in range(frames):
        frame = np.full((48, 64, 3), 20 + index, dtype=np.uint8)
        cv2.circle(frame, (10 + index % 40, 24), 4, (0, 0, 255), -1)
        writer.write(frame)
    writer.release()


class BuildDenseTrajectoryReviewBatchTests(unittest.TestCase):
    def test_dry_run_selects_failures_and_preserves_test_audit_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "clip.mp4"
            heldout = root / "heldout.mp4"
            make_video(clip)
            make_video(heldout)
            qa_manifest = root / "qa_manifest.json"
            write_json(qa_manifest, {"runs": [{"video": str(clip)}, {"video": str(heldout)}]})
            dataset = root / "dataset"
            write_json(dataset / "manifest.json", {"splits": {"clip.mp4": "train", "heldout.mp4": "test"}})
            track_metrics = root / "track_metrics.json"
            write_json(
                track_metrics,
                {
                    "rows": [
                        {
                            "kind": "positive",
                            "result": "fail",
                            "video": "clip.mp4",
                            "split": "train",
                            "time_sec": 0.5,
                            "center_error_px": 80,
                        },
                        {
                            "kind": "hard_negative",
                            "result": "not_applicable",
                            "hard_negative_result": "false_positive_near_bad_point",
                            "video": "heldout.mp4",
                            "split": "test",
                            "time_sec": 0.8,
                        },
                    ]
                },
            )

            summary = build_dense_trajectory_review_batch.build_dense_trajectory_review_batch(
                qa_manifest=qa_manifest,
                track_metrics=[track_metrics],
                dataset=dataset,
                out_dir=root / "out",
                max_clips=4,
                dry_run=True,
            )

            self.assertEqual(summary["selected_clips"], 2)
            by_video = {clip["source_video"]: clip for clip in summary["clips"]}
            self.assertEqual(by_video["clip.mp4"]["training_use"], "train_or_calibration")
            self.assertEqual(by_video["heldout.mp4"]["training_use"], "audit_only")

    def test_writes_clip_frames_schema_and_label_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "clip.mp4"
            make_video(clip, frames=36, fps=12.0)
            qa_manifest = root / "qa_manifest.json"
            write_json(qa_manifest, {"runs": [{"video": str(clip)}]})
            dataset = root / "dataset"
            write_json(dataset / "manifest.json", {"splits": {"clip.mp4": "validation"}})
            track_metrics = root / "track_metrics.json"
            write_json(
                track_metrics,
                {
                    "rows": [
                        {
                            "kind": "positive",
                            "result": "missing_track_point",
                            "video": "clip.mp4",
                            "split": "validation",
                            "time_sec": 1.0,
                        }
                    ]
                },
            )

            summary = build_dense_trajectory_review_batch.build_dense_trajectory_review_batch(
                qa_manifest=qa_manifest,
                track_metrics=[track_metrics],
                dataset=dataset,
                out_dir=root / "out",
                max_clips=1,
                seconds_before=0.25,
                seconds_after=0.25,
                frame_stride=3,
            )

            self.assertEqual(summary["selected_clips"], 1)
            self.assertGreater(summary["counts"]["frames_exported"], 0)
            self.assertTrue((root / "out" / "dense_trajectory_schema.json").exists())
            combined = root / "out" / "dense_trajectory_labels_template.jsonl"
            rows = [json.loads(line) for line in combined.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows)
            self.assertEqual(rows[0]["visibility"], "unlabeled")
            self.assertEqual(rows[0]["split"], "validation")
            self.assertTrue((root / "out" / summary["clips"][0]["contact_sheet"]).exists())


if __name__ == "__main__":
    unittest.main()
