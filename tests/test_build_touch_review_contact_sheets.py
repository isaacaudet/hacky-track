import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import build_touch_review_contact_sheets as sheets


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("could not open synthetic video writer")
    try:
        for index in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = index * 20
            frame[:, :, 1] = 80
            frame[:, :, 2] = 180
            writer.write(frame)
    finally:
        writer.release()


class TouchReviewContactSheetTests(unittest.TestCase):
    def make_args(self, root: Path, out_dir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            review_manifest=root / "touch_review_manifest.json",
            candidates_dir=root / "audio_candidates",
            labels_dir=root / "visual_touch_labels",
            out_dir=out_dir,
            split="test_frozen",
            filters=["likely_unchecked", "audio_only"],
            only_incomplete=True,
            review_match_tolerance_sec=0.05,
            candidate_cluster_gap_sec=0.04,
            max_videos=None,
            max_candidates_per_sheet=1,
            cols=1,
            thumb_width=80,
            label_height=50,
        )

    def test_renders_unchecked_likely_and_audio_tail_sheets_without_writing_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "test-a.MOV"
            write_video(video_path)
            write_json(
                root / "touch_review_manifest.json",
                {
                    "items": [
                        {
                            "video_id": "test-a",
                            "video_name": "test-a.MOV",
                            "video_path": str(video_path),
                            "split": "test_frozen",
                        }
                    ]
                },
            )
            write_json(
                root / "audio_candidates" / "test-a.touch_candidates.json",
                {
                    "audio_candidates": [
                        {"time_sec": 0.1, "strength": 1.0},
                        {"time_sec": 0.3, "strength": 1.0},
                    ],
                    "existing_event_hints": [],
                    "generated_event_hints": [
                        {"time_sec": 0.2, "event_type": "touch", "source": "generated_touch"}
                    ],
                },
            )
            label_path = root / "visual_touch_labels" / "test-a.events.json"
            write_json(
                label_path,
                {
                    "schema_version": 1,
                    "source_video": "test-a.MOV",
                    "split": "test_frozen",
                    "annotation_method": "muted_visual_touch_review",
                    "audio_muted_during_review_required": True,
                    "candidate_review_complete": False,
                    "candidate_reviews": [{"time_sec": 0.1, "decision": "no_touch", "review_status": "reviewed"}],
                    "rallies": [{"id": 1, "events": []}],
                },
            )

            manifest = sheets.build_contact_sheets(self.make_args(root, root / "sheets"))

            self.assertEqual(manifest["summary"]["videos_considered"], 1)
            self.assertEqual(manifest["summary"]["sheets_rendered"], 2)
            self.assertEqual(manifest["summary"]["candidates_rendered"], 2)
            by_filter = {sheet["filter"]: sheet for sheet in manifest["videos"][0]["sheets"]}
            self.assertEqual(by_filter["likely_unchecked"]["times_sec"], [0.2])
            self.assertEqual(by_filter["audio_only"]["times_sec"], [0.3])
            for sheet in by_filter.values():
                image_path = Path(sheet["path"])
                self.assertTrue(image_path.exists())
                with Image.open(image_path) as image:
                    self.assertGreater(image.width, 0)
            after = json.loads(label_path.read_text(encoding="utf-8"))
            self.assertFalse(after["candidate_review_complete"])
            self.assertEqual(len(after["candidate_reviews"]), 1)


if __name__ == "__main__":
    unittest.main()
