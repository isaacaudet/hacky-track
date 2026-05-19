from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import assist_detector_label_decisions


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_crop(path: Path, *, red_ball: bool) -> None:
    image = np.full((192, 192, 3), (80, 140, 80), dtype=np.uint8)
    if red_ball:
        cv2.circle(image, (96, 96), 18, (0, 0, 210), -1)
        cv2.circle(image, (88, 92), 7, (30, 30, 30), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def write_panel_ball_crop(path: Path) -> None:
    image = np.full((192, 192, 3), (80, 140, 80), dtype=np.uint8)
    cv2.circle(image, (96, 96), 22, (180, 150, 70), -1)
    cv2.ellipse(image, (87, 96), (12, 20), 0, 90, 270, (195, 190, 75), -1)
    cv2.ellipse(image, (105, 96), (12, 20), 0, -90, 90, (65, 180, 220), -1)
    cv2.line(image, (96, 74), (96, 118), (35, 55, 45), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def write_skin_crop(path: Path) -> None:
    image = np.full((192, 192, 3), (80, 140, 80), dtype=np.uint8)
    cv2.circle(image, (96, 96), 34, (95, 145, 210), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


class AssistDetectorLabelDecisionsTests(unittest.TestCase):
    def test_assists_obvious_centered_footbag_and_leaves_ambiguous_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop_good = root / "crops" / "good.jpg"
            crop_plain = root / "crops" / "plain.jpg"
            write_crop(crop_good, red_ball=True)
            write_crop(crop_plain, red_ball=False)
            manifest = root / "detector_label_review_manifest.json"
            write_json(
                manifest,
                {
                    "items": [
                        {
                            "detector_label_id": "good",
                            "source_video": "clip.MOV",
                            "crop_path": str(crop_good),
                            "radius": 18,
                            "suggested_detector_label": "footbag",
                            "qa_ball_confidence": 0.9,
                            "qa_ball_correction_px": 10,
                        },
                        {
                            "detector_label_id": "plain",
                            "source_video": "clip.MOV",
                            "crop_path": str(crop_plain),
                            "radius": 18,
                            "suggested_detector_label": "footbag",
                            "qa_ball_confidence": 0.9,
                            "qa_ball_correction_px": 10,
                        },
                    ]
                },
            )
            summary = assist_detector_label_decisions.build_assisted_decisions(manifest, root / "decisions.json")
            by_id = {item["detector_label_id"]: item for item in summary["decisions"]}
            self.assertEqual(by_id["good"]["detector_status"], "footbag")
            self.assertEqual(by_id["plain"]["detector_status"], "pending")
            self.assertTrue((root / "decisions.json").exists())

    def test_assists_blue_yellow_panel_footbag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop_good = root / "crops" / "panel.jpg"
            write_panel_ball_crop(crop_good)
            manifest = root / "manifest.json"
            write_json(
                manifest,
                {
                    "items": [
                        {
                            "detector_label_id": "panel",
                            "source_video": "clip.MOV",
                            "crop_path": str(crop_good),
                            "radius": 20,
                            "suggested_detector_label": "footbag",
                        }
                    ]
                },
            )
            summary = assist_detector_label_decisions.build_assisted_decisions(manifest, root / "decisions.json")
            self.assertEqual(summary["decisions"][0]["detector_status"], "footbag")

    def test_marks_obvious_bad_center_only_when_verify_or_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop_plain = root / "crops" / "plain.jpg"
            write_crop(crop_plain, red_ball=False)
            manifest = root / "manifest.json"
            write_json(
                manifest,
                {
                    "items": [
                        {
                            "detector_label_id": "bad",
                            "source_video": "clip.MOV",
                            "crop_path": str(crop_plain),
                            "radius": 18,
                            "suggested_detector_label": "verify_or_correct",
                            "qa_ball_confidence": 0.5,
                            "qa_ball_correction_px": 180,
                        }
                    ]
                },
            )
            summary = assist_detector_label_decisions.build_assisted_decisions(manifest, root / "decisions.json")
            self.assertEqual(summary["decisions"][0]["detector_status"], "not_footbag")

    def test_does_not_call_skin_blob_a_footbag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop_skin = root / "crops" / "skin.jpg"
            write_skin_crop(crop_skin)
            manifest = root / "manifest.json"
            write_json(
                manifest,
                {
                    "items": [
                        {
                            "detector_label_id": "skin",
                            "source_video": "clip.MOV",
                            "crop_path": str(crop_skin),
                            "radius": 20,
                            "suggested_detector_label": "not_footbag",
                        }
                    ]
                },
            )
            summary = assist_detector_label_decisions.build_assisted_decisions(manifest, root / "decisions.json")
            self.assertEqual(summary["decisions"][0]["detector_status"], "not_footbag")

    def test_marks_suggested_hard_negative_without_color_as_not_footbag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop_plain = root / "crops" / "plain.jpg"
            write_crop(crop_plain, red_ball=False)
            manifest = root / "manifest.json"
            write_json(
                manifest,
                {
                    "items": [
                        {
                            "detector_label_id": "fp",
                            "source_video": "clip.MOV",
                            "crop_path": str(crop_plain),
                            "radius": 18,
                            "suggested_detector_label": "not_footbag",
                        }
                    ]
                },
            )
            summary = assist_detector_label_decisions.build_assisted_decisions(manifest, root / "decisions.json")
            self.assertEqual(summary["decisions"][0]["detector_status"], "not_footbag")


if __name__ == "__main__":
    unittest.main()
