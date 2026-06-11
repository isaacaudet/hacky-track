import argparse
import json
import tempfile
import unittest
from pathlib import Path

from import_existing_touch_labels import import_existing_labels


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class ImportExistingTouchLabelsTests(unittest.TestCase):
    def test_imports_legacy_events_as_complete_review_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_path = root / "data" / "video-a.events.json"
            write_json(
                legacy_path,
                {
                    "source_video": "video-a.MOV",
                    "annotation_method": "user_corrected_hud_ground_truth",
                    "rallies": [
                        {
                            "events": [
                                {"type": "touch", "time_sec": 1.01, "confidence": "high"},
                                {"type": "drop_floor", "time_sec": 2.5},
                            ]
                        }
                    ],
                },
            )
            manifest_path = root / "touch_review_manifest.json"
            write_json(
                manifest_path,
                {
                    "items": [
                        {
                            "video_id": "video-a",
                            "video_name": "video-a.MOV",
                            "video_path": str(root / "video-a.MOV"),
                            "split": "train",
                            "existing_events_path": str(legacy_path),
                        }
                    ]
                },
            )
            candidates_dir = root / "audio_candidates"
            write_json(
                candidates_dir / "video-a.touch_candidates.json",
                {
                    "audio_candidates": [
                        {"time_sec": 1.0, "strength": 1.0},
                        {"time_sec": 3.0, "strength": 1.0},
                    ],
                    "existing_event_hints": [],
                },
            )
            labels_dir = root / "visual_touch_labels"

            summary = import_existing_labels(
                argparse.Namespace(
                    review_manifest=manifest_path,
                    candidates_dir=candidates_dir,
                    labels_dir=labels_dir,
                    touch_tolerance_sec=0.20,
                    cluster_gap_sec=0.04,
                    force=False,
                    dry_run=False,
                )
            )

            self.assertEqual(summary["imported"], 1)
            imported = json.loads((labels_dir / "video-a.events.json").read_text())
            self.assertEqual(imported["annotation_method"], "muted_visual_touch_review")
            self.assertTrue(imported["audio_muted_during_review_required"])
            self.assertTrue(imported["candidate_review_complete"])
            self.assertEqual(imported["legacy_import"]["source_annotation_method"], "user_corrected_hud_ground_truth")
            self.assertEqual([row["decision"] for row in imported["candidate_reviews"]], ["touch", "no_touch"])
            events = imported["rallies"][0]["events"]
            self.assertEqual([event["type"] for event in events], ["touch", "drop_floor"])

    def test_existing_label_is_not_overwritten_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_path = root / "data" / "video-a.events.json"
            write_json(legacy_path, {"source_video": "video-a.MOV", "rallies": [{"events": []}]})
            manifest_path = root / "touch_review_manifest.json"
            write_json(
                manifest_path,
                {
                    "items": [
                        {
                            "video_id": "video-a",
                            "video_name": "video-a.MOV",
                            "video_path": str(root / "video-a.MOV"),
                            "split": "train",
                            "existing_events_path": str(legacy_path),
                        }
                    ]
                },
            )
            candidates_dir = root / "audio_candidates"
            write_json(candidates_dir / "video-a.touch_candidates.json", {"audio_candidates": [], "existing_event_hints": []})
            labels_dir = root / "visual_touch_labels"
            write_json(labels_dir / "video-a.events.json", {"sentinel": True})

            summary = import_existing_labels(
                argparse.Namespace(
                    review_manifest=manifest_path,
                    candidates_dir=candidates_dir,
                    labels_dir=labels_dir,
                    touch_tolerance_sec=0.20,
                    cluster_gap_sec=0.04,
                    force=False,
                    dry_run=False,
                )
            )

            self.assertEqual(summary["skipped_existing_label"], 1)
            self.assertEqual(json.loads((labels_dir / "video-a.events.json").read_text()), {"sentinel": True})

    def test_import_ignores_generated_hints_until_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_path = root / "data" / "video-a.events.json"
            write_json(
                legacy_path,
                {
                    "source_video": "video-a.MOV",
                    "annotation_method": "user_corrected_hud_ground_truth",
                    "rallies": [{"events": [{"type": "touch", "time_sec": 1.0}]}],
                },
            )
            manifest_path = root / "touch_review_manifest.json"
            write_json(
                manifest_path,
                {
                    "items": [
                        {
                            "video_id": "video-a",
                            "video_name": "video-a.MOV",
                            "video_path": str(root / "video-a.MOV"),
                            "split": "train",
                            "existing_events_path": str(legacy_path),
                        }
                    ]
                },
            )
            candidates_dir = root / "audio_candidates"
            write_json(
                candidates_dir / "video-a.touch_candidates.json",
                {
                    "audio_candidates": [{"time_sec": 1.0, "strength": 1.0}],
                    "existing_event_hints": [],
                    "generated_event_hints": [{"time_sec": 2.0, "event_type": "touch", "source": "generated_touch"}],
                },
            )
            labels_dir = root / "visual_touch_labels"

            import_existing_labels(
                argparse.Namespace(
                    review_manifest=manifest_path,
                    candidates_dir=candidates_dir,
                    labels_dir=labels_dir,
                    touch_tolerance_sec=0.20,
                    cluster_gap_sec=0.04,
                    force=False,
                    dry_run=False,
                )
            )

            imported = json.loads((labels_dir / "video-a.events.json").read_text())
            self.assertEqual([row["time_sec"] for row in imported["candidate_reviews"]], [1.0])


if __name__ == "__main__":
    unittest.main()
