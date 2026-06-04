from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from render_touch_release_hud import (
    TrackPoint,
    apply_touch_overrides,
    apply_reviewed_touch_contact_labels,
    build_hud_doc_and_anchors,
    labeled_release_events,
    load_touch_overrides,
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class RenderTouchReleaseHudTests(unittest.TestCase):
    def video(self) -> dict:
        return {
            "video_id": "video-test",
            "video_name": "video-test.MOV",
            "video_path": "/tmp/video-test.MOV",
            "split": "train",
            "metadata": {"width": 1000, "height": 500, "duration_sec": 10.0},
        }

    def test_reviewed_stall_and_drop_labels_are_rendered_without_touch_count_inflation(self) -> None:
        points = [
            TrackPoint(time_sec=t, x=100.0 + t * 10.0, y=200.0, confidence=0.9)
            for t in [0.8, 1.0, 1.2, 2.0, 2.4, 3.0, 3.2]
        ]
        events = [
            {"event_type": "touch", "time_sec": 1.0, "confidence": 0.8},
            {"type": "stall", "time_sec": 2.0, "duration_sec": 1.2, "review_status": "approved"},
            {"type": "drop_floor", "time_sec": 3.0, "review_status": "approved"},
        ]

        doc, anchors, summary = build_hud_doc_and_anchors(
            self.video(),
            events,
            points,
            points,
            rally_gap_sec=2.2,
            max_track_gap_sec=0.4,
        )

        flat = doc["events"]
        self.assertEqual([event["type"] for event in flat], ["touch", "stall", "drop_floor"])
        self.assertEqual(doc["rallies"][0]["expected_touches"], 1)
        self.assertEqual(doc["rallies"][0]["expected_stalls"], 1)
        self.assertEqual(summary["rendered_touch_events"], 1)
        self.assertEqual(summary["rendered_stall_events"], 1)
        self.assertEqual(summary["rendered_drop_floor_events"], 1)
        self.assertEqual(len(anchors["anchors"]), 3)

    def test_label_loader_deduplicates_visual_and_legacy_non_touch_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = self.video()
            label_doc = {
                "source_video": "video-test.MOV",
                "rallies": [{"id": 1, "events": [{"type": "stall", "time_sec": 2.0, "review_status": "approved"}]}],
            }
            write_json(root / "labels" / "video-test.events.json", label_doc)
            write_json(root / "legacy" / "video-test.events.json", label_doc)

            events = labeled_release_events(video, root / "labels", root / "legacy")

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["type"], "stall")

    def test_touch_overrides_remove_and_add_reviewed_corrections(self) -> None:
        overrides = {
            "video-test": {
                "remove_touch_times_sec": [2.0],
                "add_touch_times_sec": [3.0],
                "reason": "visual audit",
            }
        }
        events = [
            {"event_type": "touch", "time_sec": 1.0, "confidence": 0.9},
            {"event_type": "touch", "time_sec": 2.03, "confidence": 0.8},
        ]

        corrected, summary = apply_touch_overrides(events, "video-test", overrides, tolerance_sec=0.08)

        self.assertEqual([event["time_sec"] for event in corrected], [1.0, 3.0])
        self.assertEqual(summary["removed_touch_times_sec"], [2.03])
        self.assertEqual(summary["added_touch_times_sec"], [3.0])
        self.assertEqual(corrected[-1]["event_match_type"], "visual_override")

    def test_load_touch_overrides_accepts_video_overrides_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overrides.json"
            write_json(path, {"video_overrides": {"video-test": {"remove_touch_times_sec": [1.0]}}})

            overrides = load_touch_overrides(path)

            self.assertEqual(overrides["video-test"]["remove_touch_times_sec"], [1.0])

    def test_reviewed_contact_labels_are_attached_to_matching_touch_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = self.video()
            label_doc = {
                "source_video": "video-test.MOV",
                "rallies": [
                    {
                        "id": 1,
                        "events": [
                            {
                                "type": "touch",
                                "time_sec": 1.02,
                                "review_status": "approved",
                                "trick_label": "left_inner_kick",
                                "contact_type": "kick",
                                "contact_side": "left",
                                "contact_side_basis": "wearer_limb",
                                "contact_surface": "inner",
                            }
                        ],
                    }
                ],
            }
            write_json(root / "labels" / "video-test.events.json", label_doc)
            touches = [{"event_type": "touch", "time_sec": 1.0, "confidence": 0.9}]

            enriched, summary = apply_reviewed_touch_contact_labels(
                touches,
                video,
                root / "labels",
                root / "legacy",
                tolerance_sec=0.2,
            )

            self.assertEqual(summary["reviewed_touch_labels_applied"], 1)
            self.assertEqual(summary["reviewed_contact_side_labels"], 1)
            self.assertEqual(summary["reviewed_contact_surface_labels"], 1)
            self.assertEqual(enriched[0]["contact_label"], "left_inner_kick")
            self.assertEqual(enriched[0]["contact_side"], "left")
            self.assertEqual(enriched[0]["contact_side_basis"], "wearer_limb")
            self.assertTrue(enriched[0]["manual_contact_label"])

    def test_reviewed_touch_without_contact_detail_does_not_create_badge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = self.video()
            label_doc = {
                "source_video": "video-test.MOV",
                "rallies": [
                    {
                        "id": 1,
                        "events": [{"type": "touch", "time_sec": 1.02, "review_status": "approved"}],
                    }
                ],
            }
            write_json(root / "labels" / "video-test.events.json", label_doc)
            touches = [{"event_type": "touch", "time_sec": 1.0, "confidence": 0.9}]

            enriched, summary = apply_reviewed_touch_contact_labels(
                touches,
                video,
                root / "labels",
                root / "legacy",
                tolerance_sec=0.2,
            )

            self.assertEqual(summary["reviewed_touch_labels_available"], 0)
            self.assertNotIn("manual_contact_label", enriched[0])

    def test_hud_doc_marks_reviewed_contact_badges_as_manual(self) -> None:
        points = [
            TrackPoint(time_sec=t, x=100.0 + t * 10.0, y=200.0, confidence=0.9)
            for t in [0.8, 1.0, 1.2]
        ]
        events = [
            {
                "event_type": "touch",
                "time_sec": 1.0,
                "confidence": 0.8,
                "contact_label": "right_outer_kick",
                "contact_label_source": "reviewed_visual_label",
                "contact_label_delta_sec": 0.02,
                "contact_type": "kick",
                "contact_side": "right",
                "contact_side_basis": "wearer_limb",
                "contact_surface": "outer",
                "manual_contact_label": True,
            }
        ]

        doc, _anchors, summary = build_hud_doc_and_anchors(
            self.video(),
            events,
            points,
            points,
            rally_gap_sec=2.2,
            max_track_gap_sec=0.4,
        )

        event = doc["events"][0]
        self.assertEqual(event["label"], "right_outer_kick")
        self.assertEqual(event["contact_label_source"], "reviewed_visual_label")
        self.assertEqual(event["contact_side"], "right")
        self.assertEqual(event["contact_surface"], "outer")
        self.assertTrue(event["manual_contact_label"])
        self.assertEqual(summary["manual_contact_badge_events"], 1)
        self.assertEqual(summary["manual_contact_side_badges"], 1)
        self.assertEqual(summary["manual_contact_surface_badges"], 1)


if __name__ == "__main__":
    unittest.main()
