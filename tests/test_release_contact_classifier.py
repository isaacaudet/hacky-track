from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import train_release_contact_classifier as contact


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class ReleaseContactClassifierTests(unittest.TestCase):
    def test_pose_candidate_maps_lower_body_part_to_soft_type_and_side(self) -> None:
        row = {
            "pose_nearest_lower_part": "left_big_toe",
            "pose_nearest_foot_part": "left_ankle",
        }

        out = contact.pose_candidate(row)

        self.assertEqual(out["pose_candidate_type"], "kick")
        self.assertEqual(out["pose_candidate_side"], "left")

    def test_readiness_blocks_when_pose_and_contact_labels_are_missing(self) -> None:
        rows = [{"video_id": "a", "candidate_time_sec": 1.0}]

        summary = contact.readiness_summary(rows, [], min_examples=1, min_videos=1)

        self.assertEqual(summary["status"], "not_ready")
        self.assertTrue(any("pose" in reason for reason in summary["reasons"]))
        self.assertTrue(any("contact labels" in reason for reason in summary["reasons"]))

    def test_readiness_passes_when_pose_features_and_clip_disjoint_labels_exist(self) -> None:
        rows = [
            {
                "video_id": "a",
                "contact_type": "left_kick",
                "contact_side": "left",
                "pose_nearest_foot_dist_px": 12.0,
                "pose_nearest_lower_part": "left_big_toe",
            },
            {
                "video_id": "b",
                "contact_type": "right_kick",
                "contact_side": "right",
                "pose_nearest_foot_dist_px": 14.0,
                "pose_nearest_lower_part": "right_big_toe",
            },
        ]

        summary = contact.readiness_summary(rows, [], min_examples=2, min_videos=2)

        self.assertEqual(summary["status"], "ready_for_training")
        self.assertEqual(summary["contact_side_counts_in_rows"], {"left": 1, "right": 1})
        self.assertEqual(summary["contact_type_counts_in_rows"], {"kick": 2})

    def test_loads_stall_contact_examples_from_reviewed_event_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "video-a.events.json",
                {
                    "source_video": "video-a.MOV",
                    "rallies": [
                        {
                            "id": 1,
                            "events": [
                                {"type": "stall", "time_sec": 2.0, "review_status": "approved"},
                                {"type": "touch", "time_sec": 3.0, "review_status": "approved"},
                            ],
                        }
                    ],
                },
            )

            examples = contact.load_label_contact_examples(root)

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0]["contact_type"], "stall")


if __name__ == "__main__":
    unittest.main()
