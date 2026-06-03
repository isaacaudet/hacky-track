from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import release_rally_analytics as analytics


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class ReleaseRallyAnalyticsTests(unittest.TestCase):
    def test_reports_touch_false_positive_and_false_negative_times(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hud_doc = {
                "source_video": "video-test.MOV",
                "rallies": [
                    {
                        "id": 1,
                        "label": "release rally 1",
                        "start_sec": 0.8,
                        "end_sec": 4.2,
                        "events": [
                            {"type": "touch", "time_sec": 1.0},
                            {"type": "touch", "time_sec": 2.0},
                            {"type": "touch", "time_sec": 4.0},
                        ],
                    }
                ],
            }
            write_json(root / "hud" / "video-test" / "release_touch_hud_events.json", hud_doc)
            write_json(
                root / "labels" / "video-test.events.json",
                {
                    "source_video": "video-test.MOV",
                    "rallies": [
                        {
                            "id": 1,
                            "events": [
                                {"type": "touch", "time_sec": 1.05, "review_status": "approved"},
                                {"type": "touch", "time_sec": 3.0, "review_status": "approved"},
                            ],
                        }
                    ],
                },
            )

            summary, errors = analytics.analyze_video(
                root / "hud" / "video-test" / "release_touch_hud_events.json",
                root / "labels",
                root / "legacy",
                0.2,
            )

            self.assertEqual(summary["touch_metrics"]["true_positive"], 1)
            self.assertEqual(summary["touch_metrics"]["false_positive"], 2)
            self.assertEqual(summary["touch_metrics"]["false_negative"], 1)
            self.assertEqual(summary["false_positive_times_sec"], [2.0, 4.0])
            self.assertEqual(summary["false_negative_times_sec"], [3.0])
            self.assertEqual(len(errors), 3)
            self.assertEqual(summary["best_rally"]["touches"], 3)


if __name__ == "__main__":
    unittest.main()
