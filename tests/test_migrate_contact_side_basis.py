import tempfile
import unittest
from pathlib import Path

import migrate_contact_side_basis as migrate


class ContactSideBasisMigrationTest(unittest.TestCase):
    def test_migrates_matching_side_specific_trick_labels_only(self) -> None:
        doc = {
            "rallies": [
                {
                    "events": [
                        {
                            "type": "touch",
                            "review_status": "approved",
                            "time_sec": 1.0,
                            "contact_side": "left",
                            "trick_label": "left_outer_kick",
                        },
                        {
                            "type": "touch",
                            "review_status": "approved",
                            "time_sec": 2.0,
                            "contact_side": "left",
                            "trick_label": "right_kick",
                        },
                        {
                            "type": "touch",
                            "review_status": "approved",
                            "time_sec": 3.0,
                            "contact_side": "right",
                            "trick_label": "right_kick",
                            "contact_side_basis": "screen_position",
                        },
                    ]
                }
            ]
        }

        migrated, changes = migrate.migrate_doc(doc)
        events = migrated["rallies"][0]["events"]

        self.assertEqual(len(changes), 1)
        self.assertEqual(events[0]["contact_side_basis"], "wearer_limb")
        self.assertNotIn("contact_side_basis", events[1])
        self.assertEqual(events[2]["contact_side_basis"], "screen_position")

    def test_run_migration_writes_report_and_updates_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels"
            labels.mkdir()
            path = labels / "video-a.events.json"
            migrate.write_json(
                path,
                {
                    "rallies": [
                        {
                            "events": [
                                {
                                    "type": "touch",
                                    "review_status": "approved",
                                    "time_sec": 1.0,
                                    "contact_side": "right",
                                    "trick_label": "right_inner_kick",
                                }
                            ]
                        }
                    ]
                },
            )

            report = migrate.run_migration(labels, root / "report.json", dry_run=False)
            updated = migrate.read_json(path)

            self.assertEqual(report["events_changed"], 1)
            self.assertEqual(updated["rallies"][0]["events"][0]["contact_side_basis"], "wearer_limb")


if __name__ == "__main__":
    unittest.main()
