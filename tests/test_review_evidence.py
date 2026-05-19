from __future__ import annotations

import unittest

import prepare_review_evidence


class ReviewEvidenceTests(unittest.TestCase):
    def test_gap_and_side_risk_buckets_are_assigned(self) -> None:
        item = {
            "kind": "drop_floor",
            "review_tags": [
                "active_learning",
                "gap_without_floor_reset",
                "likely_missing_floor_reset",
                "strict_best_rally_blocker",
                "low_side_confidence",
            ],
        }
        buckets = prepare_review_evidence.item_buckets(item)
        self.assertIn("gap_floor_reset", buckets)
        self.assertIn("drop_review", buckets)
        self.assertIn("side_contact", buckets)
        self.assertIn("all_priority", buckets)

    def test_priority_promotes_unreviewed_strict_gap_blockers(self) -> None:
        item = {
            "priority_score": 2.0,
            "kind": "drop_floor",
            "review_tags": ["strict_best_rally_blocker", "gap_without_floor_reset", "likely_missed_drop_floor"],
        }
        score = prepare_review_evidence.priority_score(item, {"evidence_score": 1.25}, "unreviewed")
        self.assertGreaterEqual(score, 24.0)

    def test_gap_sequence_samples_across_whole_gap(self) -> None:
        item = {
            "time_sec": 14.5,
            "gap_start_sec": 10.0,
            "gap_duration_sec": 4.0,
        }
        times = prepare_review_evidence.sequence_times(item, 0.5)
        self.assertEqual(len(times), 5)
        self.assertLess(times[0], 10.0)
        self.assertIn(14.5, times)
        self.assertGreater(times[-1], 13.0)


if __name__ == "__main__":
    unittest.main()
