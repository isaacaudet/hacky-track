from __future__ import annotations

import unittest

import release_event_error_audit as audit


class ReleaseEventErrorAuditTests(unittest.TestCase):
    def test_false_positive_near_reviewed_touch_is_duplicate_bucket(self) -> None:
        mode = audit.classify_release_error(
            {"error_type": "false_positive", "time_sec": 1.42},
            event_features={"audio_strength": 8.0, "trajectory_break_support": 2},
            candidates=[],
            touch_delta=0.32,
            stall_delta=None,
            predicted_delta=0.0,
        )

        self.assertEqual(mode, "duplicate-after-touch not merged enough")

    def test_false_positive_near_stall_is_stall_control_bucket(self) -> None:
        mode = audit.classify_release_error(
            {"error_type": "false_positive", "time_sec": 2.0},
            event_features={"audio_strength": 12.0, "trajectory_break_support": 4},
            candidates=[],
            touch_delta=1.2,
            stall_delta=0.1,
            predicted_delta=0.0,
        )

        self.assertEqual(mode, "stall/control mistaken as touch")

    def test_loud_reviewed_no_touch_with_motion_is_footstep_bucket(self) -> None:
        mode = audit.classify_release_error(
            {"error_type": "false_positive", "time_sec": 4.0},
            event_features={"audio_strength": 16.0, "trajectory_break_support": 3, "trajectory_impulse_score": 800},
            candidates=[{"candidate_review_decision": "no_touch"}],
            touch_delta=1.0,
            stall_delta=None,
            predicted_delta=0.0,
        )

        self.assertEqual(mode, "loud footstep with ball motion nearby")

    def test_false_negative_with_late_prediction_is_duplicate_bucket(self) -> None:
        mode = audit.classify_release_error(
            {"error_type": "false_negative", "time_sec": 7.9},
            event_features=None,
            candidates=[],
            touch_delta=0.0,
            stall_delta=None,
            predicted_delta=0.48,
        )

        self.assertEqual(mode, "duplicate-after-touch not merged enough")


if __name__ == "__main__":
    unittest.main()
