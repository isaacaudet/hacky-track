from __future__ import annotations

import unittest

import cv2
import numpy as np

from qa_rally_enrichment import BallCandidate, OUT_SIZE, classify_contact, side_evidence_from_contact
from render_best_rally_hud import rally_selection_notes


def synthetic_contact_frame(foot_x: int, ball_x: int) -> np.ndarray:
    frame = np.zeros((OUT_SIZE[1], OUT_SIZE[0], 3), dtype=np.uint8)
    frame[:, :] = (70, 145, 70)
    cv2.rectangle(frame, (foot_x - 42, 735), (foot_x + 42, 850), (28, 22, 72), -1)
    cv2.circle(frame, (ball_x, 724), 12, (0, 40, 220), -1)
    return frame


def synthetic_cross_body_frame() -> np.ndarray:
    frame = np.zeros((OUT_SIZE[1], OUT_SIZE[0], 3), dtype=np.uint8)
    frame[:, :] = (70, 145, 70)
    contour = np.array(
        [
            [392, 910],
            [430, 910],
            [390, 785],
            [318, 748],
            [276, 782],
            [306, 842],
            [356, 844],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [contour], (28, 22, 72))
    cv2.circle(frame, (292, 812), 12, (0, 40, 220), -1)
    return frame


class ContactGatingTests(unittest.TestCase):
    def test_reliable_foot_contact_uses_left_right_not_center(self) -> None:
        left_frame = synthetic_contact_frame(258, 258)
        *_prefix, left_side, left_type, _conf = classify_contact(left_frame, "touch", BallCandidate(258, 724, 12, "red_global", 0.8))
        self.assertEqual(left_side, "left")
        self.assertEqual(left_type, "foot")

        right_frame = synthetic_contact_frame(430, 430)
        *_prefix, right_side, right_type, _conf = classify_contact(
            right_frame,
            "touch",
            BallCandidate(430, 724, 12, "red_global", 0.8),
        )
        self.assertEqual(right_side, "right")
        self.assertEqual(right_type, "foot")

    def test_centerline_foot_contact_stays_unknown_side(self) -> None:
        frame = synthetic_contact_frame(344, 344)
        *_prefix, side, contact_type, _conf = classify_contact(frame, "touch", BallCandidate(344, 724, 12, "red_global", 0.8))
        self.assertEqual(side, "unknown")
        self.assertEqual(contact_type, "foot")
        side_conf, side_source, uncertainty = side_evidence_from_contact(344, 1.0, 0.0, side, contact_type)
        self.assertGreater(side_conf, 0.0)
        self.assertEqual(side_source, "limb_contour_geometry")
        self.assertEqual(uncertainty, "limb_near_frame_centerline")

    def test_ball_position_does_not_flip_centerline_foot_contact(self) -> None:
        frame = synthetic_contact_frame(344, 280)
        *_prefix, side, contact_type, _conf = classify_contact(frame, "touch", BallCandidate(280, 724, 12, "red_global", 0.8))
        self.assertEqual(side, "unknown")
        self.assertEqual(contact_type, "foot")
        side_conf, side_source, uncertainty = side_evidence_from_contact(344, 1.0, 0.0, side, contact_type, 280)
        self.assertGreater(side_conf, 0.0)
        self.assertEqual(side_source, "limb_contour_geometry")
        self.assertEqual(uncertainty, "limb_near_frame_centerline")

    def test_lower_limb_anchor_handles_cross_body_right_foot(self) -> None:
        frame = synthetic_cross_body_frame()
        *_prefix, side, contact_type, _conf = classify_contact(frame, "touch", BallCandidate(292, 812, 12, "red_global", 0.8))
        self.assertEqual(side, "right")
        self.assertEqual(contact_type, "foot")

    def test_side_evidence_marks_confident_off_center_side(self) -> None:
        side_conf, side_source, uncertainty = side_evidence_from_contact(430, 0.9, 8.0, "right", "foot")
        self.assertGreaterEqual(side_conf, 0.65)
        self.assertEqual(side_source, "limb_contour_geometry")
        self.assertIsNone(uncertainty)

    def test_high_airborne_contact_does_not_become_foot(self) -> None:
        frame = np.zeros((OUT_SIZE[1], OUT_SIZE[0], 3), dtype=np.uint8)
        frame[:, :] = (70, 145, 70)
        cv2.rectangle(frame, (372, 430), (455, 500), (28, 22, 72), -1)
        cv2.circle(frame, (414, 482), 12, (0, 40, 220), -1)
        *_prefix, side, contact_type, _conf = classify_contact(frame, "touch", BallCandidate(414, 482, 12, "red_global", 0.8))
        self.assertNotEqual(side, "right")
        self.assertNotEqual(contact_type, "foot")

    def test_foot_candidate_blocks_strict_best_rally_selection(self) -> None:
        events = [{"type": "touch", "time_sec": float(idx), "contact_type": "foot"} for idx in range(6)]
        events[-1]["contact_type"] = "foot_candidate"
        notes = rally_selection_notes(events)
        self.assertFalse(notes["strict_complete"])
        self.assertEqual(notes["ambiguous_contact_events"], 1)

    def test_gap_without_floor_reset_blocks_strict_best_rally_selection(self) -> None:
        events = [{"type": "touch", "time_sec": float(idx), "contact_type": "foot"} for idx in range(6)]
        notes = rally_selection_notes(events, {"ended_by_gap_without_floor_reset": True, "next_contact_gap_sec": 4.2})
        self.assertFalse(notes["strict_complete"])
        self.assertTrue(notes["ended_by_gap_without_floor_reset"])

    def test_terminal_drop_is_allowed_as_strict_boundary(self) -> None:
        events = [{"type": "touch", "time_sec": float(idx), "contact_type": "foot"} for idx in range(6)]
        events.append({"type": "drop_floor", "time_sec": 6.2, "contact_type": "ground"})
        notes = rally_selection_notes(events, {"end_reason": "drop_floor"})
        self.assertTrue(notes["strict_complete"])
        self.assertEqual(notes["terminal_drop_floor_events"], 1)
        self.assertEqual(notes["nonterminal_drop_floor_events"], 0)


if __name__ == "__main__":
    unittest.main()
