import unittest

from precision_gate import evaluate_precision_gate


class PrecisionGateTests(unittest.TestCase):
    def test_clear_foot_touch_is_not_vetoed(self):
        decision = evaluate_precision_gate(
            {
                "type": "touch",
                "contact_type": "foot",
                "contact_confidence": 0.95,
                "foot_distance": 70,
                "audio_z": 32,
                "visual_score": 0.7,
                "motion_score": 0.5,
                "qa_ball_y": 720,
                "yolo_v10t_agreement_px": 90,
                "velocity_dvy_at_touch_moment": -800,
            }
        )
        self.assertFalse(decision.veto)
        self.assertEqual(decision.reasons, ())

    def test_ambiguous_foot_candidate_with_tracker_disagreement_is_soft_flagged(self):
        decision = evaluate_precision_gate(
            {
                "type": "touch",
                "contact_type": "foot_candidate",
                "contact_confidence": 0.96,
                "foot_distance": 158,
                "audio_z": 70,
                "visual_score": 0.9,
                "motion_score": 1.0,
                "qa_ball_y": 853,
                "yolo_v10t_agreement_px": 210,
                "velocity_dvy_at_touch_moment": -1500,
            }
        )
        self.assertFalse(decision.veto)
        self.assertFalse(decision.hard_veto)
        self.assertTrue(decision.soft_flag)
        self.assertIn("ambiguous_foot_candidate_tracker_disagreement", decision.reasons)

    def test_unknown_contact_far_from_limb_needs_weak_support_for_hard_veto(self):
        decision = evaluate_precision_gate(
            {
                "type": "touch",
                "contact_type": "unknown_contact",
                "contact_confidence": 0.45,
                "foot_distance": 260,
                "audio_z": 15,
                "visual_score": 0.35,
                "motion_score": 0.2,
                "qa_ball_y": 666,
                "yolo_v10t_agreement_px": 17,
            }
        )
        self.assertTrue(decision.veto)
        self.assertTrue(decision.hard_veto)
        self.assertTrue(decision.soft_flag)
        self.assertIn("hard_weak_nonfoot_multimodal", decision.hard_reasons)
        self.assertIn("nonfoot_contact_far_or_weak", decision.reasons)

    def test_unknown_contact_with_strong_evidence_is_soft_flagged_only(self):
        decision = evaluate_precision_gate(
            {
                "type": "touch",
                "contact_type": "unknown_contact",
                "contact_confidence": 0.83,
                "foot_distance": 260,
                "audio_z": 41,
                "visual_score": 0.95,
                "motion_score": 0.48,
                "qa_ball_y": 760,
                "yolo_v10t_agreement_px": 17,
            }
        )
        self.assertFalse(decision.veto)
        self.assertFalse(decision.hard_veto)
        self.assertTrue(decision.soft_flag)

    def test_floor_candidate_with_severe_tracker_disagreement_is_vetoed(self):
        decision = evaluate_precision_gate(
            {
                "type": "drop_floor",
                "time_sec": 4.43,
                "contact_type": "ground",
                "foot_distance": 360,
                "visual_score": 0.0,
                "qa_ball_y": 792,
                "yolo_v10t_agreement_px": 426,
                "velocity_dvy_at_touch_moment": 564,
            }
        )
        self.assertTrue(decision.veto)
        self.assertIn("severe_floor_tracker_disagreement", decision.reasons)


if __name__ == "__main__":
    unittest.main()
