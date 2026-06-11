import unittest

import side_semantics_audit as audit


class SideSemanticsAuditTest(unittest.TestCase):
    def test_opposite_maps_only_valid_sides(self) -> None:
        self.assertEqual(audit.opposite("left"), "right")
        self.assertEqual(audit.opposite("right"), "left")
        self.assertIsNone(audit.opposite(None))
        self.assertIsNone(audit.opposite("unknown"))

    def test_screen_side_uses_deadzone_around_center(self) -> None:
        self.assertEqual(audit.screen_side({"visual_ball_x_norm": 0.25}), "left")
        self.assertEqual(audit.screen_side({"visual_ball_x_norm": 0.75}), "right")
        self.assertIsNone(audit.screen_side({"visual_ball_x_norm": 0.51}, deadzone=0.08))
        self.assertEqual(audit.screen_side({"visual_ball_x_norm": 0.59}, deadzone=0.08), "right")

    def test_screen_side_can_derive_from_pixel_coordinates(self) -> None:
        row = {"visual_crop_ball_x": 75, "visual_frame_width_px": 100}
        self.assertEqual(audit.screen_side(row), "right")

    def test_score_predictor_counts_coverage_and_accuracy(self) -> None:
        rows = [
            {"contact_side": "left", "visual_ball_x_norm": 0.25},
            {"contact_side": "right", "visual_ball_x_norm": 0.75},
            {"contact_side": "right", "visual_ball_x_norm": 0.25},
            {"contact_side": "left", "visual_ball_x_norm": None},
        ]
        score = audit.score_predictor(rows, "screen_ball")
        self.assertEqual(score["rows"], 4)
        self.assertEqual(score["covered"], 3)
        self.assertEqual(score["correct"], 2)
        self.assertAlmostEqual(score["coverage"], 0.75)
        self.assertAlmostEqual(score["accuracy"], 2 / 3)
        self.assertEqual(score["confusion"], {"left": {"left": 1}, "right": {"right": 1, "left": 1}})

    def test_best_mapping_prefers_accuracy_then_coverage(self) -> None:
        scores = {
            "low_accuracy": {"predictor": "low_accuracy", "covered": 10, "accuracy": 0.6, "coverage": 1.0},
            "high_accuracy": {"predictor": "high_accuracy", "covered": 5, "accuracy": 0.8, "coverage": 0.5},
            "unused": {"predictor": "unused", "covered": 0, "accuracy": None, "coverage": 0.0},
        }
        self.assertEqual(audit.best_mapping(scores)["predictor"], "high_accuracy")


if __name__ == "__main__":
    unittest.main()
