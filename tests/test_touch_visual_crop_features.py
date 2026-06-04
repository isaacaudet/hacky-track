from __future__ import annotations

import unittest

import numpy as np

import attach_touch_visual_crop_features as visual


class TouchVisualCropFeatureTests(unittest.TestCase):
    def test_extract_visual_crop_features_exposes_position_and_grid(self) -> None:
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[30:50, 40:60] = [0, 0, 255]
        ball = visual.BallPoint(time_sec=1.0, frame_index=30, x=50.0, y=40.0, score=0.8)

        features = visual.extract_visual_crop_features(
            frame,
            ball=ball,
            frame_index=30,
            crop_size_px=40,
            grid_size=4,
        )

        self.assertEqual(features["visual_crop_feature_status"], "ok")
        self.assertTrue(features["visual_crop_present"])
        self.assertAlmostEqual(features["visual_ball_x_norm"], 0.5)
        self.assertAlmostEqual(features["visual_ball_y_norm"], 0.5)
        self.assertIn("visual_crop_grid_0_0_hue_mean", features)
        self.assertIn("visual_crop_grid_3_3_edge_density", features)

    def test_default_visual_crop_features_preserves_missing_status(self) -> None:
        features = visual.default_visual_crop_features("missing_ball")

        self.assertEqual(features["visual_crop_feature_status"], "missing_ball")
        self.assertFalse(features["visual_crop_present"])
        self.assertTrue(features["visual_crop_ball_missing"])


if __name__ == "__main__":
    unittest.main()
