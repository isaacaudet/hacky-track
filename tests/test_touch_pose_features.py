from __future__ import annotations

import unittest

import numpy as np

import attach_touch_pose_features as pose
import train_release_contact_classifier as contact


class TouchPoseFeatureTests(unittest.TestCase):
    def test_all_feet_geometry_exposes_side_and_inner_outer_edges(self) -> None:
        keypoints = np.zeros((1, 23, 2), dtype=float)
        scores = np.zeros((1, 23), dtype=float)
        scores[:] = 0.0

        # Left foot: heel -> big toe is the inner edge; heel -> small toe is outer.
        keypoints[0, 13] = [0.0, -20.0]
        keypoints[0, 15] = [0.0, 0.0]
        keypoints[0, 17] = [0.0, 10.0]
        keypoints[0, 18] = [10.0, 10.0]
        keypoints[0, 19] = [0.0, 0.0]
        for index in (13, 15, 17, 18, 19):
            scores[0, index] = 0.9

        # Right foot is visible but far from the ball.
        keypoints[0, 14] = [100.0, -20.0]
        keypoints[0, 16] = [100.0, 0.0]
        keypoints[0, 20] = [100.0, 10.0]
        keypoints[0, 21] = [110.0, 10.0]
        keypoints[0, 22] = [100.0, 0.0]
        for index in (14, 16, 20, 21, 22):
            scores[0, index] = 0.9

        ball = pose.BallPoint(time_sec=1.0, frame_index=30, x=1.0, y=8.0, score=0.95)

        features = pose.all_feet_geometry_features(keypoints=keypoints, scores=scores, ball=ball, threshold=0.25)

        self.assertTrue(features["pose_geometry_present"])
        self.assertEqual(features["pose_geometry_nearest_side"], "left")
        self.assertEqual(features["pose_geometry_nearest_surface"], "inner")
        self.assertLess(features["pose_left_inner_edge_dist_px"], features["pose_left_outer_edge_dist_px"])
        self.assertGreater(features["pose_right_foot_min_dist_px"], features["pose_left_foot_min_dist_px"])

    def test_crop_descriptor_returns_low_dimensional_visual_features(self) -> None:
        frame = np.zeros((30, 30, 3), dtype=np.uint8)
        frame[5:20, 5:20] = [0, 0, 255]

        features = pose.crop_descriptor(frame, [(6.0, 6.0), (19.0, 19.0)], pad_px=2)

        self.assertGreater(features["crop_ball_foot_width_px"], 0)
        self.assertIn("crop_ball_foot_sat_mean", features)
        self.assertIn("crop_ball_foot_edge_density", features)

    def test_classifier_consumes_auto_geometry_and_crop_features(self) -> None:
        row = {
            "pose_left_foot_min_dist_px": 12.0,
            "pose_geometry_nearest_side": "left",
            "pose_geometry_nearest_surface": "inner",
            "crop_ball_foot_edge_density": 0.2,
            "visual_ball_x_norm": 0.25,
            "visual_crop_grid_0_0_sat_mean": 0.8,
        }

        features = contact.contact_feature_dict(row)

        self.assertEqual(features["pose_left_foot_min_dist_px"], 12.0)
        self.assertEqual(features["pose_geometry_nearest_side"], "left")
        self.assertEqual(features["pose_geometry_nearest_surface"], "inner")
        self.assertEqual(features["crop_ball_foot_edge_density"], 0.2)
        self.assertEqual(features["visual_ball_x_norm"], 0.25)
        self.assertEqual(features["visual_crop_grid_0_0_sat_mean"], 0.8)


if __name__ == "__main__":
    unittest.main()
