from __future__ import annotations

import unittest

import oracle_detector_spike


class OracleDetectorSpikeMoondreamTests(unittest.TestCase):
    def test_normalizes_moondream_boxes_to_pixel_centers(self) -> None:
        detections = oracle_detector_spike.normalize_moondream_objects(
            [{"x_min": 0.25, "y_min": 0.1, "x_max": 0.75, "y_max": 0.5}],
            image_width=200,
            image_height=100,
            prompt="a footbag",
            prompt_index=2,
            score=1.0,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["x"], 100.0)
        self.assertEqual(detections[0]["y"], 30.0)
        self.assertEqual(detections[0]["bbox"], [50.0, 10.0, 150.0, 50.0])
        self.assertEqual(detections[0]["prompt"], "a footbag")
        self.assertEqual(detections[0]["prompt_index"], 2)
        self.assertEqual(detections[0]["score"], 1.0)

    def test_dedupes_overlapping_moondream_boxes_from_prompt_variants(self) -> None:
        detections = [
            {"score": 1.0, "bbox": [10.0, 10.0, 30.0, 30.0], "prompt_index": 0, "object_index": 0},
            {"score": 1.0, "bbox": [11.0, 11.0, 31.0, 31.0], "prompt_index": 1, "object_index": 0},
            {"score": 1.0, "bbox": [100.0, 100.0, 120.0, 120.0], "prompt_index": 1, "object_index": 1},
        ]

        kept = oracle_detector_spike.dedupe_moondream_detections(detections, iou_threshold=0.7)

        self.assertEqual(kept, [detections[0], detections[2]])


if __name__ == "__main__":
    unittest.main()
