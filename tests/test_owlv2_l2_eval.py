import unittest

import owlv2_l2_eval


def record(frame: int, *, score: float | None, x: float = 10.0, y: float = 20.0, clip_id: str = "clip-a") -> dict:
    detections = [] if score is None else [{"score": score, "x": x, "y": y}]
    return {
        "clip_id": clip_id,
        "source_video": "video.mov",
        "frame_index": frame,
        "time_sec": frame / 30.0,
        "visibility": "visible",
        "is_target": True,
        "is_no_target": False,
        "detections": detections,
    }


class Owlv2L2EvalTests(unittest.TestCase):
    def test_l2_fills_short_supported_gap_only(self) -> None:
        records = [
            record(1, score=0.4, x=10, y=10),
            record(2, score=0.08, x=20, y=20),
            record(3, score=0.4, x=30, y=30),
        ]

        predictions, diagnostics = owlv2_l2_eval.l2_fill_and_clean(
            records,
            threshold=0.2,
            candidate_floor=0.05,
            support_radius_px=20.0,
            max_gap_frames=2,
            max_gap_sec=0.2,
            clean=False,
        )

        self.assertEqual(diagnostics["filled_count"], 1)
        self.assertEqual(len([row for row in predictions if row["source"] == "owlv2_l2_fill"]), 1)

    def test_l2_does_not_fill_without_detector_support(self) -> None:
        records = [
            record(1, score=0.4, x=10, y=10),
            record(2, score=None),
            record(3, score=0.4, x=30, y=30),
        ]

        predictions, diagnostics = owlv2_l2_eval.l2_fill_and_clean(
            records,
            threshold=0.2,
            candidate_floor=0.05,
            support_radius_px=20.0,
            max_gap_frames=2,
            max_gap_sec=0.2,
            clean=False,
        )

        self.assertEqual(diagnostics["filled_count"], 0)
        self.assertEqual(len(predictions), 2)

    def test_lock_on_metrics_count_no_target_runs(self) -> None:
        labels = [
            {
                "clip_id": "clip-a",
                "frame_index": 1,
                "visibility": "out_of_frame",
                "quality": "reviewed",
            },
            {
                "clip_id": "clip-a",
                "frame_index": 2,
                "visibility": "out_of_frame",
                "quality": "reviewed",
            },
            {
                "clip_id": "clip-a",
                "frame_index": 3,
                "visibility": "visible",
                "quality": "reviewed",
                "x": 100,
                "y": 100,
            },
        ]
        predictions = [
            {"clip_id": "clip-a", "frame_index": 1, "x": 1, "y": 1, "source": "owlv2_l2_fill"},
            {"clip_id": "clip-a", "frame_index": 2, "x": 2, "y": 2, "source": "owlv2_l2_fill"},
            {"clip_id": "clip-a", "frame_index": 3, "x": 101, "y": 100, "source": "owlv2_l2_anchor"},
        ]

        metrics = owlv2_l2_eval.lock_on_metrics(labels, predictions, tolerance_px=12.0)

        self.assertEqual(metrics["no_target_fp_frames"], 2)
        self.assertEqual(metrics["filled_no_target_fp_frames"], 2)
        self.assertEqual(metrics["longest_no_target_fp_run"], 2)
        self.assertEqual(metrics["longest_filled_no_target_fp_run"], 2)


if __name__ == "__main__":
    unittest.main()
