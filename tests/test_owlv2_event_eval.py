import unittest
from pathlib import Path

import numpy as np

import owlv2_event_eval


class Owlv2EventEvalAudioFusionTests(unittest.TestCase):
    def test_rally_audio_candidates_excludes_stalls_and_pre_drop_onsets(self) -> None:
        rally = {
            "start_sec": 0.0,
            "end_sec": 10.0,
            "events": [
                {"type": "stall", "time_sec": 2.0, "duration_sec": 0.5},
                {"type": "drop_floor", "time_sec": 9.5},
            ],
        }
        candidates = [
            owlv2_event_eval.AudioCandidate(time_sec=1.000, strength=1.0),
            owlv2_event_eval.AudioCandidate(time_sec=1.080, strength=3.0),
            owlv2_event_eval.AudioCandidate(time_sec=2.030, strength=9.0),
            owlv2_event_eval.AudioCandidate(time_sec=9.184, strength=9.0),
        ]

        filtered = owlv2_event_eval.rally_audio_candidates(rally, candidates)

        self.assertEqual([round(item.time_sec, 3) for item in filtered], [1.08])
        self.assertEqual(filtered[0].strength, 3.0)

    def test_fused_audio_touches_requires_nearby_trajectory_breakpoint(self) -> None:
        candidates = [
            owlv2_event_eval.AudioCandidate(time_sec=1.000, strength=1.0),
            owlv2_event_eval.AudioCandidate(time_sec=2.000, strength=1.0),
            owlv2_event_eval.AudioCandidate(time_sec=3.000, strength=1.0),
        ]
        breakpoints = [
            {"time_sec": 0.93, "lambda": 1200, "dvy": 100.0},
            {"time_sec": 0.93, "lambda": 3000, "dvy": 100.0},
            {"time_sec": 2.12, "lambda": 1200, "dvy": 100.0},
        ]

        fused, debug = owlv2_event_eval.fused_audio_touches(candidates, breakpoints, np.linspace(0.0, 4.0, 121))

        self.assertEqual([round(time_sec, 3) for time_sec in fused], [1.0])
        self.assertEqual([row["accepted"] for row in debug], [True, False, False])

    def test_fused_audio_touches_suppresses_weaker_nearby_arc_apex_candidate(self) -> None:
        candidates = [
            owlv2_event_eval.AudioCandidate(time_sec=6.912, strength=3.0),
            owlv2_event_eval.AudioCandidate(time_sec=7.168, strength=11.0),
        ]
        breakpoints = [
            {"time_sec": 6.91, "lambda": 1200, "dvy": -500.0},
            {"time_sec": 6.91, "lambda": 3000, "dvy": -500.0},
            {"time_sec": 6.91, "lambda": 6000, "dvy": -500.0},
            {"time_sec": 6.91, "lambda": 12000, "dvy": -500.0},
            {"time_sec": 6.91, "lambda": 30000, "dvy": -500.0},
            {"time_sec": 6.91, "lambda": 60000, "dvy": -500.0},
            {"time_sec": 7.15, "lambda": 1200, "dvy": 1200.0},
            {"time_sec": 7.15, "lambda": 3000, "dvy": 1200.0},
        ]

        fused, debug = owlv2_event_eval.fused_audio_touches(candidates, breakpoints, np.linspace(6.0, 8.0, 61))

        self.assertEqual([round(time_sec, 3) for time_sec in fused], [7.168])
        self.assertEqual([row["accepted"] for row in debug], [False, True])
        self.assertTrue(debug[0]["suppressed_by_nms"])

    def test_audio_cache_path_is_parameter_keyed(self) -> None:
        first = owlv2_event_eval.audio_cache_path_for(Path("/tmp/out"), Path("video.events.json"), 0.2, 0.45)
        second = owlv2_event_eval.audio_cache_path_for(Path("/tmp/out"), Path("video.events.json"), 0.1, 0.45)

        self.assertNotEqual(first, second)
        self.assertIn("delta0p2.wait0p45", first.name)


if __name__ == "__main__":
    unittest.main()
