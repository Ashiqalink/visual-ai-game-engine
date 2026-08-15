"""
test_jitter_analyzer.py — Regression tests for JitterAnalyzer.

Pins down the bugs that made the jitter readout misleading:
  * "jitter" was the std-dev of per-frame step *distance*, so fast smooth
    motion read as heavy jitter while shaking in place read as none
  * window_size <= 1 either raised ValueError or built a zero-length delta
    buffer that silently discarded every sample
  * get_stats() returned a partial dict, so consumers indexing a key on a
    no-measurement frame got KeyError
"""

import random
import unittest

from visual_ai.jitter_analyzer import JitterAnalyzer


class TestJitterAnalyzer(unittest.TestCase):

    def test_smooth_fast_motion_reports_no_jitter(self):
        """A clean 6 px/frame sweep is fast, not shaky."""
        j = JitterAnalyzer(30)
        stats = {}
        for i in range(40):
            p = (100.0 + i * 6.0, 200.0)
            stats = j.update(p, p)
        self.assertLess(stats["raw_jitter_std"], 0.1)
        self.assertAlmostEqual(stats["raw_speed_px"], 6.0, delta=0.1)

    def test_accelerating_motion_reports_little_jitter(self):
        """Steady acceleration is smooth: constant second difference, no noise."""
        j = JitterAnalyzer(30)
        stats = {}
        for i in range(40):
            p = (100.0 + 0.5 * i * i, 200.0)
            stats = j.update(p, p)
        self.assertLess(stats["raw_jitter_std"], 1.0)
        self.assertGreater(stats["raw_speed_px"], 10.0)

    def test_shaking_in_place_is_detected(self):
        j = JitterAnalyzer(30)
        stats = {}
        for i in range(40):
            stats = j.update((300.0 + (2.5 if i % 2 else -2.5), 200.0), (300.0, 200.0))
        self.assertGreater(stats["raw_jitter_std"], 1.5)
        self.assertGreater(stats["jitter_reduction_pct"], 95.0)

    def test_noise_amplitude_is_recovered(self):
        """Gaussian noise on a moving path: metric reads sigma * sqrt(dims)."""
        random.seed(7)
        j = JitterAnalyzer(30)
        stats = {}
        for i in range(60):
            trend = (100.0 + i * 6.0, 200.0)
            noisy = (trend[0] + random.gauss(0, 2.0), trend[1] + random.gauss(0, 2.0))
            stats = j.update(noisy, trend)
        self.assertAlmostEqual(stats["raw_jitter_std"], 2.0 * (2 ** 0.5), delta=0.9)

    def test_reduction_is_bounded_and_signed(self):
        """A smoother that adds noise reports bounded negative, not nonsense."""
        random.seed(11)
        j = JitterAnalyzer(30)
        stats = {}
        for i in range(40):
            trend = (100.0 + i * 4.0, 200.0)
            raw = (trend[0] + random.gauss(0, 1.0), trend[1] + random.gauss(0, 1.0))
            worse = (trend[0] + random.gauss(0, 8.0), trend[1] + random.gauss(0, 8.0))
            stats = j.update(raw, worse)
        self.assertLess(stats["jitter_reduction_pct"], 0.0)
        self.assertGreaterEqual(stats["jitter_reduction_pct"], -100.0)

    def test_no_jitter_on_either_stream_reports_zero_reduction(self):
        j = JitterAnalyzer(30)
        stats = {}
        for i in range(20):
            p = (10.0 * i, 0.0)
            stats = j.update(p, p)
        self.assertEqual(stats["jitter_reduction_pct"], 0.0)

    def test_step_statistics_still_reported(self):
        j = JitterAnalyzer(30)
        stats = {}
        for i in range(20):
            p = (100.0 + i * 5.0, 200.0)
            stats = j.update(p, p)
        self.assertAlmostEqual(stats["raw_jitter_mean_px"], 5.0, delta=0.01)
        self.assertAlmostEqual(stats["raw_max_jump_px"], 5.0, delta=0.01)

    def test_degenerate_window_sizes(self):
        for size in (0, 1, 2, -5):
            with self.subTest(size=size):
                j = JitterAnalyzer(size)
                self.assertGreaterEqual(j.window_size, JitterAnalyzer.MIN_WINDOW)
                self.assertGreaterEqual(j.deltas_raw.maxlen, 2)
                for i in range(10):
                    j.update((float(i), 0.0))

    def test_stats_shape_is_always_complete(self):
        j = JitterAnalyzer(10)
        full_keys = set(JitterAnalyzer.empty_stats())
        self.assertEqual(set(j.get_stats()), full_keys)          # nothing fed yet
        self.assertEqual(set(j.update((1.0, 1.0))), full_keys)    # one sample
        for i in range(10):
            stats = j.update((float(i), 0.0))
        self.assertEqual(set(stats), full_keys)                   # fully warmed up

    def test_smoothed_defaults_to_raw(self):
        j = JitterAnalyzer(10)
        stats = {}
        for i in range(10):
            stats = j.update((100.0 + i * 3.0, 5.0))
        self.assertEqual(stats["raw_jitter_std"], stats["smoothed_jitter_std"])
        self.assertEqual(stats["jitter_reduction_pct"], 0.0)

    def test_three_dimensional_input(self):
        j = JitterAnalyzer(10)
        stats = {}
        for i in range(15):
            p = (float(i), 2.0 * i, 0.5 * i)
            stats = j.update(p, p)
        self.assertLess(stats["raw_jitter_std"], 0.1)
        self.assertGreater(stats["raw_speed_px"], 0.0)

    def test_mismatched_arity_does_not_raise(self):
        j = JitterAnalyzer(10)
        for i in range(6):
            j.update((float(i), 0.0, 0.0), (float(i), 0.0))
        self.assertGreater(j.get_stats()["sample_count"], 0)

    def test_reset_clears_history(self):
        j = JitterAnalyzer(10)
        for i in range(10):
            j.update((float(i * 7), 0.0))
        j.reset()
        self.assertEqual(j.total_frames, 0)
        self.assertEqual(len(j.raw_positions), 0)
        self.assertEqual(j.get_stats()["sample_count"], 0)
        self.assertEqual(j.get_stats()["raw_jitter_std"], 0.0)

    def test_rolling_window_is_bounded(self):
        j = JitterAnalyzer(15)
        for i in range(200):
            j.update((float(i), 0.0))
        self.assertEqual(len(j.raw_positions), 15)
        self.assertEqual(j.total_frames, 200)


if __name__ == "__main__":
    unittest.main()
