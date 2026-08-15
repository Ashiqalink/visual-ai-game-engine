"""
test_tof_stabilizer.py — Regression tests for ToFStabilizer.

Each test here pins down a bug that made stabilization unusable:
  * calibration subtracted the absolute mean depth, collapsing every reading
    to the 0.05 m clamp and destroying the push/punch signal
  * the sampling window only advanced when a hand was in view, so calibration
    hung in "sampling" forever with the warning overlay stuck on screen
  * an all-zero depth stream (ToF disabled) activated against a meaningless
    baseline instead of reporting failure
"""

import math
import time
import unittest

from visual_ai.tof_stabilizer import ToFStabilizer


def _calibrate(stab, depth=0.45, shake=0.003, duration=1.0):
    """Run a full calibration window feeding `depth` +/- `shake` metres."""
    stab.begin(duration)
    i = 0
    while stab.state == ToFStabilizer.STATE_SAMPLING:
        stab.feed(depth + math.sin(i * 1.7) * shake)
        i += 1
        time.sleep(0.004)
    return stab


class TestToFStabilizer(unittest.TestCase):

    def test_calibration_measures_baseline_and_noise(self):
        s = _calibrate(ToFStabilizer(), depth=0.45, shake=0.004)
        self.assertEqual(s.state, ToFStabilizer.STATE_ACTIVE)
        self.assertAlmostEqual(s.z_baseline, 0.45, delta=0.005)
        self.assertGreater(s.z_noise_amplitude, 0.0005)
        self.assertGreater(s.noise_gate, 0.0)
        self.assertEqual(s.progress, 1.0)

    def test_absolute_depth_is_preserved(self):
        """Correction must not subtract the baseline outright."""
        s = _calibrate(ToFStabilizer(), depth=0.45, shake=0.003)
        for target in (0.40, 0.30, 0.60):
            with self.subTest(target=target):
                got = s.correct(target)
                self.assertAlmostEqual(got, target, delta=0.02)

    def test_readings_do_not_collapse_to_the_clamp(self):
        s = _calibrate(ToFStabilizer(), depth=0.45, shake=0.003)
        got = [s.correct(z) for z in (0.45, 0.40, 0.35, 0.30, 0.25)]
        self.assertTrue(all(g > ToFStabilizer.MIN_DEPTH_M + 0.05 for g in got), got)
        # Monotonically approaching the camera must stay monotonic.
        self.assertTrue(all(b < a for a, b in zip(got, got[1:])), got)

    def test_vibration_is_suppressed_at_rest(self):
        s = _calibrate(ToFStabilizer(), depth=0.45, shake=0.004)
        out = [s.correct(0.45 + math.sin(i * 2.3) * 0.004) for i in range(60)]
        self.assertLess(max(out) - min(out), 0.004)

    def test_real_movement_still_passes_through(self):
        s = _calibrate(ToFStabilizer(), depth=0.45, shake=0.002)
        rest = s.correct(0.45)
        pushed = None
        for z in (0.43, 0.40, 0.36, 0.32, 0.30):
            pushed = s.correct(z)
        self.assertGreater(rest - pushed, 0.10)

    def test_tick_finalises_without_any_samples(self):
        """A calibration started with no hand in view must not hang."""
        s = ToFStabilizer()
        s.begin(1.0)
        t0 = time.time()
        while time.time() - t0 < 1.3:
            s.tick()
            time.sleep(0.01)
        self.assertNotEqual(s.state, ToFStabilizer.STATE_SAMPLING)
        self.assertEqual(s.state, ToFStabilizer.STATE_INACTIVE)
        self.assertIsNotNone(s.last_error)

    def test_zero_depth_stream_is_rejected(self):
        """ToF disabled reports 0.0 m; that must not become a baseline."""
        s = ToFStabilizer()
        s.begin(1.0)
        while s.state == ToFStabilizer.STATE_SAMPLING:
            s.feed(0.0)
            time.sleep(0.004)
        self.assertEqual(s.state, ToFStabilizer.STATE_INACTIVE)
        self.assertEqual(s.sample_count, 0)

    def test_non_finite_samples_are_discarded(self):
        s = ToFStabilizer()
        s.begin(1.0)
        s.feed(float("nan"))
        s.feed(float("inf"))
        s.feed(-1.0)
        self.assertEqual(s.sample_count, 0)
        s.cancel()

    def test_correct_is_a_noop_before_calibration(self):
        s = ToFStabilizer()
        self.assertEqual(s.correct(0.42), 0.42)
        self.assertEqual(s.noise_gate, 0.0)
        s.begin(1.0)
        self.assertEqual(s.correct(0.42), 0.42)   # sampling: still a no-op
        s.cancel()

    def test_invalid_reading_passes_through_uncorrected(self):
        s = _calibrate(ToFStabilizer(), depth=0.45)
        self.assertEqual(s.correct(0.0), 0.0)
        self.assertTrue(math.isnan(s.correct(float("nan"))))

    def test_cancel_and_disable(self):
        s = ToFStabilizer()
        s.begin(2.0)
        s.cancel()
        self.assertEqual(s.state, ToFStabilizer.STATE_INACTIVE)
        self.assertEqual(s.time_remaining, 0.0)

        s = _calibrate(ToFStabilizer(), depth=0.45)
        s.disable()
        self.assertEqual(s.state, ToFStabilizer.STATE_INACTIVE)
        self.assertEqual(s.z_baseline, 0.0)
        self.assertEqual(s.z_noise_amplitude, 0.0)

    def test_short_duration_is_clamped(self):
        s = ToFStabilizer()
        s.begin(0.1)
        self.assertGreaterEqual(s.time_remaining, 0.5)
        s.cancel()

    def test_baseline_tracks_slow_drift(self):
        """Sustained in-gate offset should be absorbed, not reported forever."""
        s = _calibrate(ToFStabilizer(), depth=0.45, shake=0.004)
        gate = s.noise_gate
        drifted = 0.45 + gate * 0.8
        for _ in range(200):
            out = s.correct(drifted)
        self.assertAlmostEqual(out, drifted, delta=gate * 0.3)


if __name__ == "__main__":
    unittest.main()
