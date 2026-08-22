"""
test_noise_filter.py — Unit tests for NoiseFilter, FilteredGestureDetector, and PipelineNoiseFilter.
"""

import random
import time
import unittest

from visual_ai.noise_filter import (
    FilteredGestureDetector,
    GenericStreamFilter,
    NoiseFilter,
    OneEuroFilter,
    PipelineNoiseFilter,
    ema_alpha_to_cutoff,
)


class MockDetector:
    def __init__(self):
        self.detected_count = 0

    def detect_gesture(self, frame):
        self.detected_count += 1
        return "GESTURE_OK"


class TestNoiseFilter(unittest.TestCase):

    def test_noise_filter_duration(self):
        """Test timer noise window active state and expiration."""
        nf = NoiseFilter(noise_duration=0.2)
        # Initially active once started
        self.assertTrue(nf.is_active())

        # Wait past duration
        time.sleep(0.25)
        self.assertFalse(nf.is_active())

    def test_noise_filter_reset(self):
        """Test timer reset reactivates the suppression window."""
        nf = NoiseFilter(noise_duration=0.2)
        self.assertTrue(nf.is_active())
        time.sleep(0.25)
        self.assertFalse(nf.is_active())

        # Reset timer
        nf.reset()
        self.assertTrue(nf.is_active())

    def test_filtered_gesture_detector(self):
        """Test detector wrapper suppresses detect_gesture calls during window."""
        mock = MockDetector()
        fgd = FilteredGestureDetector(detector_or_bus=mock, noise_duration=0.2)

        # During noise window, returns None and detector method is NOT called
        res = fgd.detect_gesture("dummy_frame")
        self.assertIsNone(res)
        self.assertEqual(mock.detected_count, 0)

        # Wait past window
        time.sleep(0.25)
        res = fgd.detect_gesture("dummy_frame")
        self.assertEqual(res, "GESTURE_OK")
        self.assertEqual(mock.detected_count, 1)

    def test_pipeline_noise_filter(self):
        """Test PipelineNoiseFilter zeroes out transient gesture flags during window."""
        pnf = PipelineNoiseFilter(noise_duration=0.2)

        raw_payload = {
            "hand_visible": True,
            "index_pos": (400, 300),
            "is_pinching": True,
            "click_just_fired": True,
            "is_3_pinching": True,
            "pinch_active": True,
            "z_delta": 0.05,
        }

        # Inside noise window: transient gesture flags should be suppressed (False / 0.0)
        filtered = pnf.process_payload(raw_payload)
        self.assertTrue(filtered["hand_visible"])
        self.assertFalse(filtered["is_pinching"])
        self.assertFalse(filtered["click_just_fired"])
        self.assertFalse(filtered["is_3_pinching"])
        self.assertFalse(filtered["pinch_active"])
        self.assertEqual(filtered["z_delta"], 0.0)

        # Wait past noise window
        time.sleep(0.25)
        unfiltered = pnf.process_payload(raw_payload)
        self.assertTrue(unfiltered["is_pinching"])
        self.assertTrue(unfiltered["click_just_fired"])
        self.assertTrue(unfiltered["is_3_pinching"])
        self.assertTrue(unfiltered["pinch_active"])
        self.assertEqual(unfiltered["z_delta"], 0.05)

    def test_suppresses_the_key_the_pipeline_actually_emits(self):
        """
        VisionPipeline emits `is_3_finger_pinching`. The filter previously named
        only `is_3_pinching`, so the 3-finger pinch was never suppressed.
        """
        pnf = PipelineNoiseFilter(noise_duration=2.0)
        payload = {
            "hand_visible": True,
            "is_pinching": True,
            "click_just_fired": True,
            "is_3_finger_pinching": True,
            "z_delta": 0.05,
            "xy_drift": 12.0,
        }
        filtered = pnf.process_payload(payload)
        self.assertFalse(filtered["is_3_finger_pinching"])
        self.assertFalse(filtered["is_pinching"])
        self.assertFalse(filtered["click_just_fired"])
        self.assertEqual(filtered["z_delta"], 0.0)
        self.assertEqual(filtered["xy_drift"], 0.0)

    def test_never_invents_keys(self):
        """The filter must not add keys the pipeline never defined."""
        pnf = PipelineNoiseFilter(noise_duration=2.0)
        payload = {"hand_visible": True, "index_pos": (1, 2)}
        self.assertEqual(set(pnf.process_payload(payload)), set(payload))

    def test_does_not_mutate_the_input(self):
        pnf = PipelineNoiseFilter(noise_duration=2.0)
        payload = {"is_pinching": True}
        pnf.process_payload(payload)
        self.assertTrue(payload["is_pinching"])

    def test_suppresses_per_hand_dicts_too(self):
        """
        The payload's "hands" tuple and the handedness shortcuts share dict
        objects; a top-level-only suppression let multi-hand consumers bypass
        the gate entirely.
        """
        pnf = PipelineNoiseFilter(noise_duration=2.0)
        left = {"handedness": "Left", "is_pinching": True,
                "click_just_fired": True, "z_delta": 0.05}
        right = {"handedness": "Right", "is_pinching": True,
                 "click_just_fired": False, "z_delta": 0.0}
        payload = {
            "hand_visible": True,
            "is_pinching": True,
            "hands": (left, right),
            "hand_left": left,
            "hand_right": right,
        }
        filtered = pnf.process_payload(payload)

        for hand in filtered["hands"]:
            self.assertFalse(hand["is_pinching"])
            self.assertFalse(hand["click_just_fired"])
            self.assertEqual(hand["z_delta"], 0.0)
        # Shortcuts still alias the entries of "hands".
        self.assertIs(filtered["hand_left"], filtered["hands"][0])
        self.assertIs(filtered["hand_right"], filtered["hands"][1])
        # The caller's dicts are untouched.
        self.assertTrue(left["is_pinching"])
        self.assertTrue(right["is_pinching"])

    def test_persistent_state_is_left_alone(self):
        """
        The filter can only rewrite its payload copy, not the pipeline's internal
        state, so masking persistent fields would make the HUD snap when the
        window closed. Only transient triggers are suppressed.
        """
        pnf = PipelineNoiseFilter(noise_duration=2.0)
        filtered = pnf.process_payload({"tof_z_m": 0.5, "hand_visible": True})
        self.assertEqual(filtered["tof_z_m"], 0.5)
        self.assertTrue(filtered["hand_visible"])

    def test_none_payload_passes_through(self):
        self.assertIsNone(PipelineNoiseFilter(noise_duration=1.0).process_payload(None))

    def test_closed_window_does_not_copy_the_payload(self):
        """
        The filter ships off, so the closed-window path runs on every frame of
        every session. Copying a ~55-key dict to change nothing in it was the
        whole per-frame cost; the payload is handed straight back instead.
        """
        payload = {"is_pinching": True, "z_delta": 0.05}

        # noise_duration <= 0 is the shipped default: permanently shut.
        self.assertIs(PipelineNoiseFilter(noise_duration=0.0).process_payload(payload),
                      payload)

        # A window that has expired behaves the same way.
        pnf = PipelineNoiseFilter(noise_duration=0.05)
        self.assertIsNot(pnf.process_payload(payload), payload)   # still open
        time.sleep(0.1)
        self.assertIs(pnf.process_payload(payload), payload)      # now shut

    def test_open_window_still_copies(self):
        """Suppression must not reach back into the pipeline's own dict."""
        pnf = PipelineNoiseFilter(noise_duration=2.0)
        payload = {"is_pinching": True, "z_delta": 0.05}
        filtered = pnf.process_payload(payload)
        self.assertIsNot(filtered, payload)
        self.assertTrue(payload["is_pinching"])
        self.assertFalse(filtered["is_pinching"])


class TestGenericStreamFilter(unittest.TestCase):

    def test_scalar_smoothing_converges(self):
        f = GenericStreamFilter(alpha=0.5)
        self.assertEqual(f.filter(0.0), 0.0)
        out = [f.filter(10.0) for _ in range(20)][-1]
        self.assertAlmostEqual(out, 10.0, delta=0.01)

    def test_tuple_smoothing(self):
        f = GenericStreamFilter(alpha=0.5)
        f.filter((0.0, 0.0))
        self.assertEqual(f.filter((10.0, 20.0)), (5.0, 10.0))

    def test_shape_switch_does_not_raise(self):
        """scalar -> tuple used to raise TypeError: float is not iterable."""
        f = GenericStreamFilter(alpha=0.3)
        f.filter(1.0)
        self.assertEqual(f.filter((1.0, 2.0)), (1.0, 2.0))

    def test_arity_change_restarts_instead_of_truncating(self):
        """A short zip() silently dropped the extra component."""
        f = GenericStreamFilter(alpha=0.3)
        f.filter((1.0, 2.0, 3.0))
        self.assertEqual(f.filter((5.0, 6.0)), (5.0, 6.0))

    def test_reset(self):
        f = GenericStreamFilter(alpha=0.5)
        f.filter(100.0)
        f.reset()
        self.assertEqual(f.filter(3.0), 3.0)


class TestOneEuroFilter(unittest.TestCase):

    FREQ = 30.0

    def _filter(self, beta=0.006, alpha=0.20):
        return OneEuroFilter(
            freq=self.FREQ,
            min_cutoff=ema_alpha_to_cutoff(alpha, self.FREQ),
            beta=beta,
        )

    def test_first_sample_passes_through(self):
        f = self._filter()
        self.assertEqual(f.filter((5.0, 6.0), 0.0), (5.0, 6.0))
        self.assertTrue(f.initialized)

    def test_steady_at_rest(self):
        """Resting jitter is heavily attenuated."""
        random.seed(3)
        f = self._filter()
        out = [f.filter((500.0 + random.gauss(0, 1.5), 300.0), i / self.FREQ)
               for i in range(90)]
        xs = [p[0] for p in out[30:]]
        self.assertLess(max(xs) - min(xs), 2.5)

    def test_keeps_up_with_fast_motion(self):
        """
        The old adaptive-EMA ramp topped out at alpha=0.20 (~5 frame lag) no
        matter how fast the hand moved. One-Euro must do far better.
        """
        f = self._filter()
        out = None
        for i in range(40):
            out = f.filter((100.0 + i * 40.0, 0.0), i / self.FREQ)
        lag = (100.0 + 39 * 40.0) - out[0]
        self.assertLess(lag, 80.0)

    def test_faster_motion_is_filtered_less(self):
        slow_lag, fast_lag = [], []
        for step, bucket in ((4.0, slow_lag), (60.0, fast_lag)):
            f = self._filter()
            out = None
            for i in range(40):
                out = f.filter((0.0 + i * step, 0.0), i / self.FREQ)
            bucket.append(((39 * step) - out[0]) / step)   # lag in frames
        self.assertLess(fast_lag[0], slow_lag[0])

    def test_scalar_input(self):
        f = self._filter()
        self.assertEqual(f.filter(1.0, 0.0), 1.0)
        out = f.filter(2.0, 1.0 / self.FREQ)
        self.assertIsInstance(out, float)
        self.assertTrue(1.0 <= out <= 2.0)

    def test_duplicate_timestamp_does_not_explode(self):
        f = self._filter()
        f.filter((0.0, 0.0), 1.0)
        out = f.filter((100.0, 0.0), 1.0)      # dt == 0
        self.assertTrue(all(abs(v) < 1e4 for v in out))

    def test_rewound_timestamp_does_not_explode(self):
        f = self._filter()
        f.filter((0.0, 0.0), 5.0)
        out = f.filter((100.0, 0.0), 4.0)      # dt < 0
        self.assertTrue(all(abs(v) < 1e4 for v in out))

    def test_shape_change_resets(self):
        f = self._filter()
        f.filter((1.0, 2.0), 0.0)
        self.assertEqual(f.filter((1.0, 2.0, 3.0), 0.1), (1.0, 2.0, 3.0))

    def test_reset_and_value(self):
        f = self._filter()
        self.assertIsNone(f.value)
        f.filter((7.0, 8.0), 0.0)
        self.assertEqual(f.value, (7.0, 8.0))
        f.reset()
        self.assertIsNone(f.value)
        self.assertFalse(f.initialized)

    def test_ema_alpha_to_cutoff_round_trip(self):
        """The mapped cutoff must reproduce the source EMA's alpha at rest."""
        for alpha in (0.05, 0.20, 0.50):
            with self.subTest(alpha=alpha):
                cutoff = ema_alpha_to_cutoff(alpha, self.FREQ)
                dt = 1.0 / self.FREQ
                self.assertAlmostEqual(OneEuroFilter._alpha(cutoff, dt), alpha, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
