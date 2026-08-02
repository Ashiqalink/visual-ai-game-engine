"""
test_noise_filter.py — Unit tests for NoiseFilter, FilteredGestureDetector, and PipelineNoiseFilter.
"""

import time
import unittest
from visual_ai.noise_filter import NoiseFilter, FilteredGestureDetector, PipelineNoiseFilter


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


if __name__ == "__main__":
    unittest.main()
