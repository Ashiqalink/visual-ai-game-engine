"""
test_pipeline_payload.py — Payload contract & ToF sampling tests for VisionPipeline.

Pins down:
  * `_empty_gesture()` omitted keys that `_extract_gesture()` emits, so any
    consumer indexing e.g. payload["is_3_finger_pinching"] raised KeyError the
    moment the hand left the frame
  * payload["jitter"] was {} on no-hand frames while the HUD indexed into it
  * `pinch_pos` reported the raw centroid, discarding the smoothing computed
    for it every frame
  * ToF depth was sampled at RGB-frame coordinates against a depth map of a
    different resolution, and from a single (often zero) pixel
"""

import math
import queue
import unittest

import numpy as np

from visual_ai.pipeline import VisionPipeline


class _LM:
    """Stand-in for a MediaPipe NormalizedLandmark."""

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _hand(phase=0.0, jitter=0.0, push=0.0):
    """21 synthetic landmarks with thumb/index/middle held clearly apart."""
    pts = [(0.50, 0.60)] * 21
    sway = 0.10 * math.sin(phase)
    pts[8] = (0.50 + sway, 0.40)      # index tip
    pts[4] = (0.38 + sway, 0.50)      # thumb tip
    pts[12] = (0.62 + sway, 0.42)     # middle tip
    pts[9] = (0.50 + sway, 0.62)      # middle MCP (palm anchor)
    pts[5] = (0.45 + sway, 0.62)
    pts[16] = (0.66 + sway, 0.60)
    pts[20] = (0.70 + sway, 0.60)
    return [_LM(x + jitter, y + jitter, -0.02 - push) for (x, y) in pts]


def _pipeline(**kwargs):
    kwargs.setdefault("width", 800)
    kwargs.setdefault("height", 600)
    # Index 999 will not open; the pipeline thread is never started here anyway.
    kwargs.setdefault("camera_index", 999)
    return VisionPipeline(result_queue=queue.Queue(maxsize=2), **kwargs)


class TestPayloadContract(unittest.TestCase):

    def setUp(self):
        self.p = _pipeline()
        self.p.depth_simulated = True

    def test_no_hand_payload_has_every_emitted_key(self):
        gesture = self.p._extract_gesture(_hand())
        empty = self.p._empty_gesture()
        missing = set(gesture) - set(empty)
        self.assertEqual(missing, set(), f"no-hand payload missing {sorted(missing)}")

    def test_jitter_is_always_a_full_stat_dict(self):
        empty = self.p._empty_gesture()
        self.assertIsInstance(empty["jitter"], dict)
        for key in ("raw_jitter_std", "smoothed_jitter_std", "jitter_reduction_pct"):
            self.assertIn(key, empty["jitter"])

    def test_simulated_payload_shape_matches(self):
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        payload = self.p._empty_payload(1.0, 2.0, frame)
        for key in ("target_x", "target_y", "frame", "hand_visible", "jitter"):
            self.assertIn(key, payload)

    def test_pinch_pos_is_the_smoothed_centroid(self):
        gesture = None
        for i in range(30):
            gesture = self.p._extract_gesture(_hand(phase=i * 0.3))
        self.assertNotEqual(gesture["pinch_pos"], gesture["pinch_pos_raw"])

    def test_smoothing_reduces_reported_jitter(self):
        import random
        random.seed(5)
        gesture = None
        for i in range(60):
            gesture = self.p._extract_gesture(
                _hand(phase=i * 0.15, jitter=random.gauss(0, 0.004))
            )
        stats = gesture["jitter"]
        self.assertGreater(stats["raw_jitter_std"], 0.5)
        self.assertLess(stats["smoothed_jitter_std"], stats["raw_jitter_std"])
        self.assertGreater(stats["jitter_reduction_pct"], 25.0)

    def test_negative_landmark_does_not_reseed_the_filter(self):
        """
        MediaPipe emits negative coordinates when a fingertip leaves the frame.
        The old bootstrap test was `if smooth_ix < 0`, so such a sample re-seeded
        the filter mid-gesture: the output became the raw value exactly, with all
        velocity history thrown away.
        """
        for i in range(20):
            self.p._extract_gesture(_hand(phase=0.0))
        self.assertTrue(self.p._gs.smoothed)

        off_frame = _hand(phase=0.0)
        off_frame[8] = _LM(-0.05, 0.40, -0.02)     # -40 px, off the left edge
        self.p._extract_gesture(off_frame)

        # Filtered, not re-seeded: a re-seed lands exactly on the raw -40 px and
        # drops the filter's history.
        self.assertGreater(self.p._gs.smooth_ix, -40.0 + 1.0)
        self.assertTrue(self.p._gs.index_filter.initialized)

        # And the next in-frame sample must still be filtered relative to the
        # retained state rather than snapping.
        before = self.p._gs.smooth_ix
        self.p._extract_gesture(_hand(phase=0.0))
        self.assertNotEqual(self.p._gs.smooth_ix, before)


class TestToFSampling(unittest.TestCase):

    def setUp(self):
        self.p = _pipeline(width=320, height=240)
        self.p.depth_simulated = True

    def test_no_tof_source_reports_inactive(self):
        p = _pipeline()
        active, z, src = p.sample_depth(100, 100, 0.0)
        self.assertFalse(active)
        self.assertEqual(z, 0.0)

    def test_depth_map_resolution_is_rescaled(self):
        """80x60 sensor behind a 320x240 frame must still read 0.5 m."""
        self.p.depth_map = np.full((60, 80), 500, dtype=np.uint16)
        for px, py in ((0, 0), (160, 120), (319, 239)):
            with self.subTest(px=px, py=py):
                active, z, _ = self.p.sample_depth(px, py, 0.0)
                self.assertTrue(active)
                self.assertAlmostEqual(z, 0.5, delta=0.01)

    def test_out_of_range_pixels_are_clamped(self):
        self.p.depth_map = np.full((60, 80), 700, dtype=np.uint16)
        active, z, _ = self.p.sample_depth(9999, -50, 0.0)
        self.assertTrue(active)
        self.assertAlmostEqual(z, 0.7, delta=0.01)

    def test_dropout_pixels_survive_via_median_patch(self):
        """A single zero-return pixel must not poison the reading."""
        self.p.depth_map = np.full((60, 80), 400, dtype=np.uint16)
        self.p.depth_map[28:32, 38:42] = 0
        active, z, _ = self.p.sample_depth(160, 120, 0.0)
        self.assertAlmostEqual(z, 0.4, delta=0.01)

    def test_all_zero_patch_falls_back_to_estimate(self):
        self.p.depth_map = np.zeros((60, 80), dtype=np.uint16)
        active, z, _ = self.p.sample_depth(160, 120, 0.0)
        self.assertTrue(active)
        self.assertGreater(z, 0.15)

    def test_simulated_depth_tracks_landmark_z(self):
        near = self.p.sample_depth(160, 120, -0.15)[1]
        far = self.p.sample_depth(160, 120, 0.15)[1]
        self.assertLess(near, far)


class TestZPushClick(unittest.TestCase):

    def test_disabled_by_default(self):
        p = _pipeline()
        p.depth_simulated = True
        for i in range(40):
            g = p._extract_gesture(_hand(jitter=0.0, push=0.10 if i > 20 else 0.0))
            self.assertEqual(g["z_delta"], 0.0)

    def test_opt_in_detects_a_push(self):
        p = _pipeline(enable_z_click=True)
        p.depth_simulated = True
        fired = 0
        for i in range(30):
            push = 0.0 if i < 14 else 0.12 * (i - 13) / 6.0
            if p._extract_gesture(_hand(jitter=0.0005, push=push))["click_just_fired"]:
                fired += 1
        self.assertGreaterEqual(fired, 1)

    def test_lateral_swipe_does_not_fire(self):
        """
        The drift anchor used to be reset every below-threshold frame, so the
        lateral-movement guard let swipes through as clicks.
        """
        p = _pipeline(enable_z_click=True)
        p.depth_simulated = True
        fired = 0
        for i in range(40):
            # Sweep sideways hard while also pushing forward.
            push = 0.0 if i < 14 else 0.12 * (i - 13) / 6.0
            g = p._extract_gesture(_hand(phase=i * 0.5, push=push))
            if g["click_just_fired"] and not g["is_3_finger_pinching"]:
                fired += 1
        self.assertEqual(fired, 0)


class TestStabilizationWiring(unittest.TestCase):

    def test_tick_is_reachable_without_a_hand(self):
        p = _pipeline()
        p.depth_simulated = True
        p.begin_stabilization(duration=1.0)
        self.assertEqual(p.tof_stabilizer.state, "sampling")
        p.cancel_stabilization()
        self.assertEqual(p.tof_stabilizer.state, "inactive")

    def test_push_survives_calibration(self):
        import time
        p = _pipeline()
        p.depth_simulated = True
        p.begin_stabilization(duration=1.0)
        i = 0
        while p.tof_stabilizer.state == "sampling":
            p._extract_gesture(_hand(phase=i * 0.05, jitter=0.0002))
            p.tof_stabilizer.tick()
            i += 1
            time.sleep(0.004)
        self.assertEqual(p.tof_stabilizer.state, "active")

        rest = p._extract_gesture(_hand(push=0.0))["tof_z_m"]
        pushed = None
        for step in (0.04, 0.08, 0.12, 0.15):
            pushed = p._extract_gesture(_hand(push=step))["tof_z_m"]
        self.assertGreater(rest - pushed, 0.04)
        self.assertGreater(pushed, 0.06)

    def test_gate_is_exposed_in_the_payload(self):
        p = _pipeline()
        p.depth_simulated = True
        self.assertIn("stabilizer_gate", p._empty_gesture())
        self.assertIn("stabilizer_gate", p._extract_gesture(_hand()))


class TestFilterTuning(unittest.TestCase):

    def test_set_filter_tuning_reaches_live_filters(self):
        p = _pipeline()
        p.set_filter_tuning(min_cutoff=3.0, beta=0.02)
        self.assertEqual(p._gs.index_filter.min_cutoff, 3.0)
        self.assertEqual(p._gs.centroid_filter.beta, 0.02)

    def test_set_smooth_alpha_remaps_cutoff(self):
        p = _pipeline()
        p.set_smooth_alpha(0.5)
        self.assertGreater(p._gs.index_filter.min_cutoff, 1.0)

    def test_gesture_reset_clears_filter_state(self):
        p = _pipeline()
        p.depth_simulated = True
        for i in range(10):
            p._extract_gesture(_hand(phase=i * 0.2))
        self.assertTrue(p._gs.smoothed)
        p._gs.reset()
        self.assertFalse(p._gs.smoothed)
        self.assertFalse(p._gs.index_filter.initialized)


class TestCalibrationOverlayCache(unittest.TestCase):
    """
    The overlay runs in the capture thread on every frame of a 3-second
    calibration. Everything size-dependent is built once per resolution, so the
    cache has to survive a resolution change rather than keep drawing the
    previous frame's geometry.
    """

    def test_chrome_is_reused_for_the_same_size(self):
        p = _pipeline()
        first = p._calibration_chrome(480, 640)
        self.assertIs(p._calibration_chrome(480, 640), first)
        self.assertIs(first["tint"], p._calibration_chrome(480, 640)["tint"])

    def test_chrome_is_rebuilt_when_the_frame_size_changes(self):
        p = _pipeline()
        small = p._calibration_chrome(480, 640)
        large = p._calibration_chrome(720, 1280)
        self.assertIsNot(small, large)
        self.assertEqual(large["tint"].shape, (720, 1280, 3))
        # ...and switching back rebuilds again rather than serving 1280-wide
        # geometry into a 640-wide frame.
        again = p._calibration_chrome(480, 640)
        self.assertEqual(again["tint"].shape, (480, 640, 3))

    def test_overlay_covers_the_whole_frame_at_any_size(self):
        p = _pipeline()
        p.depth_simulated = True
        for h, w in ((480, 640), (720, 1280), (480, 640)):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            out = p._draw_stabilizer_warning(frame, 0.5)
            self.assertEqual(out.shape, (h, w, 3))
            # The navy wash is 0.78 of (15, 10, 30) over a black frame.
            self.assertEqual(tuple(int(v) for v in out[0, 0]), (12, 8, 23))


if __name__ == "__main__":
    unittest.main()
