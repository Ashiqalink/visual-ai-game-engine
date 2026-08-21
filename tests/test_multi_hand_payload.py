"""
test_multi_hand_payload.py — the additive payload surface: hand slots, fingertip
motion, the face box, and the opt-in depth grid.

Pins down, in particular:
  * MediaPipe's `multi_hand_landmarks` order is not stable, so hands are matched
    to slots by nearest wrist. Keying per-hand One-Euro state off the list index
    made two hands swap filter histories the frame the order flipped, and both
    outputs snapped at once.
  * The flat gesture keys must keep describing a single hand for the existing
    single-hand games, whatever `max_hands` is set to.
  * `index_velocity` is differenced from the smoothed fingertip; the point of
    also carrying `index_velocity_raw` is that the gap between them is the
    filter's phase lag, so the two must not be the same signal.
"""

import queue
import unittest

import numpy as np

from visual_ai.pipeline import VisionPipeline


class _LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _LandmarkSet:
    """Stands in for a MediaPipe NormalizedLandmarkList."""

    def __init__(self, landmarks):
        self.landmark = landmarks


class _Classification:
    def __init__(self, label, score):
        self.label, self.score = label, score


class _Handedness:
    def __init__(self, label, score=0.95):
        self.classification = [_Classification(label, score)]


class _HandResults:
    def __init__(self, sets, handedness=None):
        self.multi_hand_landmarks = sets or None
        self.multi_handedness = handedness


class _FakeHands:
    """Replays a scripted list of results, one per `process()` call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def process(self, _rgb):
        result = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return result


def _hand(cx=0.5, cy=0.6, spread=0.12, z=-0.02):
    """21 landmarks forming an open hand centred on (cx, cy) in normalised space."""
    pts = [(cx, cy)] * 21
    pts[0] = (cx, cy + spread)            # wrist
    pts[8] = (cx, cy - spread)            # index tip
    pts[6] = (cx, cy - spread * 0.5)
    pts[4] = (cx - spread, cy)            # thumb tip
    pts[3] = (cx - spread * 0.6, cy + spread * 0.2)
    pts[12] = (cx + spread * 0.3, cy - spread)
    pts[10] = (cx + spread * 0.3, cy - spread * 0.4)
    pts[16] = (cx + spread * 0.6, cy - spread * 0.6)
    pts[14] = (cx + spread * 0.6, cy - spread * 0.2)
    pts[20] = (cx + spread * 0.8, cy - spread * 0.4)
    pts[18] = (cx + spread * 0.8, cy)
    pts[17] = (cx + spread * 0.9, cy + spread * 0.6)
    pts[9] = (cx, cy + spread * 0.4)
    pts[5] = (cx - spread * 0.3, cy + spread * 0.4)
    return _LandmarkSet([_LM(x, y, z) for (x, y) in pts])


def _pipeline(**kwargs):
    kwargs.setdefault("width", 800)
    kwargs.setdefault("height", 600)
    kwargs.setdefault("camera_index", 999)
    return VisionPipeline(result_queue=queue.Queue(maxsize=2), **kwargs)


def _frame():
    return np.zeros((600, 800, 3), dtype=np.uint8)


class TestBackwardsCompatibility(unittest.TestCase):
    """The three shipped games read the flat keys. Those must not move."""

    def test_every_flat_key_survives_the_multi_hand_payload(self):
        p = _pipeline()
        p._mp_face = None
        p._mp_hands = _FakeHands([_HandResults([_hand()])])
        payload = p._process_frame(_frame())
        for key in ("hand_visible", "index_pos", "pinch_pos", "is_pinching",
                    "hand_sign", "is_fist", "is_open_palm", "grip_openness",
                    "tof_z_m", "jitter", "stabilizer_state", "target_x"):
            self.assertIn(key, payload)

    def test_no_hand_payload_still_carries_every_emitted_key(self):
        p = _pipeline()
        p.tof_simulated = True
        emitted = set(p._extract_gesture(_hand().landmark))
        empty = set(p._empty_gesture())
        self.assertEqual(emitted - empty, set())

    def test_empty_payload_carries_the_multi_hand_keys(self):
        p = _pipeline()
        payload = p._empty_payload(1.0, 2.0, _frame())
        for key in ("hands", "hand_count", "hand_left", "hand_right",
                    "face_visible", "face_box", "index_velocity"):
            self.assertIn(key, payload)
        self.assertEqual(payload["hand_count"], 0)

    def test_slot_zero_is_the_legacy_state_object(self):
        p = _pipeline()
        self.assertIs(p._gs, p._gs_slots[0])
        self.assertIs(p.jitter_analyzer, p._jitter_slots[0])

    def test_single_hand_always_lands_in_slot_zero(self):
        p = _pipeline(max_hands=2)
        p._mp_face = None
        p._mp_hands = _FakeHands([_HandResults([_hand(cx=0.3 + 0.05 * i)]) for i in range(8)])
        for _ in range(8):
            payload = p._process_frame(_frame())
            self.assertEqual(payload["hand_count"], 1)
            self.assertEqual(payload["slot"], 0)


class TestSlotAssignment(unittest.TestCase):

    def test_two_hands_get_distinct_slots(self):
        p = _pipeline(max_hands=2)
        p._mp_face = None
        p._mp_hands = _FakeHands([
            _HandResults([_hand(cx=0.25), _hand(cx=0.75)],
                         [_Handedness("Left"), _Handedness("Right")])
        ])
        payload = p._process_frame(_frame())
        self.assertEqual(payload["hand_count"], 2)
        self.assertEqual({h["slot"] for h in payload["hands"]}, {0, 1})

    def test_list_order_flip_does_not_move_a_hand_between_slots(self):
        """
        The regression this exists for: MediaPipe hands the same two hands back
        in the opposite order on some frame. Slot assignment must follow the
        wrist positions, not the list index.
        """
        p = _pipeline(max_hands=2)
        p._mp_face = None
        left, right = _hand(cx=0.25), _hand(cx=0.75)
        p._mp_hands = _FakeHands([
            _HandResults([left, right]),
            _HandResults([left, right]),
            _HandResults([right, left]),      # <- order flips here
            _HandResults([right, left]),
        ])

        slot_of_left = None
        for frame_no in range(4):
            payload = p._process_frame(_frame())
            # The hand near x=0.25*800=200 px, whichever slot holds it.
            near_left = min(payload["hands"], key=lambda h: h["pinch_pos"][0])
            if frame_no == 0:
                slot_of_left = near_left["slot"]
            else:
                self.assertEqual(near_left["slot"], slot_of_left,
                                 f"left hand changed slot on frame {frame_no}")

    def test_crossing_hands_keep_their_filter_state(self):
        """
        Two hands sweeping past each other. If slot state followed list order,
        the smoothed position of a slot would jump the width of the gap between
        the hands on the swap frame.
        """
        p = _pipeline(max_hands=2)
        p._mp_face = None
        script = []
        for i in range(20):
            a = 0.30 + 0.02 * i          # sweeps right
            b = 0.70 - 0.02 * i          # sweeps left
            # Deliberately hand them back in an order that flips halfway.
            pair = [_hand(cx=a), _hand(cx=b)] if i < 10 else [_hand(cx=b), _hand(cx=a)]
            script.append(_HandResults(pair))
        p._mp_hands = _FakeHands(script)

        last = {}
        max_jump = 0.0
        for _ in range(20):
            payload = p._process_frame(_frame())
            for h in payload["hands"]:
                if h["slot"] in last:
                    max_jump = max(max_jump, abs(h["pinch_pos"][0] - last[h["slot"]]))
                last[h["slot"]] = h["pinch_pos"][0]

        # Each hand moves ~16 px/frame here; a swapped slot would jump the
        # several-hundred-px gap between the two hands instead.
        self.assertLess(max_jump, 120.0, f"a slot jumped {max_jump:.0f} px — states swapped")

    def test_handedness_shortcuts_are_populated(self):
        p = _pipeline(max_hands=2)
        p._mp_face = None
        p._mp_hands = _FakeHands([
            _HandResults([_hand(cx=0.25), _hand(cx=0.75)],
                         [_Handedness("Left"), _Handedness("Right")])
        ])
        payload = p._process_frame(_frame())
        self.assertIsNotNone(payload["hand_left"])
        self.assertIsNotNone(payload["hand_right"])
        self.assertEqual(payload["hand_left"]["handedness"], "Left")
        self.assertGreater(payload["hand_right"]["handedness_score"], 0.5)

    def test_max_hands_one_drops_the_extra_hand(self):
        p = _pipeline(max_hands=1)
        p._mp_face = None
        p._mp_hands = _FakeHands([_HandResults([_hand(cx=0.25), _hand(cx=0.75)])])
        payload = p._process_frame(_frame())
        self.assertEqual(payload["hand_count"], 1)

    def test_slot_is_released_after_the_hand_is_gone(self):
        p = _pipeline(max_hands=2)
        p._mp_face = None
        script = [_HandResults([_hand(cx=0.25), _hand(cx=0.75)])]
        script += [_HandResults([])] * 15
        p._mp_hands = _FakeHands(script)
        for _ in range(16):
            p._process_frame(_frame())
        self.assertEqual(p._slot_anchor, [None, None])


class TestFilterTuningReachesEverySlot(unittest.TestCase):
    """
    Retuning used to walk `self._gs` only, so a second hand kept the settings it
    was constructed with: `set_smooth_alpha` visibly steadied one hand and left
    the other jittering, with nothing in the payload to say why.
    """

    def _filters(self, p):
        return [f for gs in p._gs_slots
                for f in (gs.index_filter, gs.centroid_filter)]

    def test_set_filter_tuning_reaches_slots_past_zero(self):
        p = _pipeline(max_hands=3)
        p.set_filter_tuning(min_cutoff=4.5, beta=0.02)
        for f in self._filters(p):
            self.assertAlmostEqual(f.min_cutoff, 4.5)
            self.assertAlmostEqual(f.beta, 0.02)

    def test_set_smooth_alpha_reaches_slots_past_zero(self):
        p = _pipeline(max_hands=3)
        p.set_smooth_alpha(0.15)
        expected = p.filter_min_cutoff
        for f in self._filters(p):
            self.assertAlmostEqual(f.min_cutoff, expected)

    def test_tuning_does_not_reset_filter_state(self):
        """The point of tuning in place is that live histories survive it."""
        p = _pipeline(max_hands=2)
        p._mp_face = None
        p._mp_hands = _FakeHands([_HandResults([_hand(cx=0.4)]) for _ in range(4)])
        for _ in range(4):
            p._process_frame(_frame())
        before = p._gs.smooth_ix
        p.set_filter_tuning(min_cutoff=2.0)
        self.assertEqual(p._gs.smooth_ix, before)
        self.assertTrue(p._gs.index_filter.initialized)


_DT = 1.0 / 30.0     # step the clock at capture rate, not at loop speed


class TestFingertipMotion(unittest.TestCase):

    def test_velocity_points_the_way_the_hand_moves(self):
        p = _pipeline()
        p.tof_simulated = True
        gesture = None
        for i in range(30):
            gesture = p._extract_gesture(_hand(cx=0.3 + 0.01 * i).landmark, now=1000.0 + i * _DT)
        vx, vy = gesture["index_velocity"]
        self.assertGreater(vx, 0.0, "rightward motion should give positive vx")
        self.assertGreater(gesture["index_speed"], 0.0)

    def test_a_still_hand_reports_near_zero_speed(self):
        p = _pipeline()
        p.tof_simulated = True
        gesture = None
        for i in range(30):
            gesture = p._extract_gesture(_hand(cx=0.5).landmark, now=1000.0 + i * _DT)
        self.assertLess(gesture["index_speed"], 1.0)

    def test_velocity_stays_zero_until_a_second_sample_exists(self):
        p = _pipeline()
        p.tof_simulated = True
        first = p._extract_gesture(_hand().landmark, now=1000.0)
        self.assertEqual(first["index_velocity"], (0.0, 0.0))

    @staticmethod
    def _reversal_run(beta):
        """
        Triangle wave at 360 px/s: 20 frames right, then hard left. Returns
        (velocity at the first reversed frame, frames until the smoothed
        velocity recovers to 80% of the true magnitude).
        """
        p = _pipeline(filter_beta=beta)
        p.tof_simulated = True
        at_reversal, recovered = None, None
        for i in range(40):
            cx = 0.3 + 0.015 * i if i < 20 else 0.6 - 0.015 * (i - 20)
            g = p._extract_gesture(_hand(cx=cx).landmark, now=1000.0 + i * _DT)
            if i == 21:
                at_reversal = g["index_velocity"][0]
            if (i > 20 and recovered is None
                    and abs(g["index_velocity"][0]) >= 0.8 * abs(g["index_velocity_raw"][0])):
                recovered = i - 20
        return at_reversal, recovered

    def test_smoothed_velocity_is_attenuated_not_inverted_at_the_default_beta(self):
        """
        Measured, not assumed: at the shipped beta the filtered velocity turns
        around on the *same* frame as the hand — the lag shows up as amplitude,
        not as a wrong direction. A game reading `index_velocity` for direction
        can trust it; one reading it for magnitude cannot, not immediately.
        """
        at_reversal, recovered = self._reversal_run(0.006)
        self.assertLess(at_reversal, 0.0, "direction was wrong at the reversal")
        self.assertLess(abs(at_reversal), 0.2 * 360.0, "expected heavy attenuation")
        self.assertIsNotNone(recovered)

    def test_beta_zero_actually_inverts_the_reported_direction(self):
        """Why `filter_beta` is not optional: with no speed coupling the
        filtered fingertip is still travelling the old way after the reversal."""
        at_reversal, _ = self._reversal_run(0.0)
        self.assertGreater(at_reversal, 0.0)

    def test_raising_beta_monotonically_shortens_reversal_recovery(self):
        recoveries = [self._reversal_run(b)[1] for b in (0.0, 0.006, 0.02, 0.05, 0.2)]
        self.assertTrue(all(a >= b for a, b in zip(recoveries, recoveries[1:])),
                        f"recovery not monotonic in beta: {recoveries}")
        self.assertLess(recoveries[-1], recoveries[0])

    def test_motion_state_is_per_slot(self):
        p = _pipeline(max_hands=2)
        p._mp_face = None
        script = [
            _HandResults([_hand(cx=0.25), _hand(cx=0.70 + 0.01 * i)])
            for i in range(15)
        ]
        p._mp_hands = _FakeHands(script)
        payload = None
        for _ in range(15):
            payload = p._process_frame(_frame())
        speeds = {h["slot"]: h["index_speed"] for h in payload["hands"]}
        still = min(speeds.values())
        moving = max(speeds.values())
        self.assertLess(still, 1.0)
        self.assertGreater(moving, still)


class TestFaceBox(unittest.TestCase):

    def test_no_face_reports_frame_centre_and_an_empty_box(self):
        p = _pipeline()
        p._mp_face = None
        p._mp_hands = None
        payload = p._process_frame(_frame())
        self.assertFalse(payload["face_visible"])
        self.assertEqual(payload["face_box"], (0, 0, 0, 0))
        self.assertAlmostEqual(payload["target_x"], 400.0)


class TestDepthGrid(unittest.TestCase):

    def test_off_by_default(self):
        p = _pipeline()
        p._mp_face = None
        p._mp_hands = None
        self.assertNotIn("depth_grid", p._process_frame(_frame()))

    def test_real_depth_map_is_resized_and_converted_to_metres(self):
        p = _pipeline()
        p._mp_face = None
        p._mp_hands = None
        p.emit_depth_grid = True
        p.depth_map = np.full((60, 80), 750, dtype=np.uint16)
        grid = p._process_frame(_frame())["depth_grid"]
        cols, rows = p.depth_grid_size
        self.assertEqual(grid.shape, (rows, cols))
        self.assertAlmostEqual(float(grid.mean()), 0.75, places=3)

    def test_simulated_grid_puts_the_hand_nearer_than_the_background(self):
        p = _pipeline(max_hands=1)
        p._mp_face = None
        p.emit_depth_grid = True
        p.tof_simulated = True
        p._mp_hands = _FakeHands([_HandResults([_hand(cx=0.5, cy=0.5)])])
        grid = p._process_frame(_frame())["depth_grid"]
        self.assertLess(float(grid.min()), float(grid.max()))
        self.assertLess(float(grid.min()), 1.0)

    def test_grid_size_is_configurable(self):
        p = _pipeline()
        p._mp_face = None
        p._mp_hands = None
        p.emit_depth_grid = True
        p.depth_grid_size = (64, 48)
        p.depth_map = np.full((60, 80), 500, dtype=np.uint16)
        self.assertEqual(p._process_frame(_frame())["depth_grid"].shape, (48, 64))


if __name__ == "__main__":
    unittest.main()
