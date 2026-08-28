"""
test_finger_extension_hysteresis.py — the boundary a finger is judged against
has to have a width.

`fingers_extended` is a tip-reaches-past-its-own-PIP test. As a bare `>` that
decision has no width, so a finger resting near the boundary flips on landmark
noise alone. `point` is the sign that pays for it: it needs three fingers to
stay down *at the same time*, and one finger flickering is enough that no sign
ever survives `_SIGN_DEBOUNCE` frames — the hand simply reads `unknown`. That is
the "pointing is very hard to get right" report, and the accelerated hand
backend made it worse, being a few px noisier than MediaPipe's CPU graph.

So the tests here are about what happens *inside* the band, which is where a
bare threshold and a hysteretic one differ and nowhere else:

  * noise inside the band must not move the answer, however many frames it runs
  * a real move through the band must still move it
  * the band is a fraction of the palm span, so it means the same at any hand
    size or camera distance
"""

import math
import queue
import unittest

from visual_ai.pipeline import (
    _FINGER_EXT_MARGIN,
    _THUMB_EXT_MARGIN,
    SIGN_POINT,
    VisionPipeline,
    _extended,
)

WIDTH, HEIGHT = 800, 600
WRIST = (0.5, 0.9)
#: Wrist-to-pinky-MCP distance every offset below is expressed in.
PALM = 0.2
#: PIP joints all sit this far from the wrist; a tip's offset is measured
#: from here.
PIP_REACH = 1.0 * PALM


class _LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _at(reach, sideways):
    """A point exactly `reach` from the wrist, offset `sideways` across it."""
    sideways = max(-reach, min(reach, sideways))
    return (WRIST[0] + sideways,
            WRIST[1] - math.sqrt(max(0.0, reach * reach - sideways * sideways)))


def _hand(index=0.5, middle=0.5, ring=0.5, pinky=0.5, thumb=0.5):
    """
    21 landmarks with each finger's tip placed at a chosen reach past its PIP.

    Offsets are in palm spans and signed the way `finger_extension` reports
    them: positive is extended. `thumb` is its abduction, measured against the
    0.10 threshold rather than against zero.
    """
    pts = [WRIST] * 21
    pts[0] = WRIST
    pts[17] = _at(PALM, PALM * 0.9)                   # pinky MCP sets the span
    for (tip, pip), offset, across in (
            ((8, 6), index, -0.45), ((12, 10), middle, -0.15),
            ((16, 14), ring, 0.15), ((20, 18), pinky, 0.45)):
        pts[pip] = _at(PIP_REACH, PIP_REACH * across)
        pts[tip] = _at(PIP_REACH + offset * PALM, PIP_REACH * across)

    # The thumb is measured as abduction from the pinky MCP: tip further from
    # it than the IP joint means an extended thumb. Place both on the line
    # through the pinky MCP so the difference is exactly what is asked for.
    pinky_mcp = pts[17]
    pts[3] = (pinky_mcp[0] - PALM * 0.8, pinky_mcp[1])
    pts[4] = (pinky_mcp[0] - PALM * (0.8 + thumb), pinky_mcp[1])
    return [_LM(x, y, 0.0) for (x, y) in pts]


def _pipeline(**kwargs):
    kwargs.setdefault("width", WIDTH)
    kwargs.setdefault("height", HEIGHT)
    kwargs.setdefault("camera_index", 999)
    return VisionPipeline(result_queue=queue.Queue(maxsize=2), **kwargs)


class TestTheBandItself(unittest.TestCase):

    def test_outside_the_band_the_previous_answer_is_ignored(self):
        self.assertTrue(_extended(0.5, 0.0, 0.08, previous=False))
        self.assertFalse(_extended(-0.5, 0.0, 0.08, previous=True))

    def test_inside_the_band_the_previous_answer_is_kept(self):
        self.assertTrue(_extended(0.01, 0.0, 0.08, previous=True))
        self.assertFalse(_extended(0.01, 0.0, 0.08, previous=False))
        # ...on the far side of the raw threshold too, which is the point: a
        # sign that has settled does not come apart because a finger drifted a
        # hundredth of a palm across the line.
        self.assertTrue(_extended(-0.01, 0.0, 0.08, previous=True))

    def test_the_threshold_moves_the_band_with_it(self):
        self.assertFalse(_extended(0.11, 0.10, 0.05, previous=False))
        self.assertTrue(_extended(0.16, 0.10, 0.05, previous=False))


class TestFingerReadout(unittest.TestCase):

    def test_the_fixture_places_fingers_where_it_says(self):
        """A check that could not fail is not a check — verify the fixture."""
        p = _pipeline()
        gesture = p._extract_gesture(_hand(index=0.5, middle=-0.5, ring=-0.5,
                                           pinky=-0.5, thumb=0.5), now=1.0)
        thumb, index, middle, ring, pinky = gesture["finger_extension"]
        self.assertAlmostEqual(index, 0.5, places=2)
        self.assertAlmostEqual(middle, -0.5, places=2)
        self.assertAlmostEqual(thumb, 0.5, places=2)
        p.stop()

    def test_the_reported_deltas_agree_with_the_booleans(self):
        p = _pipeline()
        gesture = p._extract_gesture(_hand(index=0.5, middle=-0.5, ring=-0.5,
                                           pinky=-0.5, thumb=-0.5), now=1.0)
        flags = gesture["fingers_extended"]
        deltas = gesture["finger_extension"]
        for i, (flag, delta) in enumerate(zip(flags[1:], deltas[1:]), start=1):
            self.assertEqual(flag, delta > 0.0, f"finger {i}")
        p.stop()

    def test_an_empty_gesture_still_carries_the_deltas(self):
        p = _pipeline()
        self.assertEqual(len(p._empty_gesture()["finger_extension"]), 5)
        p.stop()


class TestPointSurvivesNoise(unittest.TestCase):
    """The reported failure, reproduced as frames."""

    #: Well inside the band, and larger than the frame-to-frame landmark noise
    #: the accelerated backend adds.
    NOISE = _FINGER_EXT_MARGIN * 0.5

    def _pointing(self, frame, noise):
        """A pointing hand whose curled ring finger sits on its boundary."""
        wobble = noise if frame % 2 == 0 else -noise
        return _hand(index=0.55, middle=-0.4, ring=wobble, pinky=-0.4,
                     thumb=-0.5)

    def test_a_finger_on_its_boundary_does_not_flicker(self):
        p = _pipeline()
        seen = set()
        for frame in range(12):
            gesture = p._extract_gesture(self._pointing(frame, self.NOISE),
                                         now=1000.0 + frame / 30.0)
            seen.add(gesture["fingers_extended"])
        self.assertEqual(len(seen), 1, f"ring finger flickered: {seen}")
        p.stop()

    def test_and_the_sign_settles_on_point(self):
        p = _pipeline()
        sign = "unknown"
        for frame in range(12):
            sign = p._extract_gesture(self._pointing(frame, self.NOISE),
                                      now=1000.0 + frame / 30.0)["hand_sign"]
        self.assertEqual(sign, SIGN_POINT)
        p.stop()

    def test_noise_wider_than_the_band_still_gets_through(self):
        """The band must not be a mute button: a real move must still land."""
        p = _pipeline()
        for frame in range(6):
            p._extract_gesture(self._pointing(frame, self.NOISE),
                               now=1000.0 + frame / 30.0)
        # Ring finger genuinely extends, well past the band.
        for frame in range(6, 12):
            gesture = p._extract_gesture(
                _hand(index=0.55, middle=-0.4, ring=0.5, pinky=-0.4, thumb=-0.5),
                now=1000.0 + frame / 30.0)
        self.assertTrue(gesture["fingers_extended"][3])
        self.assertNotEqual(gesture["hand_sign"], SIGN_POINT)
        p.stop()

    def test_the_thumb_boundary_holds_too(self):
        p = _pipeline()
        seen = set()
        for frame in range(12):
            wobble = 0.10 + (_THUMB_EXT_MARGIN * 0.5
                             * (1 if frame % 2 == 0 else -1))
            gesture = p._extract_gesture(
                _hand(index=0.5, middle=0.5, ring=0.5, pinky=0.5, thumb=wobble),
                now=1000.0 + frame / 30.0)
            seen.add(gesture["fingers_extended"][0])
        self.assertEqual(len(seen), 1, f"thumb flickered: {seen}")
        p.stop()

    def test_the_band_scales_with_the_hand(self):
        """
        A hand at arm's length is smaller in pixels but not looser in shape,
        so the same wobble as a fraction of the palm must behave the same.
        """
        global PALM, PIP_REACH
        original = PALM
        try:
            for palm in (0.08, 0.2, 0.35):
                PALM, PIP_REACH = palm, 1.0 * palm
                p = _pipeline()
                seen = set()
                for frame in range(10):
                    seen.add(p._extract_gesture(
                        self._pointing(frame, self.NOISE),
                        now=1000.0 + frame / 30.0)["fingers_extended"])
                self.assertEqual(len(seen), 1, f"flickered at palm={palm}")
                p.stop()
        finally:
            PALM, PIP_REACH = original, 1.0 * original


class TestSlotIsolation(unittest.TestCase):

    def test_each_hand_keeps_its_own_previous_answer(self):
        """Sharing the band's memory between slots would couple two hands."""
        p = _pipeline(max_hands=2)
        # Slot 0 arrives already pointing, slot 1 arrives open.
        p._extract_gesture(_hand(index=0.55, middle=-0.4, ring=-0.4,
                                 pinky=-0.4, thumb=-0.5), slot=0, now=1000.0)
        p._extract_gesture(_hand(index=0.5, middle=0.5, ring=0.5, pinky=0.5,
                                 thumb=0.5), slot=1, now=1000.0)
        # Both now sit on the boundary; each should hold its own answer.
        boundary = _hand(index=0.55, middle=0.0, ring=0.0, pinky=0.0, thumb=-0.5)
        zero = p._extract_gesture(boundary, slot=0, now=1000.05)
        one = p._extract_gesture(boundary, slot=1, now=1000.05)
        self.assertEqual(zero["fingers_extended"][2:4], (False, False))
        self.assertEqual(one["fingers_extended"][2:4], (True, True))
        p.stop()

    def test_a_lost_hand_forgets_its_boundary(self):
        p = _pipeline()
        p._extract_gesture(_hand(index=0.5, middle=0.5, ring=0.5, pinky=0.5,
                                 thumb=0.5), now=1000.0)
        p._gs_slots[0].reset()
        gesture = p._extract_gesture(
            _hand(index=0.0, middle=0.0, ring=0.0, pinky=0.0, thumb=0.10),
            now=1000.05)
        self.assertEqual(gesture["fingers_extended"], (False,) * 5)
        p.stop()


if __name__ == "__main__":
    unittest.main()
