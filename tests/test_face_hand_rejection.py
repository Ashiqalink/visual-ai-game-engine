"""
test_face_hand_rejection.py — the palm network fires on faces, and the cost of
that is a whole tracking slot.

The failure it protects against is not "a spurious hand is drawn". It is that
every backend stops looking for more hands once it holds `max_hands` of them,
so one face sitting in a slot means the player's *second* hand is never
detected at all — reported from depthpong, which needs both.

Two things are pinned here:

  * the case that must not break — a real hand held over the face. A hand in
    front of a face is nearer the camera than the face is, so it images larger
    than the head; landmarks that all fit inside a face-sized box are not a
    hand in front of that face. Confident handedness overrides even that.
  * the accelerated backend must *drop the rect*, not merely hide the hand.
    Hiding it leaves the slot occupied and the detector idle, which is the
    whole bug.
"""

import queue
import unittest

import numpy as np

from visual_ai import openvino_hands
from visual_ai.pipeline import VisionPipeline


class _LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _LandmarkSet:
    def __init__(self, landmarks):
        self.landmark = landmarks


class _Classification:
    def __init__(self, label, score):
        self.label, self.score = label, score


class _Handedness:
    def __init__(self, label="Right", score=0.95):
        self.classification = [_Classification(label, score)]


class _HandResults:
    def __init__(self, sets, handedness=None):
        self.multi_hand_landmarks = sets or None
        self.multi_handedness = handedness


class _FakeHands:
    """MediaPipe's shape: a `process(rgb)` that takes no face box."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def process(self, _rgb):
        self.calls += 1
        return self.result


class _AcceleratedFakeHands(_FakeHands):
    """The accelerated shape: takes the face box and counts its own rejects."""

    accepts_face_box = True

    def __init__(self, result, rejects_per_call=0):
        super().__init__(result)
        self.face_rejects = 0
        self.seen_boxes = []
        self._rejects_per_call = rejects_per_call

    def process(self, _rgb, face_box=None):
        self.calls += 1
        self.seen_boxes.append(face_box)
        self.face_rejects += self._rejects_per_call
        return self.result


class _RelativeBox:
    def __init__(self, xmin, ymin, width, height):
        self.xmin, self.ymin = xmin, ymin
        self.width, self.height = width, height


class _LocationData:
    def __init__(self, box):
        self.relative_bounding_box = box


class _Detection:
    def __init__(self, box):
        self.location_data = _LocationData(box)


class _FakeFace:
    """One face, centred, covering the middle fifth of the frame."""

    def __init__(self, xmin=0.4, ymin=0.3, width=0.2, height=0.25):
        self.box = _RelativeBox(xmin, ymin, width, height)

    def process(self, _rgb):
        return type("R", (), {"detections": [_Detection(self.box)]})()


WIDTH, HEIGHT = 800, 600
#: The face _FakeFace reports, in pixels: x 320..480, y 180..330.
FACE_PX = (320, 180, 160, 150)


def _hand(cx=0.5, cy=0.6, spread=0.12, z=-0.02):
    """21 landmarks forming an open hand centred on (cx, cy), normalised."""
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


def _face_sized_hand():
    """A candidate that sits wholly inside the face box — i.e. is the face."""
    return _hand(cx=0.5, cy=0.42, spread=0.05)


def _hand_at_the_side():
    return _hand(cx=0.15, cy=0.6, spread=0.12)


def _pipeline(**kwargs):
    kwargs.setdefault("width", WIDTH)
    kwargs.setdefault("height", HEIGHT)
    kwargs.setdefault("camera_index", 999)
    return VisionPipeline(result_queue=queue.Queue(maxsize=2), **kwargs)


def _frame():
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


class TestTheFilterIsWiredUp(unittest.TestCase):
    """A test that could not fail is not a test: check the fixture first."""

    def test_the_face_sized_hand_really_is_inside_the_face_box(self):
        p = _pipeline()
        inside = sum(
            1 for lm in _face_sized_hand().landmark
            if FACE_PX[0] <= lm.x * WIDTH <= FACE_PX[0] + FACE_PX[2]
            and FACE_PX[1] <= lm.y * HEIGHT <= FACE_PX[1] + FACE_PX[3]
        )
        self.assertEqual(inside, 21)
        # ...and the hand at the side really is not.
        outside = sum(
            1 for lm in _hand_at_the_side().landmark
            if not (FACE_PX[0] <= lm.x * WIDTH <= FACE_PX[0] + FACE_PX[2])
        )
        self.assertEqual(outside, 21)
        p.stop()

    def test_the_face_fixture_lands_where_the_pixels_say(self):
        p = _pipeline()
        p._mp_face = _FakeFace()
        p._mp_hands = _FakeHands(_HandResults([]))
        payload = p._process_frame(_frame())
        self.assertEqual(payload["face_box"], FACE_PX)
        p.stop()


class TestPipelineFilter(unittest.TestCase):

    def _run(self, sets, handedness, face=True, **kwargs):
        p = _pipeline(max_hands=2, **kwargs)
        p._mp_face = _FakeFace() if face else None
        p._mp_hands = _FakeHands(_HandResults(sets, handedness))
        return p, p._process_frame(_frame())

    def test_a_face_does_not_take_the_second_hand_s_slot(self):
        p, payload = self._run(
            [_face_sized_hand(), _hand_at_the_side()],
            [_Handedness("Right", 0.55), _Handedness("Left", 0.97)])
        self.assertEqual(payload["hand_count"], 1)
        self.assertEqual(payload["face_hands_rejected"], 1)
        # The hand that survived is the one at the side, not the face.
        self.assertLess(payload["hands"][0]["index_pos"][0], WIDTH * 0.4)
        p.stop()

    def test_a_hand_held_over_the_face_survives(self):
        """The case the filter must not break: a real hand, over the face."""
        p, payload = self._run(
            # Bigger than the face box, as a hand in front of a face is.
            [_hand(cx=0.5, cy=0.42, spread=0.2)],
            [_Handedness("Right", 0.55)])
        self.assertEqual(payload["hand_count"], 1)
        self.assertEqual(payload["face_hands_rejected"], 0)
        p.stop()

    def test_confident_handedness_overrides_containment(self):
        p, payload = self._run([_face_sized_hand()], [_Handedness("Right", 0.99)])
        self.assertEqual(payload["hand_count"], 1)
        self.assertEqual(payload["face_hands_rejected"], 0)
        p.stop()

    def test_no_face_means_nothing_is_rejected(self):
        p, payload = self._run(
            [_face_sized_hand(), _hand_at_the_side()],
            [_Handedness("Right", 0.55), _Handedness("Left", 0.97)], face=False)
        self.assertEqual(payload["hand_count"], 2)
        self.assertEqual(payload["face_hands_rejected"], 0)
        p.stop()

    def test_the_filter_can_be_switched_off(self):
        p = _pipeline(max_hands=2)
        p._mp_face = _FakeFace()
        p.face_hand_filter = False
        p._mp_hands = _FakeHands(_HandResults(
            [_face_sized_hand(), _hand_at_the_side()],
            [_Handedness("Right", 0.55), _Handedness("Left", 0.97)]))
        payload = p._process_frame(_frame())
        self.assertEqual(payload["hand_count"], 2)
        self.assertEqual(payload["face_hands_rejected"], 0)
        p.stop()

    def test_handedness_stays_paired_with_its_hand(self):
        """Filtering one list without the other hands slot 0 the wrong label."""
        p, payload = self._run(
            [_face_sized_hand(), _hand_at_the_side()],
            [_Handedness("Right", 0.55), _Handedness("Left", 0.97)])
        self.assertEqual(payload["hands"][0]["handedness"], "Left")
        self.assertAlmostEqual(payload["hands"][0]["handedness_score"], 0.97)
        p.stop()

    def test_the_session_total_accumulates(self):
        p = _pipeline(max_hands=2)
        p._mp_face = _FakeFace()
        p._mp_hands = _FakeHands(_HandResults([_face_sized_hand()],
                                              [_Handedness("Right", 0.55)]))
        for _ in range(3):
            payload = p._process_frame(_frame())
        self.assertEqual(payload["face_hands_rejected"], 1)
        self.assertEqual(payload["face_hands_rejected_total"], 3)
        # ...and a no-hand payload reports the same running total.
        self.assertEqual(
            p._empty_payload(0.0, 0.0, _frame())["face_hands_rejected_total"], 3)
        p.stop()


class TestAcceleratedBackendHandover(unittest.TestCase):

    def test_the_face_box_is_handed_to_a_backend_that_takes_one(self):
        p = _pipeline(max_hands=2)
        p._mp_face = _FakeFace()
        p._mp_hands = _AcceleratedFakeHands(_HandResults([_hand_at_the_side()]),
                                            rejects_per_call=2)
        payload = p._process_frame(_frame())
        self.assertEqual(p._mp_hands.seen_boxes, [FACE_PX])
        # Rejections the backend made at the source are reported too — that is
        # the only place a player can see their face eating a slot.
        self.assertEqual(payload["face_hands_rejected"], 2)
        p.stop()

    def test_a_backend_that_takes_no_face_box_is_not_handed_one(self):
        p = _pipeline(max_hands=2)
        p._mp_face = _FakeFace()
        p._mp_hands = _FakeHands(_HandResults([_hand_at_the_side()]))
        p._process_frame(_frame())          # a TypeError here is the failure
        self.assertEqual(p._mp_hands.calls, 1)
        p.stop()

    def test_the_accelerated_backend_still_advertises_the_argument(self):
        """The handover is by attribute, so the attribute is the contract."""
        self.assertTrue(
            getattr(openvino_hands.OpenVINOHands, "accepts_face_box", False))


class TestAcceleratedGate(unittest.TestCase):
    """`_adopt` without a device: the gate is arithmetic, not inference."""

    def _hands(self, gate=0.9):
        hands = object.__new__(openvino_hands.OpenVINOHands)
        hands._rects = []
        hands.max_num_hands = 2
        hands.face_gate = gate
        hands.face_rejects = 0
        return hands

    @staticmethod
    def _detection(score, cx, cy, size=60.0):
        half = size / 2.0
        return {
            "score": score,
            "box": (cx - half, cy - half, cx + half, cy + half),
            "kp": np.array([[cx, cy + half], [cx, cy],
                            [cx, cy - half]] + [[cx, cy]] * 4, dtype=np.float32),
        }

    def test_a_marginal_detection_on_the_face_is_not_adopted(self):
        hands = self._hands()
        face = openvino_hands._face_bounds(FACE_PX)
        hands._adopt([self._detection(0.75, 400.0, 255.0)], face)
        self.assertEqual(hands._rects, [])
        self.assertEqual(hands.face_rejects, 1)

    def test_the_same_detection_off_the_face_is_adopted(self):
        hands = self._hands()
        face = openvino_hands._face_bounds(FACE_PX)
        hands._adopt([self._detection(0.75, 120.0, 500.0)], face)
        self.assertEqual(len(hands._rects), 1)
        self.assertEqual(hands.face_rejects, 0)

    def test_a_confident_detection_on_the_face_is_adopted(self):
        hands = self._hands()
        face = openvino_hands._face_bounds(FACE_PX)
        hands._adopt([self._detection(0.97, 400.0, 255.0)], face)
        self.assertEqual(len(hands._rects), 1)
        self.assertEqual(hands.face_rejects, 0)

    def test_without_a_face_box_nothing_changes(self):
        hands = self._hands()
        hands._adopt([self._detection(0.75, 400.0, 255.0)], None)
        self.assertEqual(len(hands._rects), 1)
        self.assertEqual(hands.face_rejects, 0)

    def test_a_zero_sized_face_box_is_no_box(self):
        self.assertIsNone(openvino_hands._face_bounds((0, 0, 0, 0)))
        self.assertIsNone(openvino_hands._face_bounds(None))

    def test_the_box_is_dilated_rather_than_taken_tight(self):
        bounds = openvino_hands._face_bounds((100, 100, 100, 100))
        self.assertLess(bounds[0], 100)
        self.assertGreater(bounds[2], 200)


if __name__ == "__main__":
    unittest.main()
