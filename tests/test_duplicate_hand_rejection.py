"""
Duplicate-hand rejection: one hand must never reach a game twice.

Three places can produce a copy, and each has its own test here:

  * the pipeline's `_drop_duplicate_hands`, which is backend-agnostic and now
    decides on palm-centre proximity alone -- the size and pose tests it used
    to run were what let copies through;
  * `OpenVINOHands._adopt`, where a fresh palm rect is matched against the
    rects already held: IoU alone misses a concentric rect of a different
    size, which is exactly what a copy looks like there;
  * `OpenVINOHands.process`, where a rect that has converged onto a hand
    already kept this frame is dropped with its rect, freeing the slot.

Run from the engine root:  python -m pytest tests/test_duplicate_hand_rejection.py
"""

from __future__ import annotations

import queue
import unittest

import numpy as np

from visual_ai import openvino_hands
from visual_ai.pipeline import VisionPipeline

WIDTH, HEIGHT = 640, 480


# ── MediaPipe-shaped fixtures ─────────────────────────────────────────────────

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
    def __init__(self, result):
        self.result = result

    def process(self, _rgb):
        return self.result


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


def _span_px(spread):
    """Wrist to middle-finger MCP of `_hand`, in pixels: 0.6 * spread * HEIGHT."""
    return 0.6 * spread * HEIGHT


def _pipeline(**kwargs):
    kwargs.setdefault("width", WIDTH)
    kwargs.setdefault("height", HEIGHT)
    kwargs.setdefault("camera_index", 999)
    kwargs.setdefault("detect_face", False)
    return VisionPipeline(result_queue=queue.Queue(maxsize=2), **kwargs)


def _frame():
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


# ── Pipeline stage ────────────────────────────────────────────────────────────

class TestPipelineFilter(unittest.TestCase):

    def _run(self, sets, handedness):
        p = _pipeline(max_hands=2)
        p._mp_hands = _FakeHands(_HandResults(sets, handedness))
        payload = p._process_frame(_frame())
        p.stop()
        return p, payload

    def test_the_fixture_spans_what_the_test_assumes(self):
        """A test that could not fail is not a test: check the fixture first."""
        pts = _hand(spread=0.12).landmark
        span = np.hypot((pts[9].x - pts[0].x) * WIDTH, (pts[9].y - pts[0].y) * HEIGHT)
        self.assertAlmostEqual(span, _span_px(0.12), places=6)

    def test_a_copy_of_different_apparent_size_is_dropped(self):
        """The case the old size test let through: same place, spans 20% apart."""
        p, payload = self._run(
            [_hand(spread=0.12), _hand(spread=0.144)],
            [_Handedness("Right", 0.9), _Handedness("Left", 0.6)])
        self.assertEqual(payload["hand_count"], 1)
        self.assertEqual(p.duplicate_hands_rejected_total, 1)

    def test_a_copy_slightly_offset_is_dropped(self):
        """Centres a third of a span apart -- inside the proximity gate."""
        offset = (_span_px(0.12) / 3.0) / WIDTH
        p, payload = self._run(
            [_hand(cx=0.5), _hand(cx=0.5 + offset)],
            [_Handedness("Right", 0.9), _Handedness("Right", 0.9)])
        self.assertEqual(payload["hand_count"], 1)
        self.assertEqual(p.duplicate_hands_rejected_total, 1)

    def test_two_hands_a_span_apart_both_survive(self):
        """Side by side, touching: the case the proximity gate must not merge."""
        offset = (_span_px(0.12) * 1.0) / WIDTH
        p, payload = self._run(
            [_hand(cx=0.4), _hand(cx=0.4 + offset)],
            [_Handedness("Right", 0.9), _Handedness("Left", 0.9)])
        self.assertEqual(payload["hand_count"], 2)
        self.assertEqual(p.duplicate_hands_rejected_total, 0)

    def test_the_first_hand_is_the_one_kept(self):
        """The established track keeps its slot; the copy's label goes with it."""
        p, payload = self._run(
            [_hand(spread=0.12), _hand(spread=0.144)],
            [_Handedness("Right", 0.9), _Handedness("Left", 0.6)])
        self.assertEqual(payload["hands"][0]["handedness"], "Right")
        self.assertAlmostEqual(payload["hands"][0]["handedness_score"], 0.9)

    def test_one_hand_mode_never_sees_a_second(self):
        """max_hands=1 is the mode that removes the question: one slot, one hand."""
        p = _pipeline(max_hands=1)
        p._mp_hands = _FakeHands(_HandResults([_hand()], [_Handedness()]))
        payload = p._process_frame(_frame())
        p.stop()
        self.assertEqual(p.max_hands, 1)
        self.assertEqual(payload["hand_count"], 1)
        self.assertEqual(len(payload["hands"]), 1)


# ── Accelerated backend: association ─────────────────────────────────────────

class TestAcceleratedAssociation(unittest.TestCase):
    """`_adopt` without a device: the association is arithmetic, not inference."""

    def _hands(self, held=()):
        hands = object.__new__(openvino_hands.OpenVINOHands)
        hands._rects = list(held)
        hands.max_num_hands = 2
        hands.face_gate = 0.9
        hands.face_rejects = 0
        hands.duplicate_rejects = 0
        return hands

    @staticmethod
    def _detection(cx, cy, size, score=0.95):
        half = size / 2.0
        return {
            "score": score,
            "box": (cx - half, cy - half, cx + half, cy + half),
            # Keypoint 0 below keypoint 2: an upright hand, so no rotation
            # and the ROI stays centred on the box.
            "kp": np.array([[cx, cy + half], [cx, cy],
                            [cx, cy - half]] + [[cx, cy]] * 4, dtype=np.float32),
        }

    @staticmethod
    def _rect_of(detection):
        return openvino_hands._rect_from_box(
            detection["box"], detection["kp"][0], detection["kp"][2],
            openvino_hands._PALM_SCALE, openvino_hands._PALM_SHIFT_Y)

    def test_the_fixture_is_the_failing_case(self):
        """Concentric, sizes 60 vs 100: IoU under the old gate, cover at 1.0."""
        big = openvino_hands._rect_bounds(self._rect_of(self._detection(300, 250, 100)))
        small = openvino_hands._rect_bounds(self._rect_of(self._detection(300, 250, 60)))
        self.assertLess(openvino_hands._iou(big, small), openvino_hands._ASSOCIATION_IOU)
        self.assertAlmostEqual(openvino_hands._cover(big, small), 1.0)

    def test_a_smaller_concentric_find_is_not_tracked_twice(self):
        hands = self._hands(held=[self._rect_of(self._detection(300, 250, 100))])
        hands._adopt([self._detection(300, 250, 60)])
        self.assertEqual(len(hands._rects), 1)
        self.assertEqual(hands.duplicate_rejects, 1)

    def test_a_larger_concentric_find_is_not_tracked_twice(self):
        hands = self._hands(held=[self._rect_of(self._detection(300, 250, 60))])
        hands._adopt([self._detection(300, 250, 100)])
        self.assertEqual(len(hands._rects), 1)
        self.assertEqual(hands.duplicate_rejects, 1)

    def test_an_iou_match_is_association_not_a_duplicate(self):
        """MediaPipe's own case: same hand, same size. Counted as nothing."""
        hands = self._hands(held=[self._rect_of(self._detection(300, 250, 100))])
        hands._adopt([self._detection(305, 252, 100)])
        self.assertEqual(len(hands._rects), 1)
        self.assertEqual(hands.duplicate_rejects, 0)

    def test_a_hand_beside_the_held_one_is_adopted(self):
        """Centres one rect-width apart: both hands, side by side."""
        first = self._detection(300, 250, 100)
        hands = self._hands(held=[self._rect_of(first)])
        size = self._rect_of(first)[2]
        hands._adopt([self._detection(300 + size, 250, 100)])
        self.assertEqual(len(hands._rects), 2)
        self.assertEqual(hands.duplicate_rejects, 0)

    def test_the_side_by_side_case_is_under_the_cover_gate(self):
        """Rects 1.8 hand-widths wide, centres a hand-width apart: ~0.45."""
        width = 100.0
        a = (0.0, 0.0, 1.8 * width, 1.8 * width)
        b = (width, 0.0, width + 1.8 * width, 1.8 * width)
        self.assertLess(openvino_hands._cover(a, b), openvino_hands._ASSOCIATION_COVER)
        self.assertGreater(openvino_hands._cover(a, b), 0.4)


# ── Accelerated backend: tracking ─────────────────────────────────────────────

def _pixels(cx, cy, span):
    """21 landmark pixels: wrist a span below the middle MCP at (cx, cy)."""
    pts = np.full((21, 2), (cx, cy), dtype=np.float32)
    pts[0] = (cx, cy + span)
    pts[9] = (cx, cy)
    pts[5] = (cx - 0.3 * span, cy)
    pts[17] = (cx + 0.3 * span, cy)
    pts[8] = (cx, cy - span)
    return pts


class TestAcceleratedTracker(unittest.TestCase):
    """`process` with both slots held: no detector runs, only the landmark pass."""

    def _tracker(self, hands_by_rect):
        hands = object.__new__(openvino_hands.OpenVINOHands)
        hands._rects = list(hands_by_rect)
        hands._pending = None
        hands.max_num_hands = 2
        hands.async_detector = True
        hands.min_tracking_confidence = 0.45
        hands.face_gate = 0.9
        hands.face_rejects = 0
        hands.duplicate_rejects = 0
        hands.timings = {"palm_ms": 0.0, "land_ms": 0.0,
                         "detector_runs": 0, "frames": 0}
        hands._run_landmarks = (
            lambda _rgb, rects: [hands_by_rect[rect] for rect in rects])
        return hands

    @staticmethod
    def _tracked(cx, cy, span, presence=0.98):
        return {"pixels": _pixels(cx, cy, span), "z": np.zeros(21, np.float32),
                "presence": presence, "handedness": 0.8}

    def test_two_rects_on_one_hand_come_out_as_one(self):
        first, second = (300.0, 250.0, 200.0, 0.0), (310.0, 240.0, 150.0, 0.0)
        hands = self._tracker({first: self._tracked(300, 250, 60),
                               second: self._tracked(305, 255, 70)})
        result = hands.process(_frame())
        self.assertEqual(len(result.multi_hand_landmarks), 1)
        self.assertEqual(len(result.multi_handedness), 1)
        self.assertEqual(hands.duplicate_rejects, 1)

    def test_the_copy_s_slot_is_freed(self):
        """The point of dropping at this stage: the detector can re-run."""
        first, second = (300.0, 250.0, 200.0, 0.0), (310.0, 240.0, 150.0, 0.0)
        hands = self._tracker({first: self._tracked(300, 250, 60),
                               second: self._tracked(305, 255, 70)})
        hands.process(_frame())
        self.assertEqual(len(hands._rects), 1)
        self.assertLess(len(hands._rects), hands.max_num_hands)

    def test_two_hands_a_span_apart_are_both_tracked(self):
        first, second = (300.0, 250.0, 200.0, 0.0), (360.0, 250.0, 200.0, 0.0)
        hands = self._tracker({first: self._tracked(300, 250, 60),
                               second: self._tracked(360, 250, 60)})
        result = hands.process(_frame())
        self.assertEqual(len(result.multi_hand_landmarks), 2)
        self.assertEqual(len(hands._rects), 2)
        self.assertEqual(hands.duplicate_rejects, 0)

    def test_a_lost_hand_is_not_a_duplicate(self):
        """Presence below the gate drops a rect for its own reason, uncounted."""
        first, second = (300.0, 250.0, 200.0, 0.0), (310.0, 240.0, 150.0, 0.0)
        hands = self._tracker({first: self._tracked(300, 250, 60),
                               second: self._tracked(305, 255, 70, presence=0.2)})
        result = hands.process(_frame())
        self.assertEqual(len(result.multi_hand_landmarks), 1)
        self.assertEqual(hands.duplicate_rejects, 0)


if __name__ == "__main__":
    unittest.main()
