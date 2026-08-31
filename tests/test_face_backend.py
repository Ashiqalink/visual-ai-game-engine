"""
test_face_backend.py — the two face-detection backends behind one reader.

Face detection used to be MediaPipe's `FaceDetection` graph on the CPU and
nothing else. It now runs the same BlazeFace weights on whatever `accel`
resolves for the "face" consumer — the NPU by default — and keeps MediaPipe as
the fallback. Everything downstream (`target_x/y`, `face_box`, `face_count`,
and the face gate that stops a face taking a hand slot) reads one normalised
box, so what has to be pinned is:

  * both backends reduce to the same `_FaceBox`, in frame-normalised units,
    even though the accelerated one reports pixels;
  * a device that refuses the model does not silently land on OpenVINO's CPU
    plugin, which costs four cores where MediaPipe costs one;
  * `detect_face=False` still builds neither, and names no device;
  * the payload carries `face_device`, because a fallback nobody can see on
    the HUD is indistinguishable in play from an accelerator that works.

The last case needs the hardware and skips without it: it is the accuracy
check for the move, comparing the NPU's FP16 boxes against OpenVINO's own f32
CPU plugin on a recorded clip.
"""

import os
import queue
import unittest

import cv2
import numpy as np

from visual_ai import accel, openvino_zoo
from visual_ai.pipeline import VisionPipeline

FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir,
                       "benchmarks", "fixtures", "hand_motion.mp4")


class _Box:
    """Stand-in for MediaPipe's `relative_bounding_box`."""

    def __init__(self, xmin, ymin, width, height):
        self.xmin, self.ymin, self.width, self.height = xmin, ymin, width, height


class _MPFace:
    """Stand-in for `mediapipe.solutions.face_detection.FaceDetection`."""

    def __init__(self, *boxes):
        self.boxes = boxes

    def process(self, rgb):
        detections = [type("D", (), {"location_data": type("L", (), {
            "relative_bounding_box": box})()})() for box in self.boxes]
        return type("R", (), {"detections": detections})()


class _ZooFace:
    """Stand-in for `openvino_zoo.FaceDetector`: boxes in frame pixels."""

    def __init__(self, *boxes, device="NPU"):
        self.boxes = boxes
        self.device = device
        self.device_name = device
        self.closed = False

    def detect(self, rgb):
        return [openvino_zoo.Detection("face", 0.9, box) for box in self.boxes]

    def close(self):
        self.closed = True


def _pipeline(**kwargs):
    kwargs.setdefault("width", 800)
    kwargs.setdefault("height", 600)
    # Index 999 will not open; the thread is never started in these tests.
    kwargs.setdefault("camera_index", 999)
    return VisionPipeline(result_queue=queue.Queue(maxsize=2), **kwargs)


class TestOneBoxFromEitherBackend(unittest.TestCase):
    """`_face_boxes` is the only place the two backends differ."""

    def setUp(self):
        self.p = _pipeline(detect_face=False)
        self.rgb = np.zeros((600, 800, 3), dtype=np.uint8)

    def test_mediapipe_boxes_pass_through_normalised(self):
        self.p._mp_face = _MPFace(_Box(0.25, 0.10, 0.20, 0.30))
        (box,) = self.p._face_boxes(self.rgb)
        self.assertAlmostEqual(box.xmin, 0.25)
        self.assertAlmostEqual(box.ymin, 0.10)
        self.assertAlmostEqual(box.width, 0.20)
        self.assertAlmostEqual(box.height, 0.30)

    def test_accelerated_pixels_become_the_same_normalised_box(self):
        # The same face, in the units the OpenVINO wrapper reports: corners in
        # pixels of the 800x600 frame it was handed.
        self.p._face_net = _ZooFace((200.0, 60.0, 360.0, 240.0))
        (box,) = self.p._face_boxes(self.rgb)
        self.assertAlmostEqual(box.xmin, 0.25)
        self.assertAlmostEqual(box.ymin, 0.10)
        self.assertAlmostEqual(box.width, 0.20)
        self.assertAlmostEqual(box.height, 0.30)

    def test_the_accelerated_backend_wins_when_both_exist(self):
        self.p._face_net = _ZooFace((0.0, 0.0, 80.0, 60.0))
        self.p._mp_face = _MPFace(_Box(0.5, 0.5, 0.1, 0.1))
        (box,) = self.p._face_boxes(self.rgb)
        self.assertAlmostEqual(box.xmin, 0.0)

    def test_no_backend_is_an_empty_list_not_none(self):
        self.assertEqual(self.p._face_boxes(self.rgb), [])

    def test_no_face_in_frame_is_an_empty_list(self):
        self.p._mp_face = _MPFace()
        self.assertEqual(self.p._face_boxes(self.rgb), [])


class TestFallback(unittest.TestCase):

    def setUp(self):
        self._resolve = accel.resolve
        self._detector = openvino_zoo.FaceDetector

    def tearDown(self):
        accel.resolve = self._resolve
        openvino_zoo.FaceDetector = self._detector

    def _force(self, device, build):
        accel.resolve = lambda consumer, **kw: (device if consumer == "face"
                                                else self._resolve(consumer, **kw))
        openvino_zoo.FaceDetector = build

    def test_a_device_that_refuses_the_model_falls_back_to_mediapipe(self):
        # The zoo answers a refusal with its own CPU plugin. Taking that would
        # be a silent trade of MediaPipe's one core for four.
        refused = _ZooFace((0.0, 0.0, 10.0, 10.0), device="CPU")
        self._force("NPU", lambda device, **kw: refused)
        p = _pipeline(detect_face=True)
        self.assertIsNone(p._face_net)
        self.assertTrue(refused.closed)
        self.assertIn("MediaPipe", p.face_device_name)

    def test_a_device_that_takes_the_model_is_used_and_named(self):
        self._force("NPU", lambda device, **kw: _ZooFace((0.0, 0.0, 10.0, 10.0),
                                                         device=device))
        p = _pipeline(detect_face=True)
        self.assertIsNotNone(p._face_net)
        self.assertIsNone(p._mp_face)
        self.assertEqual(p.face_device_name, "NPU")

    def test_a_constructor_that_raises_falls_back_rather_than_propagating(self):
        def boom(device, **kw):
            raise RuntimeError("no driver")
        self._force("NPU", boom)
        p = _pipeline(detect_face=True)
        self.assertIsNone(p._face_net)
        self.assertIn("MediaPipe", p.face_device_name)


class TestDetectFaceOff(unittest.TestCase):

    def test_neither_backend_is_built(self):
        p = _pipeline(detect_face=False)
        self.assertIsNone(p._face_net)
        self.assertIsNone(p._mp_face)

    def test_no_device_is_named_for_a_graph_that_is_not_running(self):
        # Not "CPU (MediaPipe)": nothing is detecting faces at all, and a HUD
        # that prints a device for it is lying.
        self.assertEqual(_pipeline(detect_face=False).face_device_name, "")

    def test_a_running_graph_always_names_something(self):
        self.assertNotEqual(_pipeline(detect_face=True).face_device_name, "")


class TestPayloadCarriesTheDevice(unittest.TestCase):

    def test_the_no_hand_payload_has_face_device(self):
        p = _pipeline(detect_face=True)
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        payload = p._empty_payload(1.0, 2.0, frame)
        self.assertEqual(payload["face_device"], p.face_device_name)

    def test_status_reports_the_backend_and_the_device(self):
        p = _pipeline(detect_face=True)
        status = p.get_status()
        self.assertTrue(status["face_tracking"])
        self.assertEqual(status["face_device"], p.face_device_name)


@unittest.skipUnless("NPU" in accel.available_devices(),
                     "needs an OpenVINO NPU")
class TestNpuMatchesTheCpuPlugin(unittest.TestCase):
    """
    The accuracy check for the move, on recorded frames.

    FP16 on the NPU against f32 on OpenVINO's CPU plugin — same wrapper, same
    weights, so the only difference under test is the device. This can fail:
    a device that quantised the network differently, or a decode that read the
    output tensors in the wrong order, lands far outside these bounds.
    """

    @classmethod
    def setUpClass(cls):
        capture = cv2.VideoCapture(os.path.abspath(FIXTURE))
        cls.frames = []
        while len(cls.frames) < 12:
            ok, frame = capture.read()
            if not ok:
                break
            cls.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        if not cls.frames:
            raise unittest.SkipTest(f"no fixture clip at {FIXTURE}")

    #: A detection this far above `min_score` is not a threshold coin-flip.
    #: The clip has a 0.501 second "face" on two frames that FP16 moves to
    #: 0.499 and drops — which is the threshold moving, not the network
    #: disagreeing, and counting it as a failure would make this test noise.
    CONFIDENT = 0.55

    @staticmethod
    def _run(device, frames):
        net = openvino_zoo.FaceDetector(device)
        try:
            return [net.detect(frame) for frame in frames]
        finally:
            net.close()

    def setUp(self):
        self.on_cpu = self._run("CPU", self.frames)
        self.on_npu = self._run("NPU", self.frames)

    def test_the_same_confident_faces_are_found(self):
        def confident(per_frame):
            return [sum(d.score >= self.CONFIDENT for d in found)
                    for found in per_frame]
        self.assertEqual(confident(self.on_cpu), confident(self.on_npu))

    def test_the_primary_box_lands_in_the_same_place(self):
        deltas = [abs(np.asarray(c[0].box) - np.asarray(n[0].box)).max()
                  for c, n in zip(self.on_cpu, self.on_npu) if c and n]
        self.assertTrue(deltas, "no frame of the clip had a face on either device")
        # 2 px of a 1280x720 frame; the measured max is 1.5.
        self.assertLess(max(deltas), 2.0,
                        f"NPU box moved {max(deltas):.2f} px from the CPU's")

    def test_the_primary_score_barely_moves(self):
        deltas = [abs(c[0].score - n[0].score)
                  for c, n in zip(self.on_cpu, self.on_npu) if c and n]
        self.assertLess(max(deltas), 0.02, "FP16 changed the confidence")


if __name__ == "__main__":
    unittest.main()
