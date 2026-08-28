"""
test_openvino_backends.py — the NPU/iGPU code paths, run without an NPU or an
iGPU (or, for that matter, without OpenVINO installed at all).

`test_accel.py` covers *which* device gets picked and the geometry helpers in
isolation. This file covers what happens after that: a fake OpenVINO runtime is
pushed into `sys.modules` and the three accelerated backends are driven end to
end against it — the hand graph's decode-and-track loop, the matting session's
per-shape compile cache, and the segmenter's output handling.

The stubs return *shapes*, not weights, so nothing here says the networks are
accurate; `benchmarks/bench_hand_backend.py` does that, on hardware. What these
tests can fail on is everything between the network and the caller, which is
where all of this code actually lives: letterbox undo, the crop inverse, z
scaling, handedness, when the detector re-runs, whether a dynamic shape can
reach the NPU (it must not — it segfaults there), and whether a device that
refuses the model lands on CPU with a HUD string that says so.
"""

import sys
import types
import unittest

import numpy as np

from visual_ai import accel, matting, openvino_hands, segment

# ── a fake OpenVINO ───────────────────────────────────────────────────────────

class _Port:
    """An output port: OpenVINO identifies these by name, and order varies."""

    def __init__(self, name):
        self.name = name

    def get_any_name(self):
        return self.name


class _Dim:
    """OpenVINO exposes `is_static` as a property, not a method — a stub that
    made it a method would let a `if dimension.is_static` (always true on a
    bound method) pass here and fail on hardware."""

    def __init__(self, length):
        self._length = length

    @property
    def is_static(self):
        return self._length is not None

    def get_length(self):
        if self._length is None:
            raise RuntimeError("dynamic dimension has no length")
        return self._length


class _PartialShape:
    def __init__(self, dims):
        self.dims = list(dims)

    def __iter__(self):
        return iter(_Dim(d) for d in self.dims)

    def to_shape(self):
        return [d for d in self.dims]


class _Input:
    def __init__(self, name, dims):
        self._name, self._dims = name, dims

    def get_any_name(self):
        return self._name

    def get_partial_shape(self):
        return _PartialShape(self._dims)


class _Model:
    def __init__(self, path, dims):
        self.path = path
        self.inputs = [_Input("input", dims)]
        self.reshaped = None

    def reshape(self, mapping):
        self.reshaped = list(mapping.values())[0].dims


class _Compiled:
    """A compiled model whose call/infer returns whatever the test set up."""

    def __init__(self, model, device, outputs):
        self.model = model
        self.device = device
        self._outputs = outputs
        self.inputs = model.inputs
        self.calls = 0

    def output(self, index):
        return list(self._outputs)[index]

    def create_infer_request(self):
        return self

    def infer(self, _inputs):
        self.calls += 1
        return dict(self._outputs)

    def __call__(self, _inputs):
        self.calls += 1
        return dict(self._outputs)


class _Core:
    def __init__(self):
        self.available_devices = ["CPU", "GPU", "NPU"]
        self.compiled = []

    # set by each test before use
    dims = [1, 3, -1, -1]
    outputs_for = staticmethod(lambda model: {})
    refuse = None

    def read_model(self, path):
        return _Model(path, list(self.dims))

    def compile_model(self, model, device, _hint=None):
        if self.refuse:
            raise RuntimeError(self.refuse)
        compiled = _Compiled(model, device, self.outputs_for(model))
        self.compiled.append(compiled)
        return compiled

    def get_property(self, device, _name):
        return f"Fake {device}"


class _FakeOpenVINO:
    """Installs a stand-in `openvino` module for the body of a `with`."""

    def __init__(self, **core_attrs):
        self.core = _Core()
        for key, value in core_attrs.items():
            setattr(self.core, key, value)

    def __enter__(self):
        module = types.ModuleType("openvino")
        module.Core = lambda: self.core
        module.PartialShape = _PartialShape
        self._previous = sys.modules.get("openvino")
        sys.modules["openvino"] = module
        return self

    def __exit__(self, *exc):
        if self._previous is None:
            sys.modules.pop("openvino", None)
        else:
            sys.modules["openvino"] = self._previous


# ── the hand graph ────────────────────────────────────────────────────────────

_WIDTH, _HEIGHT = 640, 480
#: The letterbox this frame size produces, derived here rather than imported so
#: a change to `_letterbox` shows up as a failure instead of moving both sides.
_SCALE = openvino_hands._PALM_SIZE / _WIDTH          # 0.3
_TOP = (openvino_hands._PALM_SIZE - round(_HEIGHT * _SCALE)) // 2

#: A stride-8 cell near the middle of the grid: anchors are two per cell, laid
#: out row-major, so cell (12, 12) of the 24x24 map starts at 2 * (12 * 24 + 12).
_ANCHOR = 2 * (12 * 24 + 12)


def _palm_output(score=0.99, size_px=48.0):
    """
    A palm-detection output with exactly one confident detection on `_ANCHOR`.

    `size_px` is in the network's own 192-px units. Keypoint 0 sits below
    keypoint 2, which is the orientation that makes the ROI's rotation zero and
    the crop's inverse transform something a test can predict by hand.
    """
    boxes = np.zeros((1, 2016, 18), np.float32)
    boxes[0, _ANCHOR, 2] = size_px                       # width
    boxes[0, _ANCHOR, 3] = size_px                       # height
    boxes[0, _ANCHOR, 4 + 0 * 2 + 1] = size_px / 2.0     # kp0, below centre
    boxes[0, _ANCHOR, 4 + 2 * 2 + 1] = -size_px / 2.0    # kp2, above centre
    logits = np.full((1, 2016, 1), -60.0, np.float32)
    logits[0, _ANCHOR, 0] = np.log(score / (1.0 - score))
    return {_Port("boxes"): boxes, _Port("scores"): logits}


def _land_output(points, presence=0.99, handedness=0.9):
    """
    A hand-landmark output in MediaPipe's four-port shape.

    The port names matter: the module picks screen landmarks over world
    landmarks, and presence over handedness, by sorting the two same-sized
    pairs by name. These are the names the shipped `.tflite` uses.
    """
    screen = np.asarray(points, np.float32).reshape(1, 1, 1, 63)
    world = np.zeros((1, 1, 1, 63), np.float32)
    return {
        _Port("Identity"): screen,
        _Port("Identity_1"): np.asarray([[presence]], np.float32),
        _Port("Identity_2"): np.asarray([[handedness]], np.float32),
        _Port("Identity_3"): world,
    }


class _StubRequest:
    """Returns each queued output in turn, and remembers how often it ran."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0
        self.results = None

    def infer(self, _inputs):
        self.calls += 1
        return self._outputs[min(self.calls - 1, len(self._outputs) - 1)]

    # The async surface the detector goes through. A stub has always finished
    # by the time it is asked, so `wait_for` is never the reason a detection
    # is left pending — only the module's own rule about when to overlap is.
    def start_async(self, inputs):
        self.results = self.infer(inputs)

    def wait(self):
        pass

    def wait_for(self, _timeout):
        return True


def _hands(palm_outputs, land_outputs, **kwargs):
    """An `OpenVINOHands` wired to stub networks, without compiling anything."""
    hands = object.__new__(openvino_hands.OpenVINOHands)
    hands._palm = _StubRequest(palm_outputs)
    hands._land = _StubRequest(land_outputs)
    hands.device = "NPU"
    hands.tier = "full"
    hands.max_num_hands = kwargs.get("max_num_hands", 1)
    hands.min_detection_confidence = kwargs.get("min_detection_confidence", 0.7)
    hands.min_tracking_confidence = kwargs.get("min_tracking_confidence", 0.65)
    hands.async_detector = kwargs.get("async_detector", True)
    hands._rects = []
    hands._pending = None
    hands.timings ={"palm_ms": 0.0, "land_ms": 0.0, "detector_runs": 0, "frames": 0}
    return hands


def _blank_frame():
    return np.zeros((_HEIGHT, _WIDTH, 3), np.uint8)


class TestHandGraphEndToEnd(unittest.TestCase):
    """One detection, one landmark pass, and the numbers that come back out."""

    def setUp(self):
        self.rects = []
        self._real_crop = openvino_hands._crop

        def recording_crop(rgb, rect, out_size):
            self.rects.append(rect)
            return self._real_crop(rgb, rect, out_size)

        openvino_hands._crop = recording_crop
        self.addCleanup(setattr, openvino_hands, "_crop", self._real_crop)

    def test_a_detection_is_undone_back_out_of_the_letterbox(self):
        # Anchor centre 12.5/24 of a 192-px square; the frame was fitted at
        # 192/640 and centred vertically, so the same point in the frame is:
        expected_x = (12.5 / 24 * openvino_hands._PALM_SIZE) / _SCALE
        expected_y = (12.5 / 24 * openvino_hands._PALM_SIZE - _TOP) / _SCALE

        hands = _hands([_palm_output()], [_land_output(np.zeros(63))])
        hands.process(_blank_frame())

        self.assertEqual(len(self.rects), 1)
        rect_cx, rect_cy, size, angle = self.rects[0]
        self.assertAlmostEqual(rect_cx, expected_x, places=2)
        self.assertAlmostEqual(angle, 0.0, places=6)
        # The ROI is shifted up the hand by half a box height before scaling.
        box_h = 48.0 / _SCALE
        self.assertAlmostEqual(rect_cy, expected_y - 0.5 * box_h, places=2)
        self.assertAlmostEqual(size, box_h * openvino_hands._PALM_SCALE, places=2)

    def test_the_crop_centre_maps_back_to_the_roi_centre(self):
        points = np.tile([112.0, 112.0, 0.0], 21)
        hands = _hands([_palm_output()], [_land_output(points)])
        result = hands.process(_blank_frame())

        rect_cx, rect_cy, _, _ = self.rects[0]
        landmark = result.multi_hand_landmarks[0].landmark[0]
        self.assertAlmostEqual(landmark.x, rect_cx / _WIDTH, places=4)
        self.assertAlmostEqual(landmark.y, rect_cy / _HEIGHT, places=4)

    def test_a_crop_corner_maps_back_to_the_roi_corner(self):
        points = np.zeros(63, np.float32)          # every landmark at (0, 0)
        hands = _hands([_palm_output()], [_land_output(points)])
        result = hands.process(_blank_frame())

        rect_cx, rect_cy, size, _ = self.rects[0]
        landmark = result.multi_hand_landmarks[0].landmark[0]
        self.assertAlmostEqual(landmark.x, (rect_cx - size / 2) / _WIDTH, places=4)
        self.assertAlmostEqual(landmark.y, (rect_cy - size / 2) / _HEIGHT, places=4)

    def test_z_is_scaled_by_the_roi_and_normalised_like_x(self):
        # MediaPipe's z is in crop units; a consumer's threshold only survives
        # the swap of backends if it is rescaled by the ROI and by the width.
        points = np.tile([112.0, 112.0, 10.0], 21)
        hands = _hands([_palm_output()], [_land_output(points)])
        result = hands.process(_blank_frame())

        _, _, size, _ = self.rects[0]
        expected = 10.0 * (size / openvino_hands._LAND_SIZE) / _WIDTH
        self.assertAlmostEqual(result.multi_hand_landmarks[0].landmark[0].z,
                               expected, places=6)

    def test_twenty_one_landmarks_come_back(self):
        hands = _hands([_palm_output()], [_land_output(np.zeros(63))])
        result = hands.process(_blank_frame())
        self.assertEqual(len(result.multi_hand_landmarks[0].landmark), 21)


class TestHandedness(unittest.TestCase):

    def _label(self, handedness):
        hands = _hands([_palm_output()],
                       [_land_output(np.zeros(63), handedness=handedness)])
        result = hands.process(_blank_frame())
        return result.multi_handedness[0].classification[0]

    def test_a_high_score_is_a_right_hand(self):
        classification = self._label(0.92)
        self.assertEqual(classification.label, "Right")
        self.assertAlmostEqual(classification.score, 0.92, places=5)

    def test_a_low_score_is_a_left_hand_with_the_score_inverted(self):
        # MediaPipe reports the confidence of the label it gave, not P(right).
        classification = self._label(0.08)
        self.assertEqual(classification.label, "Left")
        self.assertAlmostEqual(classification.score, 0.92, places=5)


class TestTracking(unittest.TestCase):
    """When the detector runs is the whole reason this path is cheap."""

    def test_the_detector_does_not_re_run_while_tracking_holds(self):
        hands = _hands([_palm_output()], [_land_output(np.zeros(63))])
        for _ in range(4):
            hands.process(_blank_frame())
        self.assertEqual(hands._palm.calls, 1)
        self.assertEqual(hands._land.calls, 4)
        self.assertEqual(hands.timings["detector_runs"], 1)
        self.assertEqual(hands.timings["frames"], 4)

    def test_losing_presence_drops_the_hand_and_re_runs_the_detector(self):
        hands = _hands(
            [_palm_output()],
            [_land_output(np.zeros(63), presence=0.99),
             _land_output(np.zeros(63), presence=0.10),
             _land_output(np.zeros(63), presence=0.99)],
            min_tracking_confidence=0.65)

        self.assertIsNotNone(hands.process(_blank_frame()).multi_hand_landmarks)
        lost = hands.process(_blank_frame())
        self.assertIsNone(lost.multi_hand_landmarks)
        self.assertIsNone(lost.multi_handedness)
        self.assertEqual(hands._palm.calls, 1)

        hands.process(_blank_frame())
        self.assertEqual(hands._palm.calls, 2)   # re-acquired

    def test_a_second_detection_of_the_same_hand_is_not_tracked_twice(self):
        # Two overlapping palm detections: association has to collapse them, or
        # one hand is landmarked twice every frame at twice the cost.
        output = _palm_output()
        boxes = list(output.values())[0]
        neighbour = _ANCHOR + 2                 # the next cell along, overlapping
        boxes[0, neighbour] = boxes[0, _ANCHOR]
        logits = list(output.values())[1]
        logits[0, neighbour, 0] = np.log(0.95 / 0.05)

        hands = _hands([output], [_land_output(np.zeros(63))], max_num_hands=2)
        result = hands.process(_blank_frame())
        self.assertEqual(len(result.multi_hand_landmarks), 1)

    def test_a_low_score_detection_is_below_the_confidence_gate(self):
        hands = _hands([_palm_output(score=0.30)], [_land_output(np.zeros(63))],
                       min_detection_confidence=0.7)
        result = hands.process(_blank_frame())
        self.assertIsNone(result.multi_hand_landmarks)
        self.assertEqual(hands._land.calls, 0)

    def test_close_forgets_the_tracked_rois(self):
        hands = _hands([_palm_output()], [_land_output(np.zeros(63))])
        hands.process(_blank_frame())
        self.assertEqual(len(hands._rects), 1)
        hands.close()
        self.assertEqual(hands._rects, [])


class TestHandCompilation(unittest.TestCase):
    """Construction is where a missing driver has to turn into a fallback."""

    def _devices(self, *devices):
        real = accel.available_devices
        accel.available_devices = lambda: list(devices)
        self.addCleanup(setattr, accel, "available_devices", real)

    def test_both_networks_are_compiled_for_the_requested_device(self):
        self._devices("CPU", "GPU", "NPU")
        with _FakeOpenVINO() as fake:
            hands = openvino_hands.OpenVINOHands(device="NPU")
        self.assertEqual([c.device for c in fake.core.compiled], ["NPU", "NPU"])
        self.assertEqual(hands.device, "NPU")

    def test_the_latency_hint_is_asked_for(self):
        # Without it the GPU plugin picks a throughput schedule, which is the
        # 5.5x-latency half of the iGPU trade-off recorded in accel.py.
        self._devices("CPU", "GPU")
        hints = []
        with _FakeOpenVINO() as fake:
            real = fake.core.compile_model

            def watching(model, device, hint=None):
                hints.append(hint)
                return real(model, device, hint)

            fake.core.compile_model = watching
            openvino_hands.OpenVINOHands(device="GPU")
        self.assertTrue(all(h and h.get("PERFORMANCE_HINT") == "LATENCY"
                            for h in hints), hints)

    def test_model_complexity_zero_takes_the_lite_weights(self):
        self._devices("CPU", "NPU")
        with _FakeOpenVINO() as fake:
            openvino_hands.OpenVINOHands(device="NPU", model_complexity=0)
        self.assertTrue(all("lite" in c.model.path for c in fake.core.compiled),
                        [c.model.path for c in fake.core.compiled])

    def test_a_refusing_device_raises_the_catchable_error(self):
        self._devices("CPU", "NPU")
        with _FakeOpenVINO(refuse="device busy"):
            with self.assertRaises(openvino_hands.OpenVINOHandsUnavailable):
                openvino_hands.OpenVINOHands(device="NPU")

    def test_build_turns_that_into_none_so_the_caller_falls_back(self):
        self._devices("CPU", "NPU")
        with _FakeOpenVINO(refuse="device busy"):
            self.assertIsNone(openvino_hands.build(device="NPU"))

    def test_build_passes_the_resolved_device_not_the_request(self):
        # "auto" on a machine with no NPU has to reach the iGPU, not "AUTO".
        self._devices("CPU", "GPU")
        import os
        os.environ[accel.ENV_ACCEL] = "auto"
        self.addCleanup(os.environ.pop, accel.ENV_ACCEL, None)
        with _FakeOpenVINO() as fake:
            hands = openvino_hands.build(max_num_hands=2)
        self.assertIsNotNone(hands)
        self.assertEqual(hands.device, "GPU")
        self.assertEqual(hands.max_num_hands, 2)
        self.assertEqual([c.device for c in fake.core.compiled], ["GPU", "GPU"])


# ── matting ───────────────────────────────────────────────────────────────────

class TestMattingSession(unittest.TestCase):
    """
    The per-shape compile cache.

    This is not an optimisation. The NPU plugin does not refuse a dynamic
    model — it takes the process down with a segfault no `except` can catch —
    so every shape reaching `compile_model` must already be static.
    """

    def _session(self, dims=(1, 3, -1, -1)):
        fake = _FakeOpenVINO(dims=list(dims))
        fake.core.outputs_for = lambda model: {
            _Port("out"): np.zeros((1, 1, 8, 8), np.float32)}
        return fake

    def test_the_declared_shape_reports_its_dynamic_axes(self):
        with self._session([1, 3, -1, -1]) as fake:
            fake.core.read_model = lambda path: _Model(path, [1, 3, None, None])
            session = matting._OpenVINOSession("modnet.onnx", "GPU")
            declared = session.get_inputs()[0]
        self.assertEqual(declared.name, "input")
        self.assertEqual(declared.shape, [1, 3, "?", "?"])

    def test_every_compile_is_given_a_static_shape(self):
        with self._session() as fake:
            session = matting._OpenVINOSession("modnet.onnx", "NPU")
            session.run(None, {"input": np.zeros((1, 3, 352, 640), np.float32)})
        self.assertEqual(len(fake.core.compiled), 1)
        self.assertEqual(fake.core.compiled[0].model.reshaped, [1, 3, 352, 640])
        self.assertEqual(fake.core.compiled[0].device, "NPU")

    def test_the_same_shape_is_compiled_once_and_reused(self):
        with self._session() as fake:
            session = matting._OpenVINOSession("modnet.onnx", "GPU")
            for _ in range(3):
                session.run(None, {"input": np.zeros((1, 3, 352, 640), np.float32)})
        self.assertEqual(len(fake.core.compiled), 1)
        self.assertEqual(fake.core.compiled[0].calls, 3)

    def test_a_new_shape_compiles_again_rather_than_reusing_the_wrong_plan(self):
        with self._session() as fake:
            session = matting._OpenVINOSession("modnet.onnx", "GPU")
            session.run(None, {"input": np.zeros((1, 3, 352, 640), np.float32)})
            session.run(None, {"input": np.zeros((1, 3, 512, 512), np.float32)})
        self.assertEqual([c.model.reshaped for c in fake.core.compiled],
                         [[1, 3, 352, 640], [1, 3, 512, 512]])

    def test_run_returns_the_output_as_an_array(self):
        with self._session():
            session = matting._OpenVINOSession("modnet.onnx", "GPU")
            output = session.run(None, {"input": np.zeros((1, 3, 352, 640), np.float32)})
        self.assertEqual(len(output), 1)
        self.assertEqual(np.asarray(output[0]).shape, (1, 1, 8, 8))


class TestMattingFallback(unittest.TestCase):
    """A device that cannot be built must land on onnxruntime, and say so."""

    def setUp(self):
        real = accel.available_devices
        accel.available_devices = lambda: ["CPU", "GPU"]
        self.addCleanup(setattr, accel, "available_devices", real)

        import os
        os.environ[accel.ENV_ACCEL] = "gpu"
        self.addCleanup(os.environ.pop, accel.ENV_ACCEL, None)

        self.addCleanup(setattr, matting, "model_path", matting.model_path)
        matting.model_path = lambda: "modnet.onnx"

        # A stand-in onnxruntime, so the fallback branch can be reached on a
        # machine with no MODNet weights on disk.
        import onnxruntime as ort
        self.addCleanup(setattr, ort, "InferenceSession", ort.InferenceSession)
        self.addCleanup(setattr, ort, "get_available_providers",
                        ort.get_available_providers)
        ort.get_available_providers = lambda: ["CPUExecutionProvider"]
        ort.InferenceSession = lambda path, providers=None: _FakeORT(providers)

    def test_the_gpu_session_is_used_when_it_builds(self):
        fake = _FakeOpenVINO()
        fake.core.outputs_for = lambda model: {_Port("out"): np.zeros((1, 1, 4, 4))}
        with fake:
            matter = matting.PortraitMatter()
        self.assertEqual(matter.device_name, "GPU")
        self.assertIsInstance(matter._session, matting._OpenVINOSession)

    def test_a_failed_gpu_falls_back_to_onnxruntime_and_names_the_loss(self):
        real = matting._OpenVINOSession
        self.addCleanup(setattr, matting, "_OpenVINOSession", real)

        def explode(path, device):
            raise RuntimeError("simulated plugin failure")

        matting._OpenVINOSession = explode
        matter = matting.PortraitMatter()
        self.assertIsInstance(matter._session, _FakeORT)
        # The HUD has to distinguish this from a machine that never had a GPU.
        self.assertIn("CPU (onnxruntime)", matter.device_name)
        self.assertNotEqual(matter.device_name, "CPU (onnxruntime)")


class _FakeORT:
    """The three methods `PortraitMatter` asks an onnxruntime session for."""

    def __init__(self, providers=None):
        self.providers = providers

    class _Arg:
        name = "input"
        shape = [1, 3, "?", "?"]

    def get_inputs(self):
        return [self._Arg()]

    def get_providers(self):
        """What onnxruntime *accepted*, which is why the HUD reads it back
        rather than trusting the requested list."""
        return list(self.providers or ["CPUExecutionProvider"])

    def run(self, _names, feed):
        tensor = next(iter(feed.values()))
        return [np.zeros((1, 1, tensor.shape[2], tensor.shape[3]), np.float32)]


# ── segmentation ──────────────────────────────────────────────────────────────

class TestSegmenter(unittest.TestCase):
    """
    The iGPU segmenter, for the day OpenVINO's TFLite frontend learns
    `Convolution2DTransposeBias`. Until then the class cannot load the shipped
    weights — but the code around the network is testable now, and a silent
    change to it would otherwise not surface until that day.
    """

    def _fake(self, output):
        fake = _FakeOpenVINO(dims=[1, 144, 256, 1])
        fake.core.outputs_for = lambda model: {_Port("out"): output}
        return fake

    def test_the_input_size_is_read_off_the_model_as_width_then_height(self):
        with self._fake(np.zeros((1, 144, 256, 1), np.float32)):
            segmenter = segment._OpenVINOSegmenter("selfie.tflite", "GPU")
        self.assertEqual(segmenter.input_size, (256, 144))
        self.assertEqual(segmenter.device, "GPU")

    def test_a_single_channel_output_is_thresholded_at_a_half(self):
        confidence = np.zeros((1, 144, 256, 1), np.float32)
        confidence[0, :10, :10, 0] = 0.9
        confidence[0, 20:30, :10, 0] = 0.49
        with self._fake(confidence):
            segmenter = segment._OpenVINOSegmenter("selfie.tflite", "GPU")
            mask = segmenter.person_mask(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(mask.shape, (144, 256))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(mask[5, 5]), 255)
        self.assertEqual(int(mask[25, 5]), 0)

    def test_a_two_channel_output_reads_the_person_channel_not_the_background(self):
        # Channel 0 is background. Reading it would invert the mask, which
        # looks like a working segmenter until the person disappears.
        confidence = np.zeros((1, 144, 256, 2), np.float32)
        confidence[0, ..., 0] = 0.9        # background
        confidence[0, ..., 1] = 0.1        # person
        confidence[0, :10, :10, 1] = 0.8
        with self._fake(confidence):
            segmenter = segment._OpenVINOSegmenter("selfie.tflite", "GPU")
            mask = segmenter.person_mask(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(int(mask[5, 5]), 255)
        self.assertEqual(int(mask[100, 100]), 0)

    def test_the_frame_is_resized_to_what_the_network_declared(self):
        seen = []
        confidence = np.zeros((1, 144, 256, 1), np.float32)
        with self._fake(confidence):
            segmenter = segment._OpenVINOSegmenter("selfie.tflite", "GPU")
            real = segmenter._request.infer

            def watching(inputs):
                seen.append(np.asarray(inputs[0]).shape)
                return real(inputs)

            segmenter._request.infer = watching
            segmenter.person_mask(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(seen, [(1, 144, 256, 3)])

    def test_a_device_that_rejects_the_model_falls_back_and_says_which(self):
        real_path = segment._model_path
        real_class = segment._OpenVINOSegmenter
        self.addCleanup(setattr, segment, "_model_path", real_path)
        self.addCleanup(setattr, segment, "_OpenVINOSegmenter", real_class)
        segment._model_path = lambda: "selfie.tflite"

        def refuse(model_file, device):
            raise RuntimeError("No translator found for Convolution2DTransposeBias")

        segment._OpenVINOSegmenter = refuse

        real_devices = accel.available_devices
        accel.available_devices = lambda: ["CPU", "GPU"]
        self.addCleanup(setattr, accel, "available_devices", real_devices)

        import os
        os.environ[accel.ENV_ACCEL] = "gpu"
        self.addCleanup(os.environ.pop, accel.ENV_ACCEL, None)

        # Let the MediaPipe branch build without the real weights on disk.
        try:
            from mediapipe.tasks.python import vision
        except Exception as exc:                 # pragma: no cover
            self.skipTest(f"mediapipe not installed: {exc}")
        self.addCleanup(setattr, vision.ImageSegmenter, "create_from_options",
                        vision.ImageSegmenter.create_from_options)
        vision.ImageSegmenter.create_from_options = staticmethod(
            lambda _options: _FakeMediaPipeSegmenter())

        segmenter = segment.PersonSegmenter()
        name = segmenter.device_name
        segmenter.close()
        self.assertIn("GPU rejected the model", name)
        self.assertIn("CPU (MediaPipe)", name)


class _FakeMediaPipeSegmenter:
    """Stands in for `vision.ImageSegmenter`; only close() is reached here."""

    def close(self):
        pass


if __name__ == "__main__":
    unittest.main()
