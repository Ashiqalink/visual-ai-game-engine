"""
test_accel.py — device selection for the NPU/iGPU path, and the geometry the
OpenVINO hand graph rebuilds.

None of this needs an accelerator: `available_devices` is stubbed, so the
"machine has an NPU" and "machine has nothing" cases both run everywhere. The
inference itself is checked against MediaPipe in
`benchmarks/bench_hand_backend.py`, which does need the hardware.
"""

import math
import os
import unittest

import numpy as np

from visual_ai import accel, openvino_hands


class _Devices:
    """Pretend a machine has exactly `devices`, and clear the env keys."""

    def __init__(self, *devices):
        self.devices = list(devices)

    def __enter__(self):
        self._real = accel.available_devices
        accel.available_devices = lambda: list(self.devices)
        self._env = {key: os.environ.pop(key, None)
                     for key in (accel.ENV_ACCEL, accel.ENV_HAND_DEVICE,
                                 accel.ENV_MATTE_DEVICE)}
        return self

    def __exit__(self, *exc):
        accel.available_devices = self._real
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestResolve(unittest.TestCase):

    def test_auto_is_the_default(self):
        # Nothing in the environment: the accelerated path is what a plain
        # `play sling` gets now.
        with _Devices("CPU", "GPU", "NPU"):
            self.assertEqual(accel.resolve("hand"), "NPU")
            self.assertEqual(accel.resolve("matte"), "GPU")

    def test_the_default_is_still_the_cpu_path_on_a_machine_without_devices(self):
        # The whole safety argument for defaulting to "auto": every way it can
        # fail lands back on exactly the CPU code that used to be the default.
        with _Devices("CPU"):
            self.assertIsNone(accel.resolve("hand"))
            self.assertIsNone(accel.resolve("matte"))
        with _Devices():
            self.assertIsNone(accel.resolve("hand"))
            self.assertIsNone(accel.resolve("matte"))

    def test_an_explicit_preset_of_off_beats_the_default(self):
        with _Devices("CPU", "GPU", "NPU"):
            os.environ[accel.ENV_ACCEL] = "off"
            self.assertIsNone(accel.resolve("hand"))
            self.assertIsNone(accel.resolve("matte"))

    def test_auto_puts_hands_on_the_npu_and_matting_on_the_igpu(self):
        with _Devices("CPU", "GPU", "NPU"):
            os.environ[accel.ENV_ACCEL] = "auto"
            self.assertEqual(accel.resolve("hand"), "NPU")
            self.assertEqual(accel.resolve("matte"), "GPU")

    def test_auto_falls_back_to_the_igpu_for_hands_without_an_npu(self):
        with _Devices("CPU", "GPU"):
            os.environ[accel.ENV_ACCEL] = "auto"
            self.assertEqual(accel.resolve("hand"), "GPU")

    def test_auto_on_gpu_present_no_npu_resolves_to_gpu_not_cpu(self):
        # GPU-present/no-NPU is *not* a fallback to the CPU path: hands
        # resolve to GPU (OpenVINO, hand_gate 0.45), not None (MediaPipe,
        # 0.65).  The old safety claim missed this case.
        with _Devices("CPU", "GPU"):
            self.assertEqual(accel.resolve("hand"), "GPU")
            # Matting also goes to GPU in this topology.
            self.assertEqual(accel.resolve("matte"), "GPU")

    def test_auto_leaves_matting_on_the_cpu_without_an_igpu(self):
        # "matte" deliberately has no CPU entry: OpenVINO CPU is not better
        # than the onnxruntime session it would replace.
        with _Devices("CPU", "NPU"):
            os.environ[accel.ENV_ACCEL] = "auto"
            self.assertIsNone(accel.resolve("matte"))

    def test_a_missing_device_resolves_to_none_rather_than_raising(self):
        with _Devices("CPU"):
            os.environ[accel.ENV_ACCEL] = "npu"
            self.assertIsNone(accel.resolve("hand"))

    def test_an_explicit_device_beats_a_preset_of_off(self):
        with _Devices("CPU", "GPU", "NPU"):
            # Pinned, not left unset: unset is "auto" now, and this test is
            # about beating "off" specifically.
            os.environ[accel.ENV_ACCEL] = "off"
            self.assertEqual(accel.resolve("hand", explicit="GPU"), "GPU")

    def test_an_explicit_off_beats_a_preset_of_auto(self):
        with _Devices("CPU", "GPU", "NPU"):
            os.environ[accel.ENV_ACCEL] = "auto"
            self.assertIsNone(accel.resolve("hand", explicit="off"))

    def test_the_environment_carries_a_device_into_a_child_process(self):
        with _Devices("CPU", "GPU", "NPU"):
            os.environ[accel.ENV_HAND_DEVICE] = "gpu"
            self.assertEqual(accel.resolve("hand"), "GPU")

    def test_an_unknown_preset_is_off_rather_than_an_error(self):
        with _Devices("CPU", "GPU", "NPU"):
            os.environ[accel.ENV_ACCEL] = "quantum"
            self.assertIsNone(accel.resolve("hand"))


class TestDescribe(unittest.TestCase):
    """The HUD string has to distinguish 'not asked for' from 'asked and lost'."""

    def test_a_resolved_device_names_itself(self):
        with _Devices("CPU", "NPU"):
            self.assertEqual(accel.describe("hand", "NPU", "CPU (MediaPipe)"), "NPU")

    def test_no_request_reads_as_the_plain_fallback(self):
        with _Devices("CPU"):
            self.assertEqual(accel.describe("hand", None, "CPU (MediaPipe)"),
                             "CPU (MediaPipe)")

    def test_a_failed_request_says_so(self):
        with _Devices("CPU"):
            os.environ[accel.ENV_ACCEL] = "npu"
            described = accel.describe("hand", None, "CPU (MediaPipe)")
            self.assertNotEqual(described, "CPU (MediaPipe)")
            self.assertIn("CPU (MediaPipe)", described)


class TestGraphGeometry(unittest.TestCase):
    """The parts of MediaPipe's graph this module had to rebuild."""

    def test_anchor_count_matches_the_palm_networks_output(self):
        # palm_detection_full emits [1, 2016, 18]: every anchor must line up
        # with a row, or every decoded box is offset by the mismatch.
        self.assertEqual(len(openvino_hands._ANCHORS), 2016)

    def test_anchors_are_cell_centres_in_normalised_space(self):
        anchors = openvino_hands._ANCHORS
        self.assertTrue(np.all((anchors >= 0.0) & (anchors <= 1.0)))
        # The first stride-8 cell centre: (0 + 0.5) / 24.
        self.assertAlmostEqual(float(anchors[0][0]), 0.5 / 24, places=6)

    def test_rect_is_squared_on_the_long_side_and_scaled(self):
        rect = openvino_hands._rect_from_box(
            (100.0, 100.0, 200.0, 140.0), (150.0, 140.0), (150.0, 100.0),
            scale=2.0, shift_y=0.0)
        _, _, size, angle = rect
        self.assertAlmostEqual(size, 200.0)          # max(100, 40) * 2
        self.assertAlmostEqual(angle, 0.0, places=6)  # start->end already at 90 deg

    def test_the_shift_moves_along_the_rotated_axis(self):
        # Un-rotated, shift_y is straight up the image, scaled by box height.
        cx, cy, _, _ = openvino_hands._rect_from_box(
            (0.0, 0.0, 100.0, 100.0), (50.0, 100.0), (50.0, 0.0),
            scale=1.0, shift_y=-0.5)
        self.assertAlmostEqual(cx, 50.0, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)

    def test_a_crop_maps_back_to_where_it_came_from(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        rect = (160.0, 120.0, 80.0, math.radians(30.0))
        _, inverse = openvino_hands._crop(frame, rect, 224)
        centre = np.array([112.0, 112.0, 1.0])
        mapped = inverse @ centre
        self.assertAlmostEqual(float(mapped[0]), 160.0, places=3)
        self.assertAlmostEqual(float(mapped[1]), 120.0, places=3)

    def test_weighted_nms_merges_a_cluster_instead_of_picking_one(self):
        detections = [
            {"score": 0.9, "box": (0.0, 0.0, 1.0, 1.0), "kp": np.zeros((7, 2))},
            {"score": 0.1, "box": (0.0, 0.0, 1.0, 1.0), "kp": np.ones((7, 2))},
        ]
        kept = openvino_hands._nms_weighted(detections)
        self.assertEqual(len(kept), 1)
        # 0.1/(0.9+0.1) of the way toward the second detection's keypoints.
        self.assertAlmostEqual(float(kept[0]["kp"][0][0]), 0.1, places=6)

    def test_disjoint_detections_both_survive(self):
        detections = [
            {"score": 0.9, "box": (0.0, 0.0, 1.0, 1.0), "kp": np.zeros((7, 2))},
            {"score": 0.8, "box": (5.0, 5.0, 6.0, 6.0), "kp": np.ones((7, 2))},
        ]
        self.assertEqual(len(openvino_hands._nms_weighted(detections)), 2)


class TestBuildFallsBack(unittest.TestCase):

    def test_build_returns_none_when_the_preset_is_off(self):
        # Was "when nothing was asked for" — nothing asked for is "auto" now,
        # and on this machine that really does build, so the assertion had to
        # name the case it is actually about.
        with _Devices("CPU", "GPU", "NPU"):
            os.environ[accel.ENV_ACCEL] = "off"
            self.assertIsNone(openvino_hands.build())

    def test_build_returns_none_when_no_device_is_present(self):
        with _Devices():
            self.assertIsNone(openvino_hands.build())

    def test_build_returns_none_rather_than_raising_on_a_dead_device(self):
        with _Devices("CPU", "GPU", "NPU"):
            # A device OpenVINO lists but cannot compile for is the shape of a
            # driver problem; the caller has to get None, not an exception.
            real = openvino_hands.OpenVINOHands.__init__

            def explode(self, *a, **kw):
                raise openvino_hands.OpenVINOHandsUnavailable("simulated driver failure")

            openvino_hands.OpenVINOHands.__init__ = explode
            try:
                self.assertIsNone(openvino_hands.build(device="NPU"))
            finally:
                openvino_hands.OpenVINOHands.__init__ = real


if __name__ == "__main__":
    unittest.main()
