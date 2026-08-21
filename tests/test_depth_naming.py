"""The rename must not break the six games that read the old names.

"ToF" described a time-of-flight sensor this engine never had, so the keys and
attributes are now depth_*. Every old name stays as an alias until the games
move over, and these tests are what says so out loud.
"""

import importlib
import queue
import sys

import numpy as np
import pytest

from visual_ai.depth_stabilizer import DepthStabilizer, ToFStabilizer
from visual_ai.pipeline import VisionPipeline


def a_pipeline():
    return VisionPipeline(result_queue=queue.Queue(maxsize=1),
                          width=64, height=48, detect_face=False)


class TestPayloadAliases:
    def _payload(self):
        pipe = a_pipeline()
        return pipe._empty_payload(0.0, 0.0, np.zeros((48, 64, 3), np.uint8))

    def test_canonical_keys_exist(self):
        payload = self._payload()
        for key in ("depth_active", "depth_m", "depth_m_raw", "depth_source",
                    "depth_device", "depth_fps"):
            assert key in payload, key

    def test_deprecated_keys_still_exist(self):
        payload = self._payload()
        for key in ("tof_active", "tof_z_m", "tof_z_raw"):
            assert key in payload, f"{key} is still read by shipped games"

    def test_pairs_agree(self):
        payload = self._payload()
        assert payload["tof_active"] == payload["depth_active"]
        assert payload["tof_z_m"] == payload["depth_m"]
        assert payload["tof_z_raw"] == payload["depth_m_raw"]


class TestAttributeAliases:
    def test_reading_through_the_old_names(self):
        pipe = a_pipeline()
        assert pipe.tof_active is pipe.depth_active
        assert pipe.tof_simulated is pipe.depth_simulated
        assert pipe.tof_device_name == pipe.depth_device_name

    def test_writing_through_the_old_names(self):
        # flappy and labkit both assign pipeline.tof_simulated directly.
        pipe = a_pipeline()
        pipe.tof_simulated = True
        assert pipe.depth_simulated is True
        pipe.tof_active = True
        assert pipe.depth_active is True

    def test_sample_tof_depth_still_dispatches(self):
        pipe = a_pipeline()
        pipe.depth_simulated = True
        old = pipe.sample_tof_depth(32, 24, 0.0)
        new = pipe.sample_depth(32, 24, 0.0)
        assert old == new

    def test_stabilizer_alias(self):
        assert ToFStabilizer is DepthStabilizer


class TestDeprecatedModulePath:
    def test_the_old_import_path_still_resolves_and_warns(self):
        # A fresh import is what warns, so evict any earlier one.
        sys.modules.pop("visual_ai.tof_stabilizer", None)
        with pytest.warns(DeprecationWarning, match="visual_ai.depth_stabilizer"):
            shim = importlib.import_module("visual_ai.tof_stabilizer")
        assert shim.DepthStabilizer is DepthStabilizer
        assert shim.ToFStabilizer is DepthStabilizer


class TestLabelsAreHonest:
    def test_no_sensor_says_so(self):
        pipe = a_pipeline()
        _, _, label = pipe.sample_depth(32, 24)
        assert "no depth sensor" in label

    def test_simulated_is_not_called_hardware(self):
        pipe = a_pipeline()
        pipe.depth_simulated = True
        _, _, label = pipe.sample_depth(32, 24)
        assert "Simulated" in label
        assert "Hardware" not in label, "the old label claimed hardware"

    def test_a_real_sensor_is_named(self):
        pipe = a_pipeline()
        pipe.depth_active = True
        pipe.depth_device_name = "realsense"
        pipe.depth_map = np.full((48, 64), 1500, dtype=np.uint16)
        _, z_m, label = pipe.sample_depth(32, 24)
        assert label == "Depth sensor (realsense)"
        assert z_m == pytest.approx(1.5, abs=0.001)


class TestDepthIsOptional:
    def test_off_by_default(self):
        pipe = a_pipeline()
        assert pipe.depth_source_spec is None
        assert pipe.depth_active is False
        assert pipe.depth_simulated is False
        assert pipe.depth_stream is None

    def test_a_game_that_ignores_depth_still_gets_a_full_payload(self):
        pipe = a_pipeline()
        payload = pipe._empty_payload(0.0, 0.0, np.zeros((48, 64, 3), np.uint8))
        assert payload["depth_source"].startswith("RGB estimate")
        assert payload["depth_fps"] == 0.0
        assert payload["depth_device"] == ""
