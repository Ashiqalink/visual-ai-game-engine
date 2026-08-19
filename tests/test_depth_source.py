"""Depth producers, and the pipeline's consumption of what they produce.

The hardware backends cannot be exercised here -- this machine has one RGB
webcam -- so what is tested is everything around them: the frame contract they
must satisfy, the sanitiser every backend funnels through, the streaming and
staleness behaviour, and the consumer side of the pipeline reading a real
depth map instead of the landmark estimate it used to fall back on.
"""

import os
import time
import queue

import numpy as np
import pytest

from visual_ai.depth_source import (
    DepthRecorder,
    DepthStream,
    OpenNI2DepthSource,
    RealSenseDepthSource,
    ReplayDepthSource,
    SyntheticDepthSource,
    UVCDepthSource,
    MAX_VALID_MM,
    MIN_VALID_MM,
    open_depth_source,
    probe_depth_sources,
    sanitize,
)
from visual_ai.pipeline import VisionPipeline


class TestSanitize:
    def test_uint16_passes_through(self):
        frame = np.full((4, 4), 1500, dtype=np.uint16)
        assert sanitize(frame).dtype == np.uint16
        assert sanitize(frame)[0, 0] == 1500

    def test_float_metres_are_converted_to_mm(self):
        frame = np.full((4, 4), 1.25, dtype=np.float32)
        assert sanitize(frame)[0, 0] == 1250

    def test_float_already_in_mm_is_left_scaled(self):
        # Above 100 the values cannot be metres -- 100 m is beyond any sensor.
        frame = np.full((4, 4), 1250.0, dtype=np.float32)
        assert sanitize(frame)[0, 0] == 1250

    def test_int32_mm_is_narrowed(self):
        frame = np.full((4, 4), 900, dtype=np.int32)
        out = sanitize(frame)
        assert out.dtype == np.uint16 and out[0, 0] == 900

    def test_colour_frames_are_rejected(self):
        # An ordinary webcam opened as a depth device looks exactly like this,
        # and must not be mistaken for a sensor.
        assert sanitize(np.zeros((8, 8, 3), dtype=np.uint8)) is None

    def test_out_of_range_becomes_no_return(self):
        frame = np.array([[50, 1500, 60000]], dtype=np.uint16)
        out = sanitize(frame)
        assert out[0, 0] == 0, "too close is a bad reading, not a distance"
        assert out[0, 1] == 1500
        assert out[0, 2] == 0
        assert MIN_VALID_MM <= 1500 <= MAX_VALID_MM

    def test_nans_do_not_survive(self):
        frame = np.full((4, 4), np.nan, dtype=np.float32)
        assert int(sanitize(frame).max()) == 0

    def test_rejects_empty_and_none(self):
        assert sanitize(None) is None
        assert sanitize(np.zeros((0, 0), dtype=np.uint16)) is None

    def test_does_not_mutate_its_input(self):
        frame = np.array([[50, 1500]], dtype=np.uint16)
        sanitize(frame)
        assert frame[0, 0] == 50


class TestSyntheticSource:
    def test_produces_valid_depth(self):
        src = SyntheticDepthSource(width=64, height=48)
        assert src.open()
        frame = src.read()
        assert frame.dtype == np.uint16
        assert frame.shape == (48, 64)
        assert src.resolution == (64, 48)
        src.close()

    def test_hand_is_nearer_than_the_wall(self):
        src = SyntheticDepthSource(width=64, height=48)
        src.open()
        frame = src.read()
        valid = frame[frame > 0]
        assert valid.min() < 1000 < valid.max()

    def test_it_has_dropouts_like_a_real_sensor(self):
        src = SyntheticDepthSource(width=64, height=48)
        src.open()
        frame = src.read()
        assert np.count_nonzero(frame == 0) > 0

    def test_deterministic(self):
        a = SyntheticDepthSource(width=32, height=24)
        b = SyntheticDepthSource(width=32, height=24)
        a.open(); b.open()
        for _ in range(5):
            assert np.array_equal(a.read(), b.read())

    def test_the_blob_moves(self):
        src = SyntheticDepthSource(width=64, height=48)
        src.open()
        first = src.read()
        for _ in range(8):
            last = src.read()
        assert not np.array_equal(first, last)

    def test_is_flagged_synthetic(self):
        assert SyntheticDepthSource().synthetic is True


class TestReplay:
    def test_round_trip(self, tmp_path):
        rec = DepthRecorder()
        src = SyntheticDepthSource(width=32, height=24)
        src.open()
        for _ in range(6):
            rec.feed(src.read())
        path = rec.save(str(tmp_path / "rec.npz"))

        replay = ReplayDepthSource(path, loop=False)
        assert replay.open()
        frames = [replay.read() for _ in range(6)]
        assert all(f is not None for f in frames)
        assert frames[0].shape == (24, 32)
        assert replay.read() is None, "a non-looping replay should run out"

    def test_looping(self, tmp_path):
        rec = DepthRecorder()
        src = SyntheticDepthSource(width=16, height=12)
        src.open()
        for _ in range(3):
            rec.feed(src.read())
        path = rec.save(str(tmp_path / "loop.npz"))

        replay = ReplayDepthSource(path, loop=True)
        replay.open()
        first = replay.read()
        for _ in range(2):
            replay.read()
        assert np.array_equal(replay.read(), first)

    def test_missing_file_is_reported_not_raised(self):
        replay = ReplayDepthSource("no-such-file.npz")
        assert replay.open() is False
        assert "no such recording" in replay.last_error

    def test_recorder_is_bounded(self):
        rec = DepthRecorder(max_frames=3)
        frame = np.ones((4, 4), dtype=np.uint16) * 500
        assert [rec.feed(frame) for _ in range(5)] == [True, True, True, False, False]
        assert rec.full

    def test_saving_nothing_raises(self, tmp_path):
        with pytest.raises(Exception):
            DepthRecorder().save(str(tmp_path / "empty.npz"))


class TestStream:
    def test_delivers_frames(self):
        stream = DepthStream(SyntheticDepthSource(width=32, height=24))
        stream.start()
        deadline = time.perf_counter() + 2.0
        while stream.latest() is None and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert stream.latest() is not None
        assert stream.age_s() < 1.0
        stream.stop()

    def test_age_is_infinite_before_the_first_frame(self):
        stream = DepthStream(SyntheticDepthSource())
        assert stream.age_s() == float("inf")

    def test_keeps_only_the_newest(self):
        stream = DepthStream(SyntheticDepthSource(width=32, height=24))
        stream.start()
        time.sleep(0.15)
        a = stream.latest()
        time.sleep(0.15)
        b = stream.latest()
        stream.stop()
        assert a is not None and b is not None
        assert not np.array_equal(a, b), "a live stream should keep moving"

    def test_stop_closes_the_source(self):
        src = SyntheticDepthSource()
        stream = DepthStream(src)
        stream.start()
        time.sleep(0.1)
        stream.stop()
        assert src.is_open is False

    def test_recording_through_a_stream(self):
        stream = DepthStream(SyntheticDepthSource(width=16, height=12))
        stream.recorder = DepthRecorder(max_frames=5)
        stream.start()
        deadline = time.perf_counter() + 2.0
        while not stream.recorder.full and time.perf_counter() < deadline:
            time.sleep(0.01)
        stream.stop()
        assert len(stream.recorder.frames) == 5


class TestFactory:
    def test_none_and_off(self):
        assert open_depth_source(None) is None
        assert open_depth_source("none") is None
        assert open_depth_source("") is None

    def test_synthetic(self):
        src = open_depth_source("synthetic")
        assert isinstance(src, SyntheticDepthSource) and src.is_open
        src.close()

    def test_replay_spec(self, tmp_path):
        rec = DepthRecorder()
        src = SyntheticDepthSource(width=16, height=12)
        src.open()
        rec.feed(src.read())
        path = rec.save(str(tmp_path / "spec.npz"))
        got = open_depth_source(f"replay:{path}")
        assert isinstance(got, ReplayDepthSource)
        got.close()

    def test_unknown_spec_raises(self):
        with pytest.raises(ValueError):
            open_depth_source("lidar")

    def test_auto_never_falls_back_to_synthetic(self):
        # With no hardware attached "auto" must return None. Quietly handing
        # back a synthetic source would make a game claim a sensor it has not
        # got, which is the failure this module exists to end.
        got = open_depth_source("auto")
        if got is not None:            # a real sensor is attached to this box
            assert got.synthetic is False
            got.close()

    def test_hardware_backends_report_rather_than_raise(self):
        for src in (OpenNI2DepthSource(), UVCDepthSource(index=99),
                    RealSenseDepthSource()):
            assert src.open() in (True, False)
            if not src.is_open:
                assert src.last_error, "a failed open must say why"
            src.close()

    def test_probe_returns_a_row_per_backend(self):
        rows = probe_depth_sources()
        assert len(rows) >= 3
        for name, ok, detail in rows:
            assert isinstance(name, str) and isinstance(ok, bool)
            assert detail, "every row should carry a reason or a description"


class TestPipelineConsumesDepth:
    """The consumer half: a real depth map must beat the landmark estimate."""

    def _pipeline(self, **kw):
        return VisionPipeline(result_queue=queue.Queue(maxsize=1),
                              width=64, height=48, detect_face=False, **kw)

    def test_defaults_to_no_depth_source(self):
        pipe = self._pipeline()
        assert pipe.depth_stream is None
        assert pipe.tof_active is False

    def test_sampling_falls_back_without_a_sensor(self):
        pipe = self._pipeline()
        pipe.tof_simulated = True
        active, z_m, label = pipe.sample_tof_depth(32, 24, lm_z=0.0)
        assert active is True
        assert label == "ToF Hardware (Simulated)"
        assert z_m == pytest.approx(0.45, abs=0.01)

    def test_sampling_reads_the_depth_map_when_there_is_one(self):
        pipe = self._pipeline()
        pipe.tof_active = True
        pipe.depth_map = np.full((48, 64), 1800, dtype=np.uint16)
        active, z_m, label = pipe.sample_tof_depth(32, 24)
        assert active is True
        assert z_m == pytest.approx(1.8, abs=0.001)
        assert label == "ToF IR Hardware"

    def test_depth_map_resolution_need_not_match_the_frame(self):
        # The sensor is 320x240 while the frame is 64x48: the fingertip pixel
        # has to be rescaled, or the sample lands somewhere else entirely.
        pipe = self._pipeline()
        pipe.tof_active = True
        depth = np.full((240, 320), 3000, dtype=np.uint16)
        depth[100:140, 140:180] = 700           # a near patch, mid-sensor
        pipe.depth_map = depth
        _, near, _ = pipe.sample_tof_depth(32, 24)     # frame centre
        _, far, _ = pipe.sample_tof_depth(2, 2)        # frame corner
        assert near == pytest.approx(0.7, abs=0.05)
        assert far == pytest.approx(3.0, abs=0.05)

    def test_no_return_pixels_are_skipped(self):
        # Zeros are holes, not "0 m". Averaging them in would drag a fingertip
        # reading toward the camera exactly at the edges where holes cluster.
        pipe = self._pipeline()
        pipe.tof_active = True
        depth = np.zeros((48, 64), dtype=np.uint16)
        depth[24, 32] = 1200
        pipe.depth_map = depth
        _, z_m, _ = pipe.sample_tof_depth(32, 24)
        assert z_m == pytest.approx(1.2, abs=0.001)

    def test_all_holes_falls_back_rather_than_reporting_zero(self):
        pipe = self._pipeline()
        pipe.tof_active = True
        pipe.depth_map = np.zeros((48, 64), dtype=np.uint16)
        _, z_m, _ = pipe.sample_tof_depth(32, 24, lm_z=0.0)
        assert z_m > 0.15, "a hole must not read as touching the lens"

    def test_depth_grid_uses_the_sensor_when_present(self):
        pipe = self._pipeline()
        pipe.emit_depth_grid = True
        pipe.depth_map = np.full((48, 64), 2500, dtype=np.uint16)
        grid = pipe._build_depth_grid([], None)
        assert grid.shape == (pipe.depth_grid_size[1], pipe.depth_grid_size[0])
        assert float(grid.mean()) == pytest.approx(2.5, abs=0.01)

    def test_synthetic_source_does_not_claim_hardware(self):
        pipe = self._pipeline(depth_source="synthetic")
        assert pipe.depth_source_spec == "synthetic"
        # Opening happens in the thread; the label contract is what matters.
        assert pipe.tof_active is False
