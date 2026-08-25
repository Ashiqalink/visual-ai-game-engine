"""Monocular depth estimation — the backend around the network, not the network.

No weights and no onnxruntime are needed: the session is stubbed, because what
is worth testing here is everything the ONNX file does not decide. Chiefly that
this backend stays *honest* — it is an estimate wearing a sensor's return type,
and the ways that could mislead a consumer are what these tests pin down:

* it is flagged synthetic, and "auto" never selects it, so no payload can claim
  ToF hardware this machine does not have;
* it consumes the frame it was given, so a stream polling faster than the
  camera produces cannot count one estimate twice as two measurements;
* the map comes back at the network's resolution, which is the contract every
  real sensor already follows and what `sample_depth` rescales from.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from visual_ai import depth_source as ds
from visual_ai.depth_source import (
    AUTO_CANDIDATES,
    MAX_VALID_MM,
    MIN_VALID_MM,
    MonocularDepthSource,
    open_depth_source,
)


class FakeSession:
    """Stands in for an onnxruntime InferenceSession running a depth network.

    Returns a ramp in metres across the declared output, and records the last
    tensor it was fed so the preprocessing contract can be asserted on.
    """

    def __init__(self, declared_shape=(1, 3, 224, 224), out_size=(224, 224),
                 metres=(0.5, 4.0)):
        self._declared_shape = declared_shape
        self._out_size = out_size
        self._metres = metres
        self.last_tensor = None
        self.runs = 0

    def get_inputs(self):
        return [SimpleNamespace(name="rgb", shape=self._declared_shape)]

    def run(self, _outputs, feed):
        self.last_tensor = feed["rgb"]
        self.runs += 1
        height, width = self._out_size
        lo, hi = self._metres
        ramp = np.linspace(lo, hi, height * width, dtype=np.float32)
        return [ramp.reshape((1, 1, height, width))]


def make_source(session=None, unit_metres=1.0, input_size=(224, 224)):
    """A MonocularDepthSource wired to a stub, bypassing `open`."""
    source = MonocularDepthSource(model_path="stub.onnx", unit_metres=unit_metres)
    source._session = session if session is not None else FakeSession()
    source._input_name = "rgb"
    source._input_size = input_size
    source.is_open = True
    return source


def frame(width=640, height=360):
    rng = np.random.default_rng(0)
    return (rng.random((height, width, 3)) * 255).astype(np.uint8)


class TestHonesty:
    """The claims this backend is allowed to make about itself."""

    def test_marked_synthetic(self):
        # An estimate is not a measurement. If this ever flips, the pipeline
        # would start labelling inferred depth as "ToF IR Hardware".
        assert MonocularDepthSource.synthetic is True

    def test_auto_never_selects_it(self):
        assert "monocular" not in [label for label, _ in AUTO_CANDIDATES]

    def test_open_without_a_model_fails_with_a_useful_message(self, monkeypatch):
        monkeypatch.delenv(ds.DEPTH_ONNX_ENV_VAR, raising=False)
        source = MonocularDepthSource()
        assert source.open() is False
        assert ds.DEPTH_ONNX_ENV_VAR in source.last_error

    def test_open_reports_a_missing_file_rather_than_raising(self, tmp_path):
        missing = str(tmp_path / "nope.onnx")
        source = MonocularDepthSource(model_path=missing)
        assert source.open() is False
        assert "nope.onnx" in source.last_error


class TestFrameConsumption:
    def test_no_submitted_frame_yields_none(self):
        assert make_source().read() is None

    def test_frame_is_consumed_not_repeated(self):
        source = make_source()
        source.submit(frame())
        assert source.read() is not None
        # Second poll with nothing new: a repeat here would inflate `frames_read`
        # and make a stalled camera look like a live sensor.
        assert source.read() is None
        assert source.frames_read == 1

    def test_inference_does_not_run_on_submit(self):
        session = FakeSession()
        source = make_source(session)
        source.submit(frame())
        assert session.runs == 0
        source.read()
        assert session.runs == 1


class TestOutputContract:
    def test_returns_network_resolution_not_frame_resolution(self):
        source = make_source()
        source.submit(frame(width=640, height=360))
        depth = source.read()
        assert depth.shape == (224, 224)

    def test_returns_uint16_millimetres(self):
        source = make_source()
        source.submit(frame())
        depth = source.read()
        assert depth.dtype == np.uint16
        valid = depth[depth > 0]
        assert valid.size
        # The stub ramps 0.5m..4.0m, so every surviving reading must land there.
        assert valid.min() >= 500 - 1
        assert valid.max() <= 4000 + 1

    def test_out_of_range_predictions_become_no_return(self):
        # A network confidently predicting 40m indoors is wrong, and a wrong
        # distance is worse than an admitted gap.
        session = FakeSession(metres=(0.001, 40.0))
        source = make_source(session)
        source.submit(frame())
        depth = source.read()
        assert np.count_nonzero(depth == 0)
        surviving = depth[depth > 0]
        assert surviving.min() >= MIN_VALID_MM
        assert surviving.max() <= MAX_VALID_MM

    def test_unit_scale_is_applied(self):
        # An export emitting centimetres needs unit_metres=0.01, and the
        # resulting map must differ from the same export read as metres.
        as_metres = make_source(FakeSession(), unit_metres=1.0)
        as_cm = make_source(FakeSession(), unit_metres=0.01)
        for source in (as_metres, as_cm):
            source.submit(frame())
        metres_map, cm_map = as_metres.read(), as_cm.read()
        assert metres_map.max() > cm_map.max()

    def test_a_non_2d_output_is_rejected(self):
        class ThreeChannel(FakeSession):
            def run(self, _outputs, feed):
                self.last_tensor = feed["rgb"]
                return [np.zeros((1, 3, 224, 224), dtype=np.float32)]

        source = make_source(ThreeChannel())
        source.submit(frame())
        # `DepthSource.read` traps the error rather than raising at a consumer.
        assert source.read() is None
        assert "single-channel" in source.last_error


class TestPreprocess:
    def test_tensor_is_nchw_float32_at_the_declared_size(self):
        session = FakeSession()
        source = make_source(session, input_size=(224, 224))
        source.submit(frame(width=1280, height=720))
        source.read()
        assert session.last_tensor.shape == (1, 3, 224, 224)
        assert session.last_tensor.dtype == np.float32

    def test_honours_a_non_square_declared_size(self):
        session = FakeSession(out_size=(128, 160))
        source = make_source(session, input_size=(128, 160))
        source.submit(frame())
        source.read()
        assert session.last_tensor.shape == (1, 3, 128, 160)

    def test_channels_are_rgb_not_bgr(self):
        # Feeding BGR to an ImageNet-normalised encoder is silently wrong --
        # it degrades accuracy without ever erroring. A pure-blue BGR frame
        # must arrive with its energy in the *last* channel.
        session = FakeSession()
        source = make_source(session)
        blue_bgr = np.zeros((360, 640, 3), dtype=np.uint8)
        blue_bgr[..., 0] = 255
        source.submit(blue_bgr)
        source.read()
        per_channel = session.last_tensor[0].mean(axis=(1, 2))
        assert per_channel[2] > per_channel[0]

    def test_matches_the_naive_normalisation(self):
        # The fused cv2 form must agree with the textbook expression it
        # replaced; a broadcasting slip here would be invisible otherwise.
        import cv2

        source = make_source()
        img = frame()
        tensor = source._preprocess(img)

        scaled = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        naive = (scaled.astype(np.float32) / 255.0 - ds._IMAGENET_MEAN) / ds._IMAGENET_STD
        naive = np.transpose(naive, (2, 0, 1))[None]
        assert np.allclose(tensor, naive, atol=1e-5)


class TestSpecParsing:
    def test_spec_preserves_path_case(self, tmp_path, monkeypatch):
        """`open_depth_source` lowercases the spec; the path must survive it."""
        built = {}

        class Recorder(MonocularDepthSource):
            def __init__(self, model_path=None, **kw):
                super().__init__(model_path=model_path, **kw)
                built["path"] = model_path

            def _open(self):
                return None

        monkeypatch.setattr(ds, "MonocularDepthSource", Recorder)
        mixed = str(tmp_path / "FastDepth_NYU.onnx")
        open_depth_source(f"monocular:{mixed}")
        assert built["path"] == mixed

    def test_replay_path_case_also_survives(self, monkeypatch):
        seen = {}

        class Recorder(ds.ReplayDepthSource):
            def __init__(self, path):
                seen["path"] = path
                super().__init__(path)

            def _open(self):
                raise ds.DepthSourceError("not a real recording")

        monkeypatch.setattr(ds, "ReplayDepthSource", Recorder)
        open_depth_source(r"replay:C:\Recordings\HandWave.npz")
        assert seen["path"] == r"C:\Recordings\HandWave.npz"

    def test_env_var_supplies_the_model(self, tmp_path, monkeypatch):
        path = tmp_path / "m.onnx"
        path.write_bytes(b"not really onnx")
        monkeypatch.setenv(ds.DEPTH_ONNX_ENV_VAR, str(path))
        assert MonocularDepthSource().model_path == str(path)

    def test_unknown_spec_still_raises(self):
        with pytest.raises(ValueError):
            open_depth_source("monoculr:x.onnx")
