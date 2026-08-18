"""Tests for MODNet portrait matting — no ONNX weights or onnxruntime needed.

The network itself is stubbed out and no test touches the network: what is
worth testing here is everything around it — the input sizing rule (get it
wrong and onnxruntime fails with an opaque rank error), the matte's shape and
range contract, and the download-and-verify path users hit before anything
else works.
"""

import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

from visual_ai import matting


class FakeSession:
    """Stands in for an onnxruntime InferenceSession.

    Records the tensor it was fed and returns a matte of the matching size,
    half opaque and half transparent.
    """

    def __init__(self, declared_shape=("N", 3, "H", "W")):
        self._declared_shape = declared_shape
        self.last_tensor = None

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=self._declared_shape)]

    def run(self, _outputs, feed):
        tensor = feed["input"]
        self.last_tensor = tensor
        _, _, h, w = tensor.shape
        matte = np.zeros((1, 1, h, w), dtype=np.float32)
        matte[..., : h // 2, :] = 1.0
        return [matte]


def make_matter(session):
    matter = object.__new__(matting.PortraitMatter)
    matter._session = session
    matter._input_name = "input"
    matter._fixed_size = matter._declared_size()
    return matter


@pytest.mark.parametrize("size, expected", [
    ((100, 200), (512, 1024)),     # smaller than ref: shorter side up to 512
    ((1080, 1920), (512, 896)),    # larger than ref: shorter side down to 512
    ((512, 512), (512, 512)),      # already at ref: passed through untouched
    ((480, 640), (480, 640)),      # straddles ref: left alone, already on-stride
    ((600, 512), (576, 512)),      # straddles ref: only truncated to a multiple of 32
    ((20, 2000), (32, 1984)),      # never smaller than one stride
])
def test_inference_size(size, expected):
    assert matting._inference_size(*size) == expected


def test_inference_size_is_always_a_multiple_of_the_stride():
    for h in range(64, 1400, 37):
        for w in range(64, 1400, 53):
            new_h, new_w = matting._inference_size(h, w)
            assert new_h % 32 == 0 and new_w % 32 == 0


def test_preprocess_normalizes_to_unit_range():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[:32] = 255
    tensor = matting._preprocess(rgb, (64, 64))

    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == np.float32
    assert tensor.max() == pytest.approx(1.0)
    assert tensor.min() == pytest.approx(-1.0)


def test_matte_returns_input_resolution_in_unit_range():
    session = FakeSession()
    matte = make_matter(session).matte(np.zeros((480, 640, 3), dtype=np.uint8))

    assert matte.shape == (480, 640)
    assert matte.dtype == np.float32
    assert 0.0 <= matte.min() and matte.max() <= 1.0
    # The stub mattes the top half; resizing back must not move that boundary.
    assert matte[:200].mean() > 0.9
    assert matte[280:].mean() < 0.1


def test_fixed_shape_export_is_fed_its_declared_size():
    session = FakeSession(declared_shape=(1, 3, 512, 512))
    matter = make_matter(session)
    matter.matte(np.zeros((480, 640, 3), dtype=np.uint8))

    assert session.last_tensor.shape == (1, 3, 512, 512)


def test_dynamic_shape_export_is_fed_the_computed_size():
    session = FakeSession()
    matter = make_matter(session)
    matter.matte(np.zeros((480, 640, 3), dtype=np.uint8))

    assert session.last_tensor.shape == (1, 3, 480, 640)


def test_cut_out_carries_the_matte_as_alpha():
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    rgba = make_matter(FakeSession()).cut_out(image)

    assert rgba.shape == (480, 640, 4)
    assert rgba[:200, :, 3].mean() > 230
    assert rgba[280:, :, 3].mean() < 25


def test_a_non_matte_output_is_reported_rather_than_reshaped():
    class ThreeChannelSession(FakeSession):
        def run(self, _outputs, feed):
            _, _, h, w = feed["input"].shape
            return [np.zeros((1, 3, h, w), dtype=np.float32)]

    with pytest.raises(RuntimeError, match="really MODNet"):
        make_matter(ThreeChannelSession()).matte(np.zeros((64, 64, 3), dtype=np.uint8))


def test_model_path_honours_the_env_var(tmp_path, monkeypatch):
    weights = tmp_path / "modnet.onnx"
    weights.write_bytes(b"not really a model")
    monkeypatch.setenv(matting.MODNET_ENV_VAR, str(weights))

    assert matting.model_path() == str(weights)


def test_model_path_rejects_an_env_var_pointing_nowhere(tmp_path, monkeypatch):
    monkeypatch.setenv(matting.MODNET_ENV_VAR, str(tmp_path / "missing.onnx"))

    with pytest.raises(RuntimeError, match="no file exists"):
        matting.model_path()


def stub_download(monkeypatch, payload=b"weights", fail=None):
    """Replace urlretrieve, returning the list that records the URLs asked for."""
    calls = []

    def fake_urlretrieve(url, destination):
        calls.append(url)
        with open(destination, "wb") as handle:
            handle.write(payload)
        if fail is not None:
            raise fail

    monkeypatch.setattr(matting.urllib.request, "urlretrieve", fake_urlretrieve)
    return calls


def test_model_path_downloads_the_pinned_weights_once(tmp_path, monkeypatch):
    monkeypatch.delenv(matting.MODNET_ENV_VAR, raising=False)
    monkeypatch.delenv(matting.MODNET_URL_ENV_VAR, raising=False)
    monkeypatch.setattr(matting, "_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(matting, "_MODEL_SHA256", hashlib.sha256(b"weights").hexdigest())
    calls = stub_download(monkeypatch)

    first = matting.model_path()
    second = matting.model_path()

    assert first == second == os.path.join(
        str(tmp_path), "modnet_photographic_portrait_matting.onnx")
    assert calls == [matting._MODEL_URL]  # second call was served from the cache
    assert not os.path.isfile(first + ".part")


def test_the_pinned_url_carries_a_revision_rather_than_a_branch():
    # A mirror tracking a branch can change what it serves; the checksum below
    # only means anything if the URL cannot move under it.
    assert "/resolve/main/" not in matting._MODEL_URL
    assert len(matting._MODEL_SHA256) == 64


def test_weights_that_fail_the_checksum_are_not_kept(tmp_path, monkeypatch):
    monkeypatch.delenv(matting.MODNET_ENV_VAR, raising=False)
    monkeypatch.delenv(matting.MODNET_URL_ENV_VAR, raising=False)
    monkeypatch.setattr(matting, "_cache_dir", lambda: str(tmp_path))
    stub_download(monkeypatch, payload=b"something else entirely")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        matting.model_path()
    assert os.listdir(tmp_path) == []


def test_a_custom_url_is_used_and_not_checksummed(tmp_path, monkeypatch):
    monkeypatch.delenv(matting.MODNET_ENV_VAR, raising=False)
    monkeypatch.setenv(matting.MODNET_URL_ENV_VAR, "https://example.invalid/mine.onnx")
    monkeypatch.setattr(matting, "_cache_dir", lambda: str(tmp_path))
    calls = stub_download(monkeypatch, payload=b"a different export")

    path = matting.model_path()

    assert calls == ["https://example.invalid/mine.onnx"]
    assert os.path.isfile(path)


def test_a_failed_download_leaves_no_partial_file(tmp_path, monkeypatch):
    monkeypatch.delenv(matting.MODNET_ENV_VAR, raising=False)
    monkeypatch.delenv(matting.MODNET_URL_ENV_VAR, raising=False)
    monkeypatch.setattr(matting, "_cache_dir", lambda: str(tmp_path))
    stub_download(monkeypatch, payload=b"half", fail=OSError("connection reset"))

    with pytest.raises(RuntimeError, match="could not download"):
        matting.model_path()
    assert os.listdir(tmp_path) == []
