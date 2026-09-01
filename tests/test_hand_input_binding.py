"""
The hand networks are fed through buffers nothing copies. Do they see the crop?

`openvino_hands` no longer builds an input array per inference. Each request
owns a uint8 buffer, bound once with `set_input_tensor`, that `cv2.warpAffine`
and `_letterbox` write into directly; `start_async()` then takes no argument.
Two links in that chain can break without raising:

  * OpenCV's `dst=` is a request, not a promise. Given a destination of the
    wrong size or dtype `warpAffine` allocates a new one and returns it, and
    the caller that ignored the return value goes on submitting a buffer that
    still holds the previous frame.
  * `ov.Tensor(array)` copies. Sharing the array's memory is `shared_memory=
    True`, and without it the device reads whatever the buffer held at bind
    time for the life of the request -- at exactly the same speed, which is
    how it survived a timing A/B. This suite is what caught it.

Neither shows up as an error. Both show up as a hand that does not move, which
is why the checks below compare *outputs across two different inputs* rather
than checking that a call succeeded. `test_the_fold_is_the_same_arithmetic`
covers the third link, the `uint8 -> float32 / 255` now compiled into the model.

The device checks need a real accelerator and the mediapipe weights, and skip
without them. Run from the engine root:

    python -m pytest tests/test_hand_input_binding.py
"""

from __future__ import annotations

import numpy as np
import pytest

from visual_ai import openvino_hands
from visual_ai.openvino_hands import (_LAND_SIZE, _PALM_SIZE, _crop,
                                      _letterbox, _model_paths, _u8_input)

WIDTH, HEIGHT = 640, 480


def _noise(seed, height, width):
    return np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8)


# -- the buffers OpenCV is asked to write into --------------------------------

def test_crop_writes_into_the_buffer_it_is_given():
    """`warpAffine(dst=...)` filling its own array instead is the failure."""
    buffer = np.zeros((1, _LAND_SIZE, _LAND_SIZE, 3), np.uint8)
    rect = (WIDTH / 2.0, HEIGHT / 2.0, 200.0, 0.0)
    crop, _ = _crop(_noise(1, HEIGHT, WIDTH), rect, _LAND_SIZE, out=buffer[0])

    assert buffer[0].any(), "the crop went somewhere else"
    assert np.array_equal(buffer[0], crop)


def test_a_second_crop_replaces_the_first():
    """A buffer that keeps the first hand is a hand that never moves."""
    buffer = np.zeros((1, _LAND_SIZE, _LAND_SIZE, 3), np.uint8)
    frame = _noise(2, HEIGHT, WIDTH)
    _crop(frame, (200.0, 200.0, 160.0, 0.0), _LAND_SIZE, out=buffer[0])
    first = buffer[0].copy()
    _crop(frame, (400.0, 300.0, 160.0, 0.4), _LAND_SIZE, out=buffer[0])

    assert not np.array_equal(first, buffer[0])


def test_the_letterbox_clears_the_bars_it_does_not_redraw():
    """
    Only the fitted image is rewritten; the padding is last frame's picture.

    A 640x480 frame leaves padding rows top and bottom. Feed a square frame
    first so those rows hold something, then a wide one, and the bars have to
    come back black.
    """
    buffer = np.zeros((1, _PALM_SIZE, _PALM_SIZE, 3), np.uint8)
    _letterbox(np.full((480, 480, 3), 255, np.uint8), _PALM_SIZE, out=buffer[0])
    assert buffer[0, 0].any(), "the square frame should have filled the top row"

    _, _, _, top = _letterbox(_noise(3, HEIGHT, WIDTH), _PALM_SIZE, out=buffer[0])
    assert top > 0
    assert not buffer[0, :top].any(), "the letterbox bars kept the old frame"


# -- the tensor the device reads, and the fold compiled into the model --------

def _land_request(device):
    """A landmark request on `device` with a bound uint8 buffer, or a skip."""
    ov = pytest.importorskip("openvino")
    try:
        _, land_path = _model_paths("full")
    except openvino_hands.OpenVINOHandsUnavailable as exc:
        pytest.skip(str(exc))
    core = ov.Core()
    if device not in core.available_devices:
        pytest.skip(f"no {device} on this machine")
    model = _u8_input(core, core.read_model(land_path), ov)
    request = core.compile_model(
        model, device, {"PERFORMANCE_HINT": "LATENCY"}).create_infer_request()
    buffer = np.zeros((1, _LAND_SIZE, _LAND_SIZE, 3), np.uint8)
    request.set_input_tensor(ov.Tensor(buffer, shared_memory=True))
    return core, land_path, request, buffer


def _first_output(request):
    return np.asarray(next(iter(request.results.values()))).copy()


@pytest.mark.parametrize("device", ["NPU", "GPU"])
def test_the_device_reads_the_bound_buffer_as_it_is_now(device):
    """Writing into the buffer after binding must change what comes back."""
    _, _, request, buffer = _land_request(device)

    buffer[0] = _noise(4, _LAND_SIZE, _LAND_SIZE)
    request.infer()
    first = _first_output(request)
    buffer[0] = _noise(5, _LAND_SIZE, _LAND_SIZE)
    request.infer()
    second = _first_output(request)

    assert not np.array_equal(first, second), (
        "the same output for two different crops - the tensor is a copy taken "
        "at bind time, not a view of the buffer")


@pytest.mark.parametrize("device", ["NPU", "GPU"])
def test_the_fold_is_the_same_arithmetic(device):
    """
    The folded model on uint8 must agree with the plain model on float32/255.

    Not bit-for-bit: the divide happens in the device's precision now instead
    of numpy's float32. Repeated inference on either model is bit-identical, so
    the tolerance below is the whole of the difference the fold introduces --
    measured at 0.3 where these logits reach 180.
    """
    core, land_path, folded, buffer = _land_request(device)
    plain = core.compile_model(
        core.read_model(land_path), device,
        {"PERFORMANCE_HINT": "LATENCY"}).create_infer_request()

    image = _noise(6, _LAND_SIZE, _LAND_SIZE)
    buffer[0] = image
    folded.infer()
    plain.infer([(image.astype(np.float32) / 255.0)[None]])

    by_name = {port.get_any_name(): np.asarray(value)
               for port, value in plain.results.items()}
    for port, value in folded.results.items():
        reference = by_name[port.get_any_name()]
        got = np.asarray(value)
        scale = max(1.0, float(np.abs(reference).max()))
        worst = float(np.abs(got - reference).max())
        assert worst <= 0.01 * scale, (
            f"{port.get_any_name()} moved by {worst:.3f} on a {scale:.1f} scale")


@pytest.mark.parametrize("device", ["NPU", "GPU"])
def test_two_hands_run_on_two_requests_at_once(device):
    """
    The overlap is only real if the two slots keep their own results.

    One compiled model, two requests, both in flight: if they shared an output
    the second hand's landmarks would be the first hand's, which is the same
    on-screen symptom as the duplicate-hand bug and not one a timing number
    would ever show.
    """
    ov = pytest.importorskip("openvino")
    core, land_path, _, _ = _land_request(device)
    compiled = core.compile_model(
        _u8_input(core, core.read_model(land_path), ov), device,
        {"PERFORMANCE_HINT": "LATENCY"})
    requests = []
    for seed in (7, 8):
        request = compiled.create_infer_request()
        buffer = np.zeros((1, _LAND_SIZE, _LAND_SIZE, 3), np.uint8)
        buffer[0] = _noise(seed, _LAND_SIZE, _LAND_SIZE)
        request.set_input_tensor(ov.Tensor(buffer, shared_memory=True))
        requests.append(request)

    for request in requests:
        request.start_async()
    for request in requests:
        request.wait()

    assert not np.array_equal(_first_output(requests[0]),
                              _first_output(requests[1]))
