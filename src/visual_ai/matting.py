"""
matting.py — MODNet portrait matting: a person out of a photo, with soft edges.

:class:`PortraitMatter` runs MODNet (a trimap-free portrait matting network)
through ``onnxruntime`` and returns a *soft alpha matte* — a float in [0, 1]
per pixel — rather than the binary in/out decision a segmenter gives you. That
difference is the whole point: hair, motion blur and glasses frames are
genuinely partly transparent, and a category mask has no way to say so.

How this relates to the two cuts already in the SDK:

* :class:`visual_ai.segment.PersonSegmenter` — MediaPipe Selfie Segmenter,
  fast enough for every frame of a live pipeline, but a hard category mask.
  Use it for live compositing.
* :func:`visual_ai.imaging.remove_background` — ``rembg``, general-purpose and
  trained on objects, not people. Fine for artwork, softer than it should be
  on a face and worse on hair.
* This module — the best portrait edge of the three, and by far the slowest
  (hundreds of milliseconds on CPU). Use it for one-shot captures: a player
  photographed once at the start of a session and reused as a sprite.

Weights are not bundled — they are downloaded once, on first use, into the
same per-user cache :mod:`visual_ai.segment` uses, the same "pay only if you
use it" shape ``rembg`` has. MODNet's authors publish the PyTorch checkpoint
but no first-party ONNX build, so the pinned source is a community mirror at a
fixed revision, and the download is checked against a known SHA-256 rather
than trusted on name alone. ``VISUAL_AI_MODNET_ONNX`` points at a local file
instead; ``VISUAL_AI_MODNET_URL`` swaps the download source (and skips the
digest check, since the expected digest belongs to the pinned file).

Channel order follows :mod:`visual_ai.imaging`: three-channel input is assumed
**RGB**. Pipeline frames are BGR — pass them through
:func:`visual_ai.imaging.bgr_to_rgb` first.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import urllib.request

import numpy as np

from visual_ai import accel
from visual_ai.imaging import to_rgba

__all__ = [
    "PortraitMatter",
    "cut_out_person",
    "matting_device",
    "model_path",
    "MODNET_AVAILABLE",
    "MODNET_ENV_VAR",
    "MODNET_URL_ENV_VAR",
]

MODNET_ENV_VAR = "VISUAL_AI_MODNET_ONNX"
MODNET_URL_ENV_VAR = "VISUAL_AI_MODNET_URL"
_MODEL_FILENAME = "modnet_photographic_portrait_matting.onnx"

# The official MODNet repo ships a PyTorch checkpoint and an export script, not
# an ONNX build, so this is a community mirror — pinned to a revision hash
# rather than a branch, and verified against the digest below, because a
# third-party mirror is exactly the kind of URL whose contents can change
# under you. The file is the photographic (not webcam) portrait model, 25 MB.
_MODEL_URL = (
    "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/"
    "903cc06311b3b12071edfa1b42b534e4a31c718a/models/"
    "modnet_photographic_portrait_matting.onnx"
)
_MODEL_SHA256 = "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9"

# MODNet's reference input size. The network is fully convolutional and
# downsamples by 32x internally, so both input dimensions must be multiples of
# 32; 512 is what the model was trained around and what the reference demo
# uses. Larger inputs do not buy detail, they just cost time.
REF_SIZE = 512
_STRIDE = 32


# find_spec rather than a trial import: this module is imported eagerly by the
# package, and actually importing onnxruntime costs a second or so of load.
MODNET_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None


def _cache_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, ".visual_ai", "models")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: str, expect_sha256: str | None) -> None:
    """
    Fetch ``url`` to ``path`` via a .part file, verifying the digest first.

    Downloading straight to the final path would leave a truncated file behind
    on a dropped connection, and every later run would load that corpse and
    fail somewhere far less obvious than here.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        if expect_sha256 is not None:
            actual = _sha256(tmp_path)
            if actual != expect_sha256:
                raise RuntimeError(
                    f"checksum mismatch: expected {expect_sha256}, got {actual}. "
                    "The pinned weights have changed or the download was corrupted; "
                    f"fetch the file yourself and point {MODNET_ENV_VAR} at it.")
        os.replace(tmp_path, path)
    except Exception as exc:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"could not download MODNet weights from {url}: {exc}\n"
            "Portrait matting needs network access on first use; retry once online, "
            f"or set {MODNET_ENV_VAR} to a copy of the weights you already have."
        ) from exc


def model_path() -> str:
    """
    Local path to the MODNet ONNX weights, downloading them if needed.

    Checked in order: ``VISUAL_AI_MODNET_ONNX``, then the per-user cache
    directory shared with :mod:`visual_ai.segment`, then a one-time download.
    The download is verified against a pinned SHA-256 unless
    ``VISUAL_AI_MODNET_URL`` redirected it somewhere else, in which case the
    expected digest no longer applies and the caller owns what they pointed at.
    """
    override = os.environ.get(MODNET_ENV_VAR)
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(
                f"{MODNET_ENV_VAR} is set to {override!r} but no file exists there.")
        return override

    path = os.path.join(_cache_dir(), _MODEL_FILENAME)
    if os.path.isfile(path):
        return path

    url = os.environ.get(MODNET_URL_ENV_VAR)
    _download(url or _MODEL_URL, path, None if url else _MODEL_SHA256)
    return path


def _inference_size(height: int, width: int) -> tuple[int, int]:
    """
    Pick the (height, width) MODNet should actually see.

    Follows the reference implementation: the image is rescaled so its
    *shorter* side is ``REF_SIZE`` — but only when it is meaningfully off that
    scale already, so a frame that is roughly the right size is fed through
    untouched rather than resampled for nothing. Both dimensions are then
    truncated to a multiple of 32, which is the network's downsample factor.
    """
    if max(height, width) < REF_SIZE or min(height, width) > REF_SIZE:
        if width >= height:
            new_h = REF_SIZE
            new_w = int(width / height * REF_SIZE)
        else:
            new_w = REF_SIZE
            new_h = int(height / width * REF_SIZE)
    else:
        new_h, new_w = height, width

    new_h = max(_STRIDE, new_h - new_h % _STRIDE)
    new_w = max(_STRIDE, new_w - new_w % _STRIDE)
    return new_h, new_w


def _preprocess(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """RGB uint8 -> NCHW float32 tensor normalised to [-1, 1] at ``size``."""
    import cv2

    new_h, new_w = size
    scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    normalized = (scaled.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.transpose(normalized, (2, 0, 1))[None]


class _OpenVINOSession:
    """
    The three pieces of the ``onnxruntime`` session API :class:`PortraitMatter`
    uses, backed by OpenVINO so MODNet can run on the iGPU.

    Worth it: MODNet at 640x352 measures 48.4 ms on this CPU (and eleven cores
    of it) against 11.9 ms on an Arc 130T. The NPU is not the right home for
    this one — same speed as the CPU there, and its FP16 weights move the alpha
    matte by up to 0.13.

    Compilation is per input shape and cached, rather than compiling the model
    dynamic once, for two reasons: a dynamic compile re-plans on every new
    shape anyway, and the NPU plugin does not merely refuse a dynamic model —
    it takes the process down with a segfault that no ``except`` can catch.
    """

    class _Input:
        """Stands in for an onnxruntime NodeArg: a name and a shape."""

        def __init__(self, name: str, shape: list):
            self.name, self.shape = name, shape

    def __init__(self, path: str, device: str):
        import openvino as ov

        self._core = ov.Core()
        self._path = path
        self._model = self._core.read_model(path)
        self.device = device
        self._compiled: dict[tuple[int, int], object] = {}

    def get_inputs(self):
        source = self._model.inputs[0]
        shape = [dimension.get_length() if dimension.is_static else "?"
                 for dimension in source.get_partial_shape()]
        return [self._Input(source.get_any_name(), shape)]

    def _compiled_for(self, height: int, width: int):
        key = (height, width)
        if key not in self._compiled:
            import openvino as ov

            model = self._core.read_model(self._path)
            model.reshape({model.inputs[0]: ov.PartialShape([1, 3, height, width])})
            self._compiled[key] = self._core.compile_model(
                model, self.device, {"PERFORMANCE_HINT": "LATENCY"})
        return self._compiled[key]

    def run(self, _output_names, feed: dict):
        tensor = next(iter(feed.values()))
        compiled = self._compiled_for(int(tensor.shape[2]), int(tensor.shape[3]))
        return [np.asarray(compiled([tensor])[compiled.output(0)])]


class PortraitMatter:
    """
    A loaded MODNet session. Build once, matte many times.

    Constructing this loads the weights and builds the ``onnxruntime`` graph,
    which takes long enough that doing it per call would dominate the runtime.
    ``path`` overrides the usual lookup; ``providers`` overrides execution
    provider selection, which otherwise prefers CUDA, then DirectML, when the
    installed ``onnxruntime`` build offers them, and falls back to CPU.
    DirectML is what an AMD or NVIDIA machine gets here, since the OpenVINO
    device below is Intel-only; it needs the ``onnxruntime-directml`` wheel,
    which replaces plain ``onnxruntime`` rather than sitting beside it.

    ``device`` asks for an OpenVINO device instead — "GPU" for the integrated
    GPU, which is where this network belongs on an Intel Core Ultra machine.
    It defaults to whatever ``play <title> --accel ...`` selected, and falls
    back to ``onnxruntime`` (reporting it through :attr:`device_name`) when
    OpenVINO or the device is not there.
    """

    def __init__(self, path: str | None = None,
                 providers: list[str] | None = None,
                 device: str | None = None):
        weights = path or model_path()

        # The accelerated path first, and only if one was asked for: a machine
        # with no OpenVINO, no iGPU, or a preset of "off" lands on exactly the
        # onnxruntime session this class has always built.
        self._session = None
        resolved = accel.resolve("matte", explicit=device)
        if resolved:
            try:
                self._session = _OpenVINOSession(weights, resolved)
            except Exception as exc:
                print(f"[PortraitMatter] {resolved} unavailable, using onnxruntime: {exc}")
                resolved = None
        self.device_name = accel.describe("matte", resolved, "CPU (onnxruntime)")

        if self._session is None:
            if not MODNET_AVAILABLE:
                raise RuntimeError(
                    "onnxruntime is not installed, so MODNet matting is unavailable.\n"
                    "Install it with:\n"
                    "    pip install onnxruntime\n"
                    "Or use visual_ai.imaging.remove_background(), which needs no "
                    "model file, at the cost of a softer edge on hair."
                )
            import onnxruntime as ort

            if providers is None:
                available = ort.get_available_providers()
                # CUDA where an onnxruntime-gpu build offers it, then DirectML:
                # the vendor-neutral Windows path, and the only accelerated
                # option on AMD and NVIDIA, whose GPUs OpenVINO's Intel-only
                # GPU plugin never enumerates. A provider the installed wheel
                # lacks filters out here, so this is a no-op on the CPU wheel.
                providers = [p for p in ("CUDAExecutionProvider",
                                         "DmlExecutionProvider",
                                         "CPUExecutionProvider")
                             if p in available] or available

            self._session = ort.InferenceSession(weights, providers=providers)
            # Which provider won is not knowable until the session exists:
            # onnxruntime drops one it cannot initialize without raising, so
            # asking for DirectML is not evidence of getting it. Naming the
            # active one keeps a silent fall to CPU visible on the HUD.
            active = (self._session.get_providers() or ["CPUExecutionProvider"])[0]
            self.device_name = accel.describe("matte", resolved, {
                "CUDAExecutionProvider": "GPU (onnxruntime CUDA)",
                "DmlExecutionProvider": "GPU (onnxruntime DirectML)",
            }.get(active, "CPU (onnxruntime)"))

        self._input_name = self._session.get_inputs()[0].name
        self._fixed_size = self._declared_size()

    def _declared_size(self) -> tuple[int, int] | None:
        """
        The (h, w) the model demands, or None if it accepts any size.

        Exports vary: some are fully dynamic, others were traced at a fixed
        512x512 and report integer dimensions here. Feeding a fixed-shape model
        anything else fails inside onnxruntime with an unhelpful rank error, so
        read the declared shape up front and honour it.
        """
        shape = self._session.get_inputs()[0].shape
        if len(shape) != 4:
            return None
        height, width = shape[2], shape[3]
        if isinstance(height, int) and isinstance(width, int):
            return height, width
        return None

    def matte(self, image: np.ndarray) -> np.ndarray:
        """
        Soft alpha matte for ``image``: float32 in [0, 1], input's (h, w).

        The raw network output is used as-is. There is no threshold step the
        way there would be with a category mask — MODNet's output *is* the
        coverage estimate, and rounding it off is exactly what throws hair
        away.
        """
        import cv2

        rgb = to_rgba(image)[..., :3]
        orig_h, orig_w = rgb.shape[:2]
        size = self._fixed_size or _inference_size(orig_h, orig_w)

        tensor = _preprocess(rgb, size)
        output = self._session.run(None, {self._input_name: tensor})[0]

        matte = np.squeeze(np.asarray(output)).astype(np.float32)
        if matte.ndim != 2:
            raise RuntimeError(
                f"expected a single-channel matte from the model, got shape "
                f"{np.asarray(output).shape} — is this file really MODNet?")
        matte = cv2.resize(matte, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(matte, 0.0, 1.0)

    def cut_out(self, image: np.ndarray) -> np.ndarray:
        """
        RGBA cutout: ``image``'s colour with the matte as its alpha channel.

        Edges are bled outward — see :func:`visual_ai.imaging.bleed_edges` —
        because a transparent pixel here still holds background colour, and
        anything that filters the sprite without weighting by alpha would drag
        it back in as a rim.
        """
        from visual_ai.imaging import bleed_edges

        rgba = to_rgba(image).copy()
        rgba[..., 3] = np.clip(self.matte(rgba) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return bleed_edges(rgba)

    def close(self) -> None:
        self._session = None


_shared: PortraitMatter | None = None


def _shared_matter() -> PortraitMatter:
    global _shared
    if _shared is None:
        _shared = PortraitMatter()
    return _shared


def cut_out_person(image: np.ndarray) -> np.ndarray:
    """RGBA cutout of the person in ``image``, via a lazily-built shared session."""
    return _shared_matter().cut_out(image)


def matting_device() -> str:
    """
    Where the shared session is running — "GPU", "GPU (onnxruntime CUDA)",
    "GPU (onnxruntime DirectML)", "CPU (onnxruntime)", or "" if nothing has
    been matted yet. For a HUD line: the iGPU path is five times faster here,
    so a game that quietly lost it should be able to say so.
    """
    return _shared.device_name if _shared is not None else ""
