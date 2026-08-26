"""
segment.py — PersonSegmenter: live person-vs-background masking.

Wraps MediaPipe's Tasks ``ImageSegmenter`` (the ``selfie_segmenter_landscape``
model) behind a single ``segment(frame) -> mask`` call. Unlike Hands and
FaceDetection, the segmentation model's weights are not bundled with the
``mediapipe`` pip package — :func:`_model_path` downloads them once to a local
cache on first use and reuses that file afterwards, the same "pay only if you
use it" shape :mod:`imaging`'s ``remove_background`` uses for ``rembg``.

This module is only imported when ``VisionPipeline.emit_person_mask`` is
turned on, so games that never touch the feature never pay its import or
download cost.
"""

from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np

from visual_ai import accel

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite"
)
_MODEL_FILENAME = "selfie_segmenter_landscape.tflite"

# Feeding the model the raw capture frame costs ~35ms (measured on 1280x720);
# downsampling first and upsampling the mask after costs ~11ms at this size
# and stays visually indistinguishable, since the model's native input is
# 144x256 anyway.
_INFER_WIDTH, _INFER_HEIGHT = 640, 360


def _cache_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, ".visual_ai", "models")


def _model_path() -> str:
    """Return a local path to the segmenter weights, downloading them if needed."""
    path = os.path.join(_cache_dir(), _MODEL_FILENAME)
    if os.path.isfile(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".part"
    try:
        urllib.request.urlretrieve(_MODEL_URL, tmp_path)
        os.replace(tmp_path, path)
    except Exception as exc:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"could not download the person-segmentation model from {_MODEL_URL}: {exc}\n"
            "person_mask requires network access on first use; retry once online, "
            "or leave VisionPipeline.emit_person_mask off."
        ) from exc
    return path


class _OpenVINOSegmenter:
    """
    The same ``.tflite`` weights, run on an OpenVINO device instead.

    MediaPipe Tasks builds a CPU TFLite interpreter here — there is no GPU
    delegate for it on Windows — while the network itself is small enough that
    the iGPU turns it into rounding error next to the frame's other work.
    OpenVINO reads the TFLite file directly, so this is the same model, not a
    conversion.

    The model is fully static, so nothing here needs the per-shape
    recompilation :mod:`visual_ai.matting` does.

    **This does not work with the model this module downloads.** OpenVINO
    2026.3's TFLite frontend has no translator for `Convolution2DTransposeBias`,
    a MediaPipe custom op the selfie segmenter's decoder is built from, and
    refuses the file outright::

        No translator found for Convolution2DTransposeBias node.

    So segmentation stays on MediaPipe's CPU interpreter today and says so on
    the HUD. The class is kept because the gap is on Intel's side of the line,
    not in this code: it will start working when either the frontend learns the
    op or the weights are re-exported without it, and nothing else has to
    change. MODNet matting, which is the expensive one, does reach the iGPU —
    155 ms to 32 ms on this machine.
    """

    def __init__(self, model_file: str, device: str):
        import openvino as ov

        core = ov.Core()
        model = core.read_model(model_file)
        self._compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
        self._request = self._compiled.create_infer_request()
        shape = self._compiled.inputs[0].get_partial_shape().to_shape()
        self.input_size = (int(shape[2]), int(shape[1]))     # (width, height)
        self.device = device

    def person_mask(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 person mask at the network's own resolution, 255 = person."""
        resized = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_AREA)
        tensor = (resized.astype(np.float32) / 255.0)[None]
        output = np.asarray(self._request.infer([tensor])[self._compiled.output(0)])
        confidence = np.squeeze(output)
        if confidence.ndim == 3:
            # Two-channel export: channel 0 is background, 1 is person. The
            # single-channel export is already P(person).
            confidence = confidence[..., 1] if confidence.shape[-1] == 2 else confidence[..., 0]
        return (confidence >= 0.5).astype(np.uint8) * 255

    def close(self) -> None:
        self._request = None


class PersonSegmenter:
    """
    Lazily-built wrapper around MediaPipe's ``ImageSegmenter``.

    Construction downloads the model on first use (if not already cached) and
    creates the segmenter graph. Callers should hold one instance for the
    lifetime of the pipeline rather than rebuilding it per frame.

    ``device`` asks for an OpenVINO device — "GPU" for the integrated GPU —
    and defaults to whatever ``play <title> --accel ...`` selected. Anything
    that stops that path from being built (no OpenVINO, no such device) falls
    back to MediaPipe and says so in :attr:`device_name`.
    """

    def __init__(self, device: str | None = None):
        model_file = _model_path()

        self._openvino = None
        rejected = ""
        resolved = accel.resolve("matte", explicit=device)
        if resolved:
            try:
                self._openvino = _OpenVINOSegmenter(model_file, resolved)
            except Exception as exc:
                print(f"[PersonSegmenter] {resolved} cannot run this model, "
                      f"using MediaPipe: {exc}")
                rejected, resolved = f" ({resolved} rejected the model)", None
        self.device_name = accel.describe("matte", resolved, "CPU (MediaPipe)") + rejected
        if self._openvino is not None:
            self._segmenter = None
            self._frame_index = 0
            return

        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        base_options = mp_tasks.BaseOptions(model_asset_path=model_file)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        self._frame_index = 0

    def segment(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Return a full-resolution person mask for ``bgr_frame``.

        uint8, same (height, width) as the input, 255 where the model judges
        the pixel to be the person and 0 for background. Downsampled before
        inference and upsampled after — see the module docstring for why.
        """
        height, width = bgr_frame.shape[:2]
        small = cv2.resize(bgr_frame, (_INFER_WIDTH, _INFER_HEIGHT), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if self._openvino is not None:
            small_mask = self._openvino.person_mask(rgb)
            return cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_LINEAR)

        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._segmenter.segment_for_video(mp_image, self._frame_index)
        self._frame_index += 1

        category_mask = result.category_mask.numpy_view()
        small_mask = (category_mask == 0).astype(np.uint8) * 255
        return cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        if self._openvino is not None:
            self._openvino.close()
        else:
            self._segmenter.close()
