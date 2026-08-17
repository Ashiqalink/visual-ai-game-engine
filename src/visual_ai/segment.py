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


class PersonSegmenter:
    """
    Lazily-built wrapper around MediaPipe's ``ImageSegmenter``.

    Construction downloads the model on first use (if not already cached) and
    creates the segmenter graph. Callers should hold one instance for the
    lifetime of the pipeline rather than rebuilding it per frame.
    """

    def __init__(self):
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        base_options = mp_tasks.BaseOptions(model_asset_path=_model_path())
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
        import mediapipe as mp

        height, width = bgr_frame.shape[:2]
        small = cv2.resize(bgr_frame, (_INFER_WIDTH, _INFER_HEIGHT), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._segmenter.segment_for_video(mp_image, self._frame_index)
        self._frame_index += 1

        category_mask = result.category_mask.numpy_view()
        small_mask = (category_mask == 0).astype(np.uint8) * 255
        return cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        self._segmenter.close()
