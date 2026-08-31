"""
openvino_zoo.py — object detection, body pose, faces and classification on the
NPU / iGPU / CPU.

`openvino_hands.py` put one network family (MediaPipe Hands) on the NPU. This
module is the rest of what a webcam game might ask a vision network for, each
behind one small class with the same shape — construct on a device, call it
with an RGB frame, read plain results in frame pixels:

    ObjectDetector    YOLOv10n, 80 COCO classes, NMS-free (boxes come out of
                      the network already deduplicated)
    SSDDetector       Intel's person-detection-0200 / face-detection-0200:
                      one class, 256x256, the cheapest detectors here
    FaceDetector      MediaPipe BlazeFace (short range): box + 6 keypoints
    BodyPose          MediaPipe Pose: 33 body landmarks for one person, with
                      the detector -> ROI -> landmark tracking loop rebuilt
                      the way `openvino_hands` does it for hands
    HeatmapPose       Intel's human-pose-estimation-0001 (OpenPose): 18 body
                      parts from heatmap peaks, single person
    ImageClassifier   MobileNetV2 / ResNet50, ImageNet top-k

Every class resolves its device the same way: an explicit argument, then
``VISUAL_AI_ZOO_DEVICE``, then the ``--accel`` preset (`accel.preset`), then
the first of NPU / GPU / CPU that is present. A device that refuses the model
falls back to OpenVINO's CPU plugin and *says so* in `device_name` — a silent
fallback is indistinguishable in play from an accelerator that is working.

What the NPU is and is not good at, measured on a Core Ultra 5 225H (AI Boost
NPU, arch 3720, 6.5 TFLOPS FP16 / 13 TOPS INT8; Arc 130T iGPU; OpenVINO
2026.3) — `benchmarks/bench_npu_traits.py` reproduces the numbers:

  * Static shapes only. Every dynamic axis must be reshaped away before
    compile; a graph whose *outputs* are dynamic (a TF SSD with NMS inside)
    does not raise on the NPU, it segfaults the process (exit 139).
  * FP16 and INT8 only. FP32 weights are converted at compile time, so
    logits differ from the CPU in the third decimal and thresholds tuned on
    CPU need re-checking near the boundary.
  * A fixed ~0.8-1 ms per inference. A 128x128 face detector is *slower* on
    the NPU (0.84 ms) than on the CPU (0.64 ms); the NPU wins from roughly
    a 224x224 MobileNet upward (1.1 ms vs 3.5 ms) and the gap widens with
    the network — ResNet50 3.7 vs 14.8 ms, YOLOv10n@640 9.7 vs 20 ms,
    OpenPose 6.2 vs 28.9 ms.
  * The iGPU is faster on everything here (YOLOv10n 3.0 ms, OpenPose 2.0
    ms) but costs 2.3 CPU cores of driver time against the NPU's 1.4, and a
    game is already drawing on it. Transformers are the NPU's weak spot:
    Depth-Anything-V2-S 97 ms on the NPU against 19 on the iGPU, RT-DETR
    43 vs 12 ms.
  * Compile is slow cold (30 s for a ViT-S, 1-2 s for a CNN) and fast warm:
    the NPU driver caches compiled blobs, so the second process to compile
    the same model at the same shape gets it in ~50-90 ms.
  * Preprocessing belongs in the graph. `PrePostProcessor` folds the
    uint8 -> float, layout and mean/scale into the compiled model, so the
    frame goes to the device as the resized uint8 crop and nothing is
    normalised in numpy.

Weights come from three places, all fetched on first use into the same
per-user cache `segment.py` and `matting.py` share, and checked against a
pinned SHA-256: the MediaPipe `.tflite` files inside the installed
``mediapipe`` package (no download), ONNX files from the ONNX model zoo and
Hugging Face, and Intel IR pairs from the Open Model Zoo storage.

Frames are **RGB** uint8, the way `pipeline.py` hands them to the trackers.
Coordinates come back in that frame's pixels — nothing here flips.

**Untested in real play.** Every number here comes from recorded frames and a
bench; live capture with a person in front of the camera needs a human.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
import urllib.request

import cv2
import numpy as np

from visual_ai import accel
from visual_ai.openvino_hands import _crop, _letterbox, _nms_weighted

__all__ = [
    "OpenVINOZooUnavailable",
    "Detection",
    "ObjectDetector",
    "SSDDetector",
    "FaceDetector",
    "BodyPose",
    "PoseResult",
    "HeatmapPose",
    "ImageClassifier",
    "resolve_device",
    "model_path",
    "mediapipe_model",
    "COCO_LABELS",
    "POSE_LANDMARK_NAMES",
    "OPENPOSE_PART_NAMES",
    "FACE_KEYPOINT_NAMES",
]

#: Pin one consumer to one device, the way ``VISUAL_AI_HAND_DEVICE`` does for
#: hands. "auto" / empty follows the ``--accel`` preset.
ENV_DEVICE = "VISUAL_AI_ZOO_DEVICE"

#: Preference order per ``--accel`` preset. Unlike hands and matting there is
#: no non-OpenVINO path to fall back to here, so "off" means the OpenVINO CPU
#: plugin rather than "do not run".
_PREFERENCE: dict[str, tuple[str, ...]] = {
    "off": ("CPU",),
    "auto": ("NPU", "GPU", "CPU"),
    "npu": ("NPU", "CPU"),
    "gpu": ("GPU", "CPU"),
    "cpu": ("CPU",),
}


class OpenVINOZooUnavailable(RuntimeError):
    """OpenVINO is missing, or no device — not even CPU — would take the model."""


# ── weights ──────────────────────────────────────────────────────────────────

_OMZ = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"
_ONNX_ZOO = "https://github.com/onnx/models/raw/main/validated/vision"

#: name -> (filename, url, sha256). IR models are two files; the entry names
#: the .xml and the .bin sits beside it under the same stem.
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "yolov10n": (
        "yolov10n.onnx",
        "https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model.onnx",
        "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b"),
    "mobilenetv2": (
        "mobilenetv2-12.onnx",
        f"{_ONNX_ZOO}/classification/mobilenet/model/mobilenetv2-12.onnx",
        "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"),
    "resnet50": (
        "resnet50-v2-7.onnx",
        f"{_ONNX_ZOO}/classification/resnet/model/resnet50-v2-7.onnx",
        "79102261eb6e5fd7af5d27f41316293e388c5cb691e5d25bfb035c4f64fefe31"),
    "imagenet-labels": (
        "imagenet_classes.txt",
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "1f386e0d1cb6e28b9c2dac651c3dea6801e98ad1b41a14ce6bb1a093d72069f5"),
    "person-detection-0200": (
        "person-detection-0200.xml",
        f"{_OMZ}/person-detection-0200/FP16/person-detection-0200.xml",
        "6a393e1a58607cf65ff58b437b0aeb0bf6c46a18926e1c5718374c804c419faa"),
    "person-detection-0200.bin": (
        "person-detection-0200.bin",
        f"{_OMZ}/person-detection-0200/FP16/person-detection-0200.bin",
        "cebd5b36edce228fd7a519b372d824f220466c4c7a0d75bfe22fbb6c5aca4fcd"),
    "face-detection-0200": (
        "face-detection-0200.xml",
        f"{_OMZ}/face-detection-0200/FP16/face-detection-0200.xml",
        "62d01703062d76f94d607584e111feabdfd9f8414cb1d89572387f8ba86a5fc7"),
    "face-detection-0200.bin": (
        "face-detection-0200.bin",
        f"{_OMZ}/face-detection-0200/FP16/face-detection-0200.bin",
        "7218f93856d3f2c9c5739a0a3c8bc755757ed8a846f4848a20ddfa9c071793a0"),
    "human-pose-estimation-0001": (
        "human-pose-estimation-0001.xml",
        f"{_OMZ}/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml",
        "ebd70031f92e52b7f1d6ef3b1aead6eff0c9c52130e65ecf77a2447b90a32b84"),
    "human-pose-estimation-0001.bin": (
        "human-pose-estimation-0001.bin",
        f"{_OMZ}/human-pose-estimation-0001/FP16/human-pose-estimation-0001.bin",
        "fd4604233dd9ca09fba51c098b662e5fe6b03bf5dac174b686c3d6d5977cf8d5"),
}


def _cache_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, ".visual_ai", "models")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: str, expect_sha256: str) -> None:
    tmp = path + ".part"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        got = _sha256(tmp)
        if got != expect_sha256:
            os.remove(tmp)
            raise OpenVINOZooUnavailable(
                f"{os.path.basename(path)} downloaded from {url} has SHA-256 {got}, "
                f"expected {expect_sha256}. The pinned weights have changed or the "
                "download was corrupted; delete the file and retry, or pin the new digest.")
        os.replace(tmp, path)
    except OpenVINOZooUnavailable:
        raise
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise OpenVINOZooUnavailable(
            f"could not download {os.path.basename(path)} from {url}: {exc}") from exc


def model_path(name: str) -> str:
    """
    Local path to a registered weight file, downloading it on first use.

    For an IR model this returns the ``.xml`` and makes sure the ``.bin`` is
    beside it, which is where OpenVINO looks for it.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown zoo model {name!r}; known: {sorted(_REGISTRY)}")
    filename, url, digest = _REGISTRY[name]
    path = os.path.join(_cache_dir(), filename)
    if not os.path.isfile(path):
        _download(url, path, digest)
    if filename.endswith(".xml"):
        model_path(name + ".bin")
    return path


def mediapipe_model(relative: str) -> str:
    """A `.tflite` inside the installed ``mediapipe`` package, e.g.
    ``"pose_detection/pose_detection.tflite"``."""
    try:
        import mediapipe
    except Exception as exc:                                  # pragma: no cover
        raise OpenVINOZooUnavailable(
            f"mediapipe is not installed, so its weights are not on disk: {exc}") from exc
    path = os.path.join(os.path.dirname(mediapipe.__file__), "modules", relative)
    if not os.path.isfile(path):
        raise OpenVINOZooUnavailable(f"missing mediapipe weights: {path}")
    return path


# ── devices ──────────────────────────────────────────────────────────────────

def resolve_device(explicit: str | None = None) -> str:
    """
    The OpenVINO device a zoo network should compile for.

    `explicit` (or ``VISUAL_AI_ZOO_DEVICE``) names one device; otherwise the
    ``--accel`` preset's preference order applies. CPU is always the last
    candidate — it is the one plugin every OpenVINO install has.
    """
    wanted = (explicit or os.environ.get(ENV_DEVICE) or "").strip().upper()
    if wanted in ("", "AUTO"):
        candidates = _PREFERENCE[accel.preset()]
    elif wanted in ("OFF", "NONE"):
        candidates = ("CPU",)
    else:
        candidates = (wanted, "CPU")
    present = accel.available_devices()
    for candidate in candidates:
        if candidate in present:
            return candidate
    return "CPU"


_CORE = None


def _core():
    """One `ov.Core` per process: it owns the device handles and plugin
    caches, and every network here shares it."""
    global _CORE
    if _CORE is None:
        try:
            import openvino as ov
        except Exception as exc:
            raise OpenVINOZooUnavailable(f"openvino is not installed: {exc}") from exc
        _CORE = ov.Core()
    return _CORE


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # ±80 stays inside float32 (exp(80) ≈ 5.5e34); ±100 overflowed f32 logits.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


# ── one compiled network ─────────────────────────────────────────────────────

def _prepare_model(path: str, *, height: int, width: int, model_layout: str = "NCHW",
                   scale: float = 255.0, mean=None, std=None, bgr: bool = False):
    """
    Read `path`, fix its input to ``1 x height x width``, and fold the
    preprocessing into the graph: a uint8 NHWC tensor in, converted to f32,
    channels reversed if `bgr`, divided by `scale`, `mean` subtracted, then
    divided by `std`. Returns the model ready for `Core.compile_model`.

    Static ONNX inputs are reshaped too — YOLOv10n ships at 640 and runs at
    320/416/512 after a reshape; only the anchors' count changes.
    """
    import openvino as ov
    from openvino.preprocess import PrePostProcessor

    model = _core().read_model(path)
    port = model.inputs[0]
    shape = [1, 3, height, width] if model_layout == "NCHW" else [1, height, width, 3]
    if port.get_partial_shape().is_dynamic or list(port.get_shape()) != shape:
        model.reshape({port: ov.PartialShape(shape)})

    ppp = PrePostProcessor(model)
    ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC"))
    ppp.input().model().set_layout(ov.Layout(model_layout))
    steps = ppp.input().preprocess().convert_element_type(ov.Type.f32)
    if bgr:
        steps = steps.reverse_channels()
    if scale and scale != 1.0:
        steps = steps.scale(float(scale))
    if mean is not None:
        steps = steps.mean(list(mean) if isinstance(mean, (tuple, list)) else float(mean))
    if std is not None:
        steps = steps.scale(list(std) if isinstance(std, (tuple, list)) else float(std))
    return ppp.build()


class _Net:
    """
    One network compiled for one device, with a uint8 HWC input buffer bound.

    `scale` / `mean` follow OpenVINO's PrePostProcessor: the uint8 input is
    converted to float, divided by `scale`, then `mean` is subtracted, then
    divided by `std` — all inside the compiled graph, so the caller only ever
    writes pixels into `self.input`. `model_layout` names the layout the
    graph wants ("NCHW" for the ONNX / IR models, "NHWC" for TFLite), and the
    transpose, when one is needed, is the device's job too.

    Construction is where an absent or refusing device fails, deliberately —
    the wrappers below catch that and fall back to CPU at startup rather than
    on the first frame.
    """

    def __init__(self, path: str, device: str, *, height: int, width: int,
                 model_layout: str = "NCHW", scale: float = 255.0,
                 mean: tuple[float, ...] | float | None = None,
                 std: tuple[float, ...] | float | None = None,
                 bgr: bool = False, hint: str = "LATENCY"):
        import openvino as ov

        model = _prepare_model(path, height=height, width=width, model_layout=model_layout,
                               scale=scale, mean=mean, std=std, bgr=bgr)
        started = time.perf_counter()
        compiled = _core().compile_model(model, device, {"PERFORMANCE_HINT": hint})
        self.compile_ms = (time.perf_counter() - started) * 1e3
        self.request = compiled.create_infer_request()
        self.input = np.zeros((1, height, width, 3), np.uint8)
        # `shared_memory=True` or the request keeps a *copy* of the zeros and
        # every frame written into `self.input` afterwards is never seen.
        self.request.set_input_tensor(ov.Tensor(self.input, shared_memory=True))
        self.outputs = [out.get_any_name() for out in compiled.outputs]
        self.device = device
        self.height, self.width = height, width
        self.infer_ms = 0.0

    def infer(self) -> dict[str, np.ndarray]:
        """Run on whatever is in `self.input`; outputs by port name."""
        started = time.perf_counter()
        self.request.infer()
        self.infer_ms = (time.perf_counter() - started) * 1e3
        return {name: np.asarray(self.request.get_output_tensor(i).data)
                for i, name in enumerate(self.outputs)}

    def start(self) -> None:
        self.request.start_async()

    def wait(self) -> dict[str, np.ndarray]:
        self.request.wait()
        return {name: np.asarray(self.request.get_output_tensor(i).data)
                for i, name in enumerate(self.outputs)}


def _compile_with_fallback(build, device: str | None, what: str):
    """
    `build(device)` on the resolved device, then on CPU if that refuses.

    Returns ``(net, device_name)`` where `device_name` is the HUD string:
    the device when it worked, or ``"CPU (NPU refused: ...)"`` naming the
    device that lost and why.
    """
    resolved = resolve_device(device)
    try:
        return build(resolved), resolved
    except Exception as exc:
        if resolved == "CPU":
            raise OpenVINOZooUnavailable(
                f"could not compile {what} for CPU: {exc}") from exc
        reason = str(exc).splitlines()[0][:80]
    try:
        return build("CPU"), f"CPU ({resolved} refused: {reason})"
    except Exception as exc:
        raise OpenVINOZooUnavailable(
            f"could not compile {what} for {resolved} or CPU: {exc}") from exc


# ── results ──────────────────────────────────────────────────────────────────

class Detection:
    """One detected thing: `label`, `score`, `box` as ``(x0, y0, x1, y1)`` in
    frame pixels, and `keypoints` as an ``(N, 2)`` pixel array or None."""

    __slots__ = ("label", "score", "box", "keypoints")

    def __init__(self, label: str, score: float, box, keypoints=None):
        self.label = label
        self.score = float(score)
        self.box = tuple(float(v) for v in box)
        self.keypoints = keypoints

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def __repr__(self) -> str:
        x0, y0, x1, y1 = self.box
        return f"Detection({self.label!r}, {self.score:.2f}, [{x0:.0f} {y0:.0f} {x1:.0f} {y1:.0f}])"


# ── object detection: YOLOv10n ───────────────────────────────────────────────

COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush")


class ObjectDetector:
    """
    YOLOv10n, 80 COCO classes.

    v10 is the NMS-free YOLO: the graph ends in a fixed ``[1, 300, 6]`` of
    ``x0, y0, x1, y1, score, class`` in input pixels, already deduplicated,
    which is exactly the static, post-processing-free shape the NPU wants.
    `size` is the square the frame is letterboxed into. The Hugging Face
    export is fixed at 640: its anchor-split constants do not survive a
    reshape, so any other `size` fails at compile with OpenVINO's reason
    (measured in `bench_npu_traits.py`); a smaller input needs a re-export.
    """

    def __init__(self, device: str | None = None, size: int = 640,
                 min_score: float = 0.35):
        path = model_path("yolov10n")
        self._net, self.device_name = _compile_with_fallback(
            lambda dev: _Net(path, dev, height=size, width=size, scale=255.0),
            device, "YOLOv10n")
        self.device = self._net.device
        self.size = int(size)
        self.min_score = float(min_score)
        self.compile_ms = self._net.compile_ms
        self.timings = {"infer_ms": 0.0, "frames": 0}

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        _, scale, left, top = _letterbox(rgb, self.size, out=self._net.input[0])
        raw = next(iter(self._net.infer().values()))[0]
        self.timings["infer_ms"] = self._net.infer_ms
        self.timings["frames"] += 1
        return self._decode(raw, scale, left, top)

    def _decode(self, raw: np.ndarray, scale: float, left: int, top: int) -> list[Detection]:
        keep = raw[:, 4] >= self.min_score
        found = []
        for x0, y0, x1, y1, score, cls in raw[keep]:
            index = int(cls)
            label = COCO_LABELS[index] if 0 <= index < len(COCO_LABELS) else str(index)
            found.append(Detection(label, score, (
                (x0 - left) / scale, (y0 - top) / scale,
                (x1 - left) / scale, (y1 - top) / scale)))
        return found

    def close(self) -> None:
        self._net = None


# ── one-class SSDs: Intel person / face 0200 ─────────────────────────────────

class SSDDetector:
    """
    Intel's ``person-detection-0200`` or ``face-detection-0200`` (`kind`).

    MobileNetV2 SSD-lite at 256x256, one class, output ``[1, 1, 200, 7]`` of
    ``image_id, label, score, x0, y0, x1, y1`` with the box normalised to the
    input. The frame is resized straight to the square — no letterbox — which
    is what the models were trained on and why a person is found at 2 ms.
    """

    _KINDS = {"person": "person-detection-0200", "face": "face-detection-0200"}

    def __init__(self, kind: str = "person", device: str | None = None,
                 min_score: float = 0.5):
        if kind not in self._KINDS:
            raise ValueError(f"kind must be one of {sorted(self._KINDS)}, not {kind!r}")
        path = model_path(self._KINDS[kind])
        self._net, self.device_name = _compile_with_fallback(
            lambda dev: _Net(path, dev, height=256, width=256, scale=1.0, bgr=True),
            device, self._KINDS[kind])
        self.device = self._net.device
        self.kind = kind
        self.min_score = float(min_score)
        self.compile_ms = self._net.compile_ms
        self.timings = {"infer_ms": 0.0, "frames": 0}

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        height, width = rgb.shape[:2]
        cv2.resize(rgb, (256, 256), dst=self._net.input[0], interpolation=cv2.INTER_LINEAR)
        raw = next(iter(self._net.infer().values())).reshape(-1, 7)
        self.timings["infer_ms"] = self._net.infer_ms
        self.timings["frames"] += 1
        return self._decode(raw, width, height)

    def _decode(self, raw: np.ndarray, width: int, height: int) -> list[Detection]:
        found = []
        for image_id, _label, score, x0, y0, x1, y1 in raw:
            if image_id < 0 or score < self.min_score:
                continue
            found.append(Detection(self.kind, score, (
                x0 * width, y0 * height, x1 * width, y1 * height)))
        return found

    def close(self) -> None:
        self._net = None


# ── MediaPipe SSD anchors ────────────────────────────────────────────────────

def _anchors(size: int, layers: tuple[tuple[int, int], ...]) -> np.ndarray:
    """
    Anchor centres, normalised, for MediaPipe's fixed-size SSD heads.

    `layers` is ``((stride, anchors_per_cell), ...)``; the palm detector's
    (8,2),(16,6) gives the 2016 `openvino_hands` builds, BlazeFace's 896 is
    (8,2),(16,6) at 128, pose detection's 2254 is (8,2),(16,2),(32,6) at 224.
    """
    centres = []
    for stride, per_cell in layers:
        cells = int(math.ceil(size / stride))
        for y in range(cells):
            for x in range(cells):
                centres.extend([((x + 0.5) / cells, (y + 0.5) / cells)] * per_cell)
    return np.asarray(centres, dtype=np.float32)


def _decode_ssd(boxes: np.ndarray, logits: np.ndarray, anchors: np.ndarray,
                size: int, num_keypoints: int, min_score: float) -> list[dict]:
    """MediaPipe's TensorsToDetections for a fixed-anchor head: regressions
    are in input pixels relative to the anchor centre."""
    scores = _sigmoid(logits[:, 0])
    keep = scores >= min_score
    found: list[dict] = []
    if not keep.any():
        return found
    regressed = boxes[keep] / size
    centres = anchors[keep]
    cx = regressed[:, 0] + centres[:, 0]
    cy = regressed[:, 1] + centres[:, 1]
    w, h = regressed[:, 2], regressed[:, 3]
    keypoints = (regressed[:, 4:4 + num_keypoints * 2]
                 .reshape(-1, num_keypoints, 2) + centres[:, None, :])
    for i, score in enumerate(scores[keep]):
        found.append({"score": float(score), "kp": keypoints[i],
                      "box": (cx[i] - w[i] / 2, cy[i] - h[i] / 2,
                              cx[i] + w[i] / 2, cy[i] + h[i] / 2)})
    return found


def _unletterbox(detections: list[dict], size: int, scale: float, left: int, top: int):
    """Letterbox-normalised boxes and keypoints -> frame pixels, in place."""
    for detection in detections:
        kp = detection["kp"]
        detection["kp"] = np.stack([(kp[:, 0] * size - left) / scale,
                                    (kp[:, 1] * size - top) / scale], axis=1)
        x0, y0, x1, y1 = detection["box"]
        detection["box"] = ((x0 * size - left) / scale, (y0 * size - top) / scale,
                            (x1 * size - left) / scale, (y1 * size - top) / scale)
    return detections


# ── face detection: BlazeFace short range ────────────────────────────────────

FACE_KEYPOINT_NAMES = ("right_eye", "left_eye", "nose", "mouth", "right_ear", "left_ear")

_FACE_SIZE = 128
_FACE_ANCHORS = _anchors(_FACE_SIZE, ((8, 2), (16, 6)))


class FaceDetector:
    """
    MediaPipe's BlazeFace, short-range model (faces within ~2 m — a webcam).

    128x128 letterbox, input -1..1, 896 anchors, 6 keypoints. This is the same
    network `pipeline.py`'s MediaPipe FaceDetection runs on the CPU, so the
    boxes are comparable — and at 128x128 it is the network the NPU is
    *worst* at relative to the CPU (0.84 vs 0.64 ms): the fixed per-inference
    cost dominates. It is here so a game that already has the NPU busy can
    keep everything on one device, not because it is faster there.
    """

    def __init__(self, device: str | None = None, min_score: float = 0.5,
                 max_faces: int = 2):
        path = mediapipe_model("face_detection/face_detection_short_range.tflite")
        self._net, self.device_name = _compile_with_fallback(
            lambda dev: _Net(path, dev, height=_FACE_SIZE, width=_FACE_SIZE,
                             model_layout="NHWC", scale=127.5, mean=1.0),
            device, "BlazeFace")
        self.device = self._net.device
        self.min_score = float(min_score)
        self.max_faces = max(1, int(max_faces))
        self.compile_ms = self._net.compile_ms
        self.timings = {"infer_ms": 0.0, "frames": 0}

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        _, scale, left, top = _letterbox(rgb, _FACE_SIZE, out=self._net.input[0])
        outputs = self._net.infer()
        self.timings["infer_ms"] = self._net.infer_ms
        self.timings["frames"] += 1
        boxes, logits = _split_ssd_outputs(outputs, 16)
        found = _decode_ssd(boxes, logits, _FACE_ANCHORS, _FACE_SIZE, 6, self.min_score)
        found = _unletterbox(_nms_weighted(found)[: self.max_faces],
                             _FACE_SIZE, scale, left, top)
        return [Detection("face", d["score"], d["box"], d["kp"]) for d in found]

    def close(self) -> None:
        self._net = None


def _split_ssd_outputs(outputs: dict[str, np.ndarray], box_width: int):
    """The regressor and classifier tensors, whichever order the ports came in."""
    arrays = [np.asarray(v)[0] for v in outputs.values()]
    if arrays[0].shape[-1] == box_width:
        return arrays[0], arrays[1]
    return arrays[1], arrays[0]


# ── body pose: MediaPipe Pose ────────────────────────────────────────────────

POSE_LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index")

_POSE_DET_SIZE = 224
_POSE_LAND_SIZE = 256
_POSE_ANCHORS = _anchors(_POSE_DET_SIZE, ((8, 2), (16, 2), (32, 6)))
#: pose_detection_to_roi / pose_landmarks_to_roi: the square is 1.25x the
#: alignment distance either side of the hip centre.
_POSE_ROI_SCALE = 1.25
#: The detector's four keypoints: hip centre, full-body scale point,
#: shoulder centre, upper-body scale point. The ROI comes from the first two.
_POSE_DET_KP = 4
#: The landmark head returns 39 points: 33 body landmarks then 6 auxiliary
#: ones; 33 and 34 are the hip centre and scale point the next frame's ROI
#: is rebuilt from.
_POSE_NUM_LANDMARKS = 39
_POSE_AUX_CENTRE, _POSE_AUX_SCALE = 33, 34


class PoseResult:
    """
    One tracked body. `landmarks` is ``(33, 4)`` — x, y in frame pixels, z in
    the same units (negative toward the camera, like MediaPipe), visibility
    in 0..1 — ordered as `POSE_LANDMARK_NAMES`. `score` is the network's
    presence flag for the crop. `crop_mask` is the 256x256 person mask the
    landmark network produces alongside, in crop space; `mask(shape)` warps it
    back over the frame.
    """

    __slots__ = ("landmarks", "landmarks_all", "score", "rect", "crop_mask", "_inverse")

    def __init__(self, landmarks, score, rect, crop_mask, inverse, landmarks_all=None):
        self.landmarks = landmarks
        #: All 39 points the network returns (33 body + 6 auxiliary); the
        #: tracker rebuilds the next ROI from auxiliary 33 and 34.
        self.landmarks_all = landmarks if landmarks_all is None else landmarks_all
        self.score = float(score)
        self.rect = rect
        self.crop_mask = crop_mask
        self._inverse = inverse

    def mask(self, shape) -> np.ndarray:
        """The person mask as float32 0..1 over a frame of `shape`."""
        height, width = shape[:2]
        return cv2.warpAffine(self.crop_mask, self._inverse, (width, height),
                              flags=cv2.INTER_LINEAR, borderValue=0.0)

    def point(self, name: str) -> tuple[float, float]:
        index = POSE_LANDMARK_NAMES.index(name)
        return float(self.landmarks[index, 0]), float(self.landmarks[index, 1])


def _alignment_rect(centre, scale_point, scale: float):
    """MediaPipe's AlignmentPointsRectsCalculator + RectTransformation: a
    square centred on `centre`, twice the distance to `scale_point` on a
    side, rotated so centre->scale_point points straight up, then scaled."""
    cx, cy = float(centre[0]), float(centre[1])
    dx, dy = float(scale_point[0]) - cx, float(scale_point[1]) - cy
    size = 2.0 * math.hypot(dx, dy) * scale
    angle = math.pi * 0.5 - math.atan2(-dy, dx)
    angle -= 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))
    return cx, cy, size, angle


class BodyPose:
    """
    MediaPipe Pose on one device: 33 landmarks for one person.

    The graph is MediaPipe's: letterbox to 224 -> pose detection (SSD, 2254
    anchors) -> weighted NMS -> rotated square ROI from the hip centre and
    scale keypoints -> 256x256 landmark network -> next frame's ROI from the
    auxiliary landmarks, with the detector re-run only when the crop's
    presence flag drops below `min_tracking_confidence`. Single person by
    design — MediaPipe's is too.

    Heatmap refinement (the full graph nudges each landmark toward the peak
    of its 64x64 heatmap) is not applied; the regressed coordinates are used
    as-is.

    NPU note (measured 2026-08-29, fixture clip, untested in real play): the
    *detector* graph loses precision on the NPU — above-threshold keypoint
    regressions sit 11-18 px (224-space) from the CPU f32 result where the
    GPU sits < 1 px, so the ROI lands 20-47 px off and 5-8 % small on a
    640x480 frame. The landmark net matches CPU within ~2 px on any device.
    `detector_device="GPU"` (or "CPU") keeps the ROI stage exact while the
    per-frame landmark net stays on the NPU; the detector only runs when
    tracking is lost, so its device costs nothing per tracked frame.
    """

    def __init__(self, device: str | None = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 detector_device: str | None = None):
        det_path = mediapipe_model("pose_detection/pose_detection.tflite")
        land_path = mediapipe_model("pose_landmark/pose_landmark_full.tflite")

        def build_landmark(dev):
            return _Net(land_path, dev, height=_POSE_LAND_SIZE, width=_POSE_LAND_SIZE,
                        model_layout="NHWC", scale=255.0)

        def build_detector(dev):
            return _Net(det_path, dev, height=_POSE_DET_SIZE, width=_POSE_DET_SIZE,
                        model_layout="NHWC", scale=127.5, mean=1.0)

        self._land, self.device_name = _compile_with_fallback(
            build_landmark, device, "MediaPipe Pose landmarks")
        self._det, detector_name = _compile_with_fallback(
            build_detector, device if detector_device is None else detector_device,
            "MediaPipe Pose detector")
        self.device = self._land.device
        if self._det.device != self._land.device:
            self.device_name = f"{self.device_name} (detector {detector_name})"
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.compile_ms = self._det.compile_ms + self._land.compile_ms
        self._rect = None
        self.timings = {"det_ms": 0.0, "land_ms": 0.0, "detector_runs": 0, "frames": 0}

    def _detect(self, rgb: np.ndarray):
        _, scale, left, top = _letterbox(rgb, _POSE_DET_SIZE, out=self._det.input[0])
        outputs = self._det.infer()
        self.timings["det_ms"] = self._det.infer_ms
        self.timings["detector_runs"] += 1
        boxes, logits = _split_ssd_outputs(outputs, 4 + 2 * _POSE_DET_KP)
        found = _decode_ssd(boxes, logits, _POSE_ANCHORS, _POSE_DET_SIZE,
                            _POSE_DET_KP, self.min_detection_confidence)
        found = _unletterbox(_nms_weighted(found)[:1], _POSE_DET_SIZE, scale, left, top)
        if not found:
            return None
        kp = found[0]["kp"]
        return _alignment_rect(kp[0], kp[1], _POSE_ROI_SCALE)

    def process(self, rgb: np.ndarray) -> PoseResult | None:
        self.timings["frames"] += 1
        self.timings["det_ms"] = 0.0
        if self._rect is None:
            self._rect = self._detect(rgb)
            if self._rect is None:
                self.timings["land_ms"] = 0.0
                return None

        _, inverse = _crop(rgb, self._rect, _POSE_LAND_SIZE, out=self._land.input[0])
        outputs = self._land.infer()
        self.timings["land_ms"] = self._land.infer_ms
        result = self._decode(outputs, inverse)
        if result.score < self.min_tracking_confidence:
            self._rect = None
            return None
        aux = result.landmarks_all
        self._rect = _alignment_rect(aux[_POSE_AUX_CENTRE, :2], aux[_POSE_AUX_SCALE, :2],
                                     _POSE_ROI_SCALE)
        return result

    def _decode(self, outputs: dict[str, np.ndarray], inverse) -> PoseResult:
        arrays = {np.asarray(v).size: np.asarray(v) for v in outputs.values()}
        raw = arrays[_POSE_NUM_LANDMARKS * 5].reshape(_POSE_NUM_LANDMARKS, 5)
        flag = float(arrays[1].ravel()[0])
        if not 0.0 <= flag <= 1.0:
            flag = float(_sigmoid(np.float32(flag)))
        mask = arrays.get(_POSE_LAND_SIZE * _POSE_LAND_SIZE)
        crop_mask = (_sigmoid(mask.reshape(_POSE_LAND_SIZE, _POSE_LAND_SIZE))
                     .astype(np.float32) if mask is not None else None)

        points = raw[:, :2]
        pixels = np.stack([
            inverse[0, 0] * points[:, 0] + inverse[0, 1] * points[:, 1] + inverse[0, 2],
            inverse[1, 0] * points[:, 0] + inverse[1, 1] * points[:, 1] + inverse[1, 2],
        ], axis=1)
        depth = raw[:, 2] * (self._rect[2] / _POSE_LAND_SIZE)
        visibility = _sigmoid(raw[:, 3])
        everything = np.column_stack([pixels, depth, visibility]).astype(np.float32)
        return PoseResult(everything[:33], flag, self._rect, crop_mask, inverse,
                          landmarks_all=everything)

    def close(self) -> None:
        self._rect = None
        self._det = self._land = None


# ── body parts: OpenPose heatmaps ────────────────────────────────────────────

OPENPOSE_PART_NAMES = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear")

_OPENPOSE_H, _OPENPOSE_W = 256, 456


class HeatmapPose:
    """
    Intel's ``human-pose-estimation-0001`` (OpenPose, MobileNet backbone):
    18 body parts as heatmap peaks.

    The network returns 19 heatmaps and 38 part-affinity fields at 1/8 of
    its 256x456 input. Only the heatmaps are read: one peak per part, above
    `min_score`, which is a single-person reading — the PAF grouping that
    turns peaks into several skeletons is not implemented. The frame is
    resized straight to 456x256 (no letterbox), so a 4:3 frame is stretched
    ~4% wider than it was trained on; the peaks come back in frame pixels.
    """

    def __init__(self, device: str | None = None, min_score: float = 0.3):
        path = model_path("human-pose-estimation-0001")
        self._net, self.device_name = _compile_with_fallback(
            lambda dev: _Net(path, dev, height=_OPENPOSE_H, width=_OPENPOSE_W,
                             scale=1.0, bgr=True),
            device, "human-pose-estimation-0001")
        self.device = self._net.device
        self.min_score = float(min_score)
        self.compile_ms = self._net.compile_ms
        self.timings = {"infer_ms": 0.0, "frames": 0}

    def detect(self, rgb: np.ndarray) -> dict[str, tuple[float, float, float]]:
        """``{part_name: (x, y, score)}`` in frame pixels, for parts found."""
        height, width = rgb.shape[:2]
        cv2.resize(rgb, (_OPENPOSE_W, _OPENPOSE_H), dst=self._net.input[0],
                   interpolation=cv2.INTER_LINEAR)
        outputs = self._net.infer()
        self.timings["infer_ms"] = self._net.infer_ms
        self.timings["frames"] += 1
        heatmaps = min((np.asarray(v) for v in outputs.values()), key=lambda a: a.shape[1])[0]
        return self._decode(heatmaps, width, height)

    def _decode(self, heatmaps: np.ndarray, width: int, height: int):
        parts = {}
        map_h, map_w = heatmaps.shape[1:3]
        for index, name in enumerate(OPENPOSE_PART_NAMES):
            heat = heatmaps[index]
            flat = int(np.argmax(heat))
            score = float(heat.flat[flat])
            if score < self.min_score:
                continue
            y, x = divmod(flat, map_w)
            parts[name] = ((x + 0.5) / map_w * width, (y + 0.5) / map_h * height, score)
        return parts

    def close(self) -> None:
        self._net = None


# ── classification: ImageNet ─────────────────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageClassifier:
    """
    MobileNetV2 (default) or ResNet50 from the ONNX model zoo, ImageNet-1k.

    The frame's centre square is resized to 224x224 and normalised with the
    ImageNet mean/std inside the graph. `classify` returns the top `k`
    ``(label, probability)`` pairs. The 1000 class names are fetched once
    from the PyTorch hub list.
    """

    _MODELS = ("mobilenetv2", "resnet50")

    def __init__(self, model: str = "mobilenetv2", device: str | None = None,
                 top_k: int = 5):
        if model not in self._MODELS:
            raise ValueError(f"model must be one of {self._MODELS}, not {model!r}")
        path = model_path(model)
        self._net, self.device_name = _compile_with_fallback(
            lambda dev: _Net(path, dev, height=224, width=224, scale=255.0,
                             mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            device, model)
        self.device = self._net.device
        self.model = model
        self.top_k = int(top_k)
        self.compile_ms = self._net.compile_ms
        with open(model_path("imagenet-labels"), encoding="utf-8") as handle:
            self.labels = [line.strip() for line in handle if line.strip()]
        self.timings = {"infer_ms": 0.0, "frames": 0}

    def classify(self, rgb: np.ndarray) -> list[tuple[str, float]]:
        height, width = rgb.shape[:2]
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        cv2.resize(rgb[top:top + side, left:left + side], (224, 224),
                   dst=self._net.input[0], interpolation=cv2.INTER_AREA)
        logits = next(iter(self._net.infer().values())).reshape(-1).astype(np.float64)
        self.timings["infer_ms"] = self._net.infer_ms
        self.timings["frames"] += 1
        return self._decode(logits)

    def _decode(self, logits: np.ndarray) -> list[tuple[str, float]]:
        shifted = np.exp(logits - logits.max())
        probs = shifted / shifted.sum()
        order = np.argsort(-probs)[: self.top_k]
        return [(self.labels[i] if i < len(self.labels) else str(i), float(probs[i]))
                for i in order]

    def close(self) -> None:
        self._net = None
