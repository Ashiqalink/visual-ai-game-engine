"""
openvino_hands.py — MediaPipe Hands, run on the NPU or the iGPU.

MediaPipe's Python `solutions.hands` is CPU-only on Windows: the TFLite
interpreter it builds has no GPU delegate there and no NPU path at all, and no
flag changes that. The weights, though, are plain `.tflite` files inside the
installed `mediapipe` package, and OpenVINO reads TFLite directly — so this
module rebuilds the *graph* around those same two networks and hands them to
whichever device `accel.resolve` picked.

The graph is MediaPipe's own, in the same order:

    letterbox to 192x192 -> palm detection -> SSD decode over 2016 anchors
    -> weighted NMS -> rotated square ROI from palm keypoints 0 and 2
    -> 224x224 hand landmark -> next frame's ROI from landmarks 0 and 9

The detector only re-runs when fewer than `max_num_hands` are being tracked,
which is what keeps the per-frame cost at one small network most of the time.
When a hand is already being tracked and the detector re-runs to look for
another, it is submitted asynchronously and read back on a later frame, so that
frame is blocked only for the landmark network. Acquiring the *first* hand
still blocks: there is no landmark work to overlap with then, and a caller that
consumes frames faster than the device detects would never see the detection
land. Pass `async_detector=False` to block in both cases.

`process()` returns the same shape `mediapipe.solutions.hands.Hands.process()`
does — `.multi_hand_landmarks[i].landmark[j].x/.y/.z` normalised the same way,
`.multi_handedness[i].classification[0].label/.score` — so `pipeline.py` reads
one or the other through identical code.

It takes one argument MediaPipe's does not: an optional `face_box`. The palm
network fires on faces, and a false hand costs more here than a missed one,
because a rect held on a face is a tracking slot the player's second hand never
gets — see `_FACE_GATE`. `pipeline.py` has already detected the face by the
time it calls this, so it passes the box along; a caller that does not gets
exactly the old behaviour.

Two constants below were fitted rather than read. MediaPipe ships its graph as
`hand_landmark_tracking_cpu.binarypb` with the subgraphs unexpanded (they are
registered in C++), so the palm network's input range and the landmark->ROI
transform are not recoverable from the install. Both were settled against
MediaPipe's own output on `benchmarks/fixtures/hand_motion.mp4`: the input
range is 0..1 (-1..1 produces logits in the tens of thousands), and
`_LAND_SCALE`/`_LAND_SHIFT_Y` are what reproduce MediaPipe's detection rate on
that clip — 30% of frames, against 20% for the textbook 2.0/-0.1.

Accuracy against MediaPipe on that clip, index fingertip, 640x480: median 4.4
px, mean offset (-1.4, -3.5) px. For scale, MediaPipe's own two model tiers
disagree by 7.8 px rmse on the same frames.

**Untested in real play.** Every number here comes from a recorded clip. Live
capture, motion blur and re-acquisition feel need a human at the webcam.
"""
from __future__ import annotations

import math
import os
import time

import cv2
import numpy as np

from visual_ai import accel

_PALM_SIZE = 192
_LAND_SIZE = 224
_NUM_PALM_KP = 7

#: Palm detection ROI, from MediaPipe's palm_detection_detection_to_roi.
_PALM_SCALE, _PALM_SHIFT_Y = 2.6, -0.5
#: Landmark ROI — fitted; see the module docstring.
_LAND_SCALE, _LAND_SHIFT_Y = 1.8, -0.2

#: Two ROIs this close are the same hand, so the detector's find is dropped
#: rather than tracked twice. MediaPipe's AssociationNormRect threshold.
_ASSOCIATION_IOU = 0.5

#: A candidate whose landmarks sit on the detected face has to clear this
#: instead of the ordinary gates. The palm network fires on a face often
#: enough to matter, and a false hand costs far more here than a missed one:
#: it holds a tracking rect, and `process` only re-runs the detector while
#: fewer than `max_num_hands` rects are held — so one face permanently
#: occupies the slot the player's second hand needed. `min_tracking_confidence`
#: is 0.45 precisely so a real hand is not dropped, which is also low enough
#: for a face to survive indefinitely.
#:
#: A hand held *over* the face is the case this must not break, and presence is
#: what separates them: the landmark network reports a real hand in the crop
#: near 1.0 whether or not there is a face behind it, while a crop containing
#: only a face sits just over the loose gate. The strict gate applies only
#: inside the face box, so nothing outside it changes.
_FACE_GATE = 0.90

#: The face box is grown by this much before the containment test. The box
#: MediaPipe's FaceDetection returns is tight to the eyes and mouth, and the
#: hand the palm network hallucinates from a face spans the whole head.
_FACE_BOX_DILATE = 1.15


class OpenVINOHandsUnavailable(RuntimeError):
    """Raised when the accelerated path cannot be built. Callers fall back."""


# ── MediaPipe's SSD anchors ───────────────────────────────────────────────────

def _ssd_anchors(input_size: int = _PALM_SIZE, num_layers: int = 4,
                 strides: tuple[int, ...] = (8, 16, 16, 16),
                 offset: float = 0.5) -> np.ndarray:
    """
    The 2016 anchor centres the palm network's box regressions are relative to.

    MediaPipe's SsdAnchorsCalculator with `fixed_anchor_size` and an
    interpolated scale aspect ratio, which together reduce to "two anchors per
    feature-map cell, at the cell centre" — the box sizes the anchors would
    otherwise carry are unused because the network regresses absolute w/h.
    """
    anchors: list[tuple[float, float]] = []
    layer = 0
    while layer < num_layers:
        repeats = 0
        index = layer
        while index < num_layers and strides[index] == strides[layer]:
            repeats += 2
            index += 1
        feature_map = int(math.ceil(input_size / strides[layer]))
        for y in range(feature_map):
            for x in range(feature_map):
                for _ in range(repeats):
                    anchors.append(((x + offset) / feature_map,
                                    (y + offset) / feature_map))
        layer = index
    return np.asarray(anchors, dtype=np.float32)


_ANCHORS = _ssd_anchors()


# ── Geometry ──────────────────────────────────────────────────────────────────

def _letterbox(rgb: np.ndarray, size: int):
    """Aspect-preserving fit into a square, zero-padded — MediaPipe's ImageToTensor."""
    height, width = rgb.shape[:2]
    scale = size / max(height, width)
    new_h, new_w = int(round(height * scale)), int(round(width * scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, left, top


def _iou(a: tuple[float, float, float, float], b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _face_bounds(face_box) -> tuple[float, float, float, float] | None:
    """`(x, y, w, h)` in pixels -> dilated `(x0, y0, x1, y1)`, or None."""
    if not face_box:
        return None
    x, y, w, h = (float(v) for v in face_box)
    if w <= 0.0 or h <= 0.0:
        return None
    grow_x = w * (_FACE_BOX_DILATE - 1.0) / 2.0
    grow_y = h * (_FACE_BOX_DILATE - 1.0) / 2.0
    return (x - grow_x, y - grow_y, x + w + grow_x, y + h + grow_y)


def _inside(x: float, y: float, bounds) -> bool:
    return bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]


def _nms_weighted(detections: list[dict], threshold: float = 0.3) -> list[dict]:
    """
    MediaPipe's WEIGHTED non-max suppression.

    Not plain NMS: a survivor is the score-weighted *mean* of its whole
    cluster, which is what stops the ROI jumping between two near-identical
    anchors on consecutive frames.
    """
    if not detections:
        return []
    remaining = sorted(detections, key=lambda d: -d["score"])
    kept: list[dict] = []
    while remaining:
        best = remaining[0]
        cluster, rest = [best], []
        for other in remaining[1:]:
            (cluster if _iou(best["box"], other["box"]) > threshold else rest).append(other)
        weight = sum(c["score"] for c in cluster)
        kept.append({
            "score": best["score"],
            "kp": sum(c["kp"] * c["score"] for c in cluster) / weight,
            "box": tuple(sum(np.asarray(c["box"]) * c["score"] for c in cluster) / weight),
        })
        remaining = rest
    return kept


def _rect_from_box(box, start, end, scale: float, shift_y: float):
    """
    MediaPipe's DetectionsToRects + RectTransformation, in pixels.

    The rect is centred on the detection box, rotated so `start`->`end` points
    at 90 degrees, shifted along the rotated axes against the *pre-scale* box
    size, then squared on its long side and scaled — the order the calculator
    applies them in, and the reason a shift written as a fraction of the final
    size lands somewhere else entirely.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    box_w, box_h = x1 - x0, y1 - y0
    angle = math.pi * 0.5 - math.atan2(-(end[1] - start[1]), end[0] - start[0])
    angle -= 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))
    if shift_y:
        cx += (-shift_y * math.sin(angle)) * box_w
        cy += (shift_y * math.cos(angle)) * box_h
    return cx, cy, max(box_w, box_h) * scale, angle


def _rect_bounds(rect) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of a rotated rect, for association only."""
    cx, cy, size, _ = rect
    half = size / 2.0
    return cx - half, cy - half, cx + half, cy + half


def _crop(rgb: np.ndarray, rect, out_size: int):
    """Rotated crop to a square, plus the affine that maps landmarks back."""
    cx, cy, size, angle = rect
    half = size / 2.0
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    source = np.float32([
        [cx - half * cos_a + half * sin_a, cy - half * sin_a - half * cos_a],
        [cx + half * cos_a + half * sin_a, cy + half * sin_a - half * cos_a],
        [cx - half * cos_a - half * sin_a, cy - half * sin_a + half * cos_a],
    ])
    target = np.float32([[0, 0], [out_size, 0], [0, out_size]])
    forward = cv2.getAffineTransform(source, target)
    crop = cv2.warpAffine(rgb, forward, (out_size, out_size), flags=cv2.INTER_LINEAR)
    return crop, cv2.invertAffineTransform(forward)


# ── MediaPipe-shaped results ──────────────────────────────────────────────────

class _Landmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z


class _LandmarkList:
    __slots__ = ("landmark",)

    def __init__(self, landmarks: list[_Landmark]):
        self.landmark = landmarks


class _Classification:
    __slots__ = ("label", "score", "index")

    def __init__(self, label: str, score: float, index: int):
        self.label, self.score, self.index = label, score, index


class _ClassificationList:
    __slots__ = ("classification",)

    def __init__(self, classification: list[_Classification]):
        self.classification = classification


class _Result:
    __slots__ = ("multi_hand_landmarks", "multi_handedness")

    def __init__(self, landmarks, handedness):
        self.multi_hand_landmarks = landmarks
        self.multi_handedness = handedness


# ── The tracker ───────────────────────────────────────────────────────────────

def _model_paths(tier: str) -> tuple[str, str]:
    """The two `.tflite` files inside the installed mediapipe package."""
    try:
        import mediapipe
    except Exception as exc:                                  # pragma: no cover
        raise OpenVINOHandsUnavailable(
            f"mediapipe is not installed, so its hand weights are not on disk: {exc}") from exc
    modules = os.path.join(os.path.dirname(mediapipe.__file__), "modules")
    palm = os.path.join(modules, "palm_detection", f"palm_detection_{tier}.tflite")
    land = os.path.join(modules, "hand_landmark", f"hand_landmark_{tier}.tflite")
    for path in (palm, land):
        if not os.path.exists(path):
            raise OpenVINOHandsUnavailable(f"missing hand weights: {path}")
    return palm, land


class OpenVINOHands:
    """
    A drop-in for ``mediapipe.solutions.hands.Hands`` that runs on `device`.

    Construction compiles both networks, which is where an absent or busy
    device fails — deliberately, so the caller can fall back at startup rather
    than on the first frame.
    """

    #: How `pipeline.py` knows it may hand `process()` a face box. Asked of the
    #: object every frame rather than settled at construction, so a swapped-in
    #: backend (a test double, MediaPipe's own) is never called with an
    #: argument it does not take.
    accepts_face_box = True

    def __init__(self, device: str = "NPU", max_num_hands: int = 2,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.7,
                 # Looser than MediaPipe's 0.65 on purpose. The presence scalar
                 # comes back lower here than from the CPU graph — both the NPU
                 # (FP16) and the iGPU exhibit this — so 0.65 drops a hand that
                 # is still plainly in frame, and every drop costs a full palm
                 # re-detection. Sling's 2026-08-25 --accel npu logs show the
                 # cost: 6-11 tracking dropouts a run against 2-5 on CPU, and
                 # four shots fired with cause 'lost' (none on CPU) because the
                 # hand vanished mid-pull. dGPU behavior at 0.45 is untested.
                 min_tracking_confidence: float = 0.45,
                 face_gate: float = _FACE_GATE,
                 async_detector: bool = True):
        try:
            import openvino as ov
        except Exception as exc:
            raise OpenVINOHandsUnavailable(
                f"openvino is not installed: {exc}") from exc

        tier = "full" if int(model_complexity) >= 1 else "lite"
        palm_path, land_path = _model_paths(tier)
        try:
            core = ov.Core()
            hint = {"PERFORMANCE_HINT": "LATENCY"}
            self._palm = core.compile_model(
                core.read_model(palm_path), device, hint).create_infer_request()
            self._land = core.compile_model(
                core.read_model(land_path), device, hint).create_infer_request()
        except Exception as exc:
            raise OpenVINOHandsUnavailable(
                f"could not compile the hand networks for {device}: {exc}") from exc

        self.device = device
        self.tier = tier
        self.max_num_hands = max(1, int(max_num_hands))
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.face_gate = float(face_gate)
        self.async_detector = bool(async_detector)
        self._rects: list[tuple[float, float, float, float]] = []
        #: A detection in flight, as (input tensor, letterbox scale, left, top).
        self._pending: tuple | None = None
        #: Candidates dropped by the face gate, cumulative. A player who cannot
        #: get a second hand tracked needs to know whether their face is eating
        #: the slot, and this is the only place that is visible.
        self.face_rejects = 0
        #: Per-stage timings, so a HUD or bench can see where the frame went.
        self.timings = {"palm_ms": 0.0, "land_ms": 0.0, "detector_runs": 0, "frames": 0}

    # ── networks ──────────────────────────────────────────────────────────────

    def _start_palms(self, rgb: np.ndarray) -> None:
        """Submit one palm detection. `_finish_palms` decodes it, later."""
        letterboxed, scale, left, top = _letterbox(rgb, _PALM_SIZE)
        # 0..1, not -1..1: see the module docstring.
        tensor = (letterboxed.astype(np.float32) / 255.0)[None]

        started = time.perf_counter()
        self._palm.start_async([tensor])
        self.timings["palm_ms"] += (time.perf_counter() - started) * 1e3
        self.timings["detector_runs"] += 1
        # `tensor` is the device's input buffer until the request completes, so
        # it is held here rather than left for the garbage collector.
        self._pending = (tensor, scale, left, top)

    def _finish_palms(self, block: bool) -> list[dict] | None:
        """
        The submitted detection's boxes in pixels, or None when `block` is
        False and the device has not finished with it yet.
        """
        _, scale, left, top = self._pending

        started = time.perf_counter()
        if not block and not self._palm.wait_for(0):
            self.timings["palm_ms"] += (time.perf_counter() - started) * 1e3
            return None
        self._palm.wait()
        outputs = self._palm.results
        self.timings["palm_ms"] += (time.perf_counter() - started) * 1e3
        self._pending = None

        raw = [np.asarray(value) for value in outputs.values()]
        boxes = raw[0][0] if raw[0].shape[-1] == 18 else raw[1][0]
        logits = raw[1][0] if raw[0].shape[-1] == 18 else raw[0][0]
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -100.0, 100.0)))

        keep = scores >= self.min_detection_confidence
        detections: list[dict] = []
        if keep.any():
            regressed = boxes[keep] / _PALM_SIZE
            anchors = _ANCHORS[keep]
            kept_scores = scores[keep]
            cx = regressed[:, 0] + anchors[:, 0]
            cy = regressed[:, 1] + anchors[:, 1]
            w, h = regressed[:, 2], regressed[:, 3]
            keypoints = (regressed[:, 4:4 + _NUM_PALM_KP * 2]
                         .reshape(-1, _NUM_PALM_KP, 2) + anchors[:, None, :])
            for i in range(len(anchors)):
                detections.append({
                    "score": float(kept_scores[i]),
                    "kp": keypoints[i],
                    "box": (cx[i] - w[i] / 2, cy[i] - h[i] / 2,
                            cx[i] + w[i] / 2, cy[i] + h[i] / 2),
                })

        detections = _nms_weighted(detections)[: self.max_num_hands]
        for detection in detections:          # letterbox space -> pixels
            detection["kp"] = np.stack([
                (detection["kp"][:, 0] * _PALM_SIZE - left) / scale,
                (detection["kp"][:, 1] * _PALM_SIZE - top) / scale], axis=1)
            x0, y0, x1, y1 = detection["box"]
            detection["box"] = ((x0 * _PALM_SIZE - left) / scale,
                                (y0 * _PALM_SIZE - top) / scale,
                                (x1 * _PALM_SIZE - left) / scale,
                                (y1 * _PALM_SIZE - top) / scale)
        return detections

    def _landmarks(self, rgb: np.ndarray, rect) -> dict:
        crop, inverse = _crop(rgb, rect, _LAND_SIZE)
        tensor = (crop.astype(np.float32) / 255.0)[None]

        started = time.perf_counter()
        outputs = self._land.infer([tensor])
        self.timings["land_ms"] += (time.perf_counter() - started) * 1e3

        # Four same-named outputs; sort by port name so the mapping is fixed:
        # the 63-wide pair is screen then world landmarks, the 1-wide pair is
        # hand presence then handedness.
        by_size: dict[int, list] = {}
        for port, value in outputs.items():
            array = np.asarray(value)
            by_size.setdefault(array.size, []).append((port.get_any_name(), array))
        landmarks = sorted(by_size[63], key=lambda kv: kv[0])[0][1].reshape(21, 3)
        scalars = sorted(by_size[1], key=lambda kv: kv[0])
        presence = float(scalars[0][1].ravel()[0])
        handedness = float(scalars[1][1].ravel()[0])

        points = landmarks[:, :2]
        pixels = np.stack([
            inverse[0, 0] * points[:, 0] + inverse[0, 1] * points[:, 1] + inverse[0, 2],
            inverse[1, 0] * points[:, 0] + inverse[1, 1] * points[:, 1] + inverse[1, 2],
        ], axis=1)
        # MediaPipe scales z like x — by the image width — so a consumer's
        # z thresholds mean the same thing on either backend.
        depth = landmarks[:, 2] * (rect[2] / _LAND_SIZE)
        return {"pixels": pixels, "z": depth, "presence": presence,
                "handedness": handedness}

    def _adopt(self, detections: list[dict], face=None) -> None:
        """Turn fresh palm boxes into tracking rects, skipping hands already held."""
        for detection in detections:
            if len(self._rects) >= self.max_num_hands:
                break
            rect = _rect_from_box(detection["box"], detection["kp"][0],
                                  detection["kp"][2], _PALM_SCALE, _PALM_SHIFT_Y)
            bounds = _rect_bounds(rect)
            if any(_iou(bounds, _rect_bounds(known)) > _ASSOCIATION_IOU
                   for known in self._rects):
                continue                          # already tracking this hand
            # Never adopt a marginal detection that is sitting on the face: the
            # rect it would take is the one the second hand needs, and holding
            # it also stops the detector re-running to find that hand.
            if (face is not None and detection["score"] < self.face_gate
                    and _inside(rect[0], rect[1], face)):
                self.face_rejects += 1
                continue
            self._rects.append(rect)

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, rgb: np.ndarray, face_box=None):
        """
        One frame in, a MediaPipe-shaped result out. `rgb` is HxWx3 uint8 RGB.

        `face_box` is the frame's detected face as `(x, y, w, h)` in the same
        pixels, or None. Given one, a candidate sitting on the face has to
        clear `face_gate` rather than the ordinary confidences — see the
        `_FACE_GATE` note. The argument is optional so the call site can stay
        identical to MediaPipe's.
        """
        height, width = rgb.shape[:2]
        face = _face_bounds(face_box)
        self.timings["frames"] += 1

        # The detector is left in flight only while a hand is already being
        # tracked. That is the only time the frame has other work — the
        # landmark network — to hide it behind, and it keeps acquisition
        # exactly as prompt as the blocking path: a caller that consumes
        # frames faster than the device detects, an offline clip or a loop
        # catching up after a drop, would otherwise never acquire a hand at
        # all, since the detection it is waiting on is never given the time
        # to land.
        if self._pending is not None:
            detections = self._finish_palms(
                block=not (self.async_detector and self._rects))
            if detections is not None:
                self._adopt(detections, face)
        if self._pending is None and len(self._rects) < self.max_num_hands:
            self._start_palms(rgb)
            if not (self.async_detector and self._rects):
                self._adopt(self._finish_palms(block=True), face)

        surviving: list[tuple[float, float, float, float]] = []
        landmark_lists: list[_LandmarkList] = []
        handedness_lists: list[_ClassificationList] = []

        for rect in self._rects:
            hand = self._landmarks(rgb, rect)
            if hand["presence"] < self.min_tracking_confidence:
                continue                          # tracking lost; detector re-runs
            pixels, depth = hand["pixels"], hand["z"]
            # Dropping the rect rather than only hiding the hand is the point:
            # a face kept as a rect is a slot the second hand never gets, and
            # the detector does not re-run while the slot is held.
            if (face is not None and hand["presence"] < self.face_gate
                    and _inside(float(pixels[:, 0].mean()),
                                float(pixels[:, 1].mean()), face)):
                self.face_rejects += 1
                continue
            landmark_lists.append(_LandmarkList([
                _Landmark(float(x) / width, float(y) / height, float(z) / width)
                for (x, y), z in zip(pixels, depth)
            ]))
            # The landmark network's second scalar is P(right hand) on a frame
            # in selfie orientation, which is the orientation pipeline.py has
            # already flipped into by the time it calls this.
            score = hand["handedness"]
            label = "Right" if score >= 0.5 else "Left"
            handedness_lists.append(_ClassificationList([
                _Classification(label, score if label == "Right" else 1.0 - score, 0)]))
            surviving.append(_rect_from_box(
                (float(pixels[:, 0].min()), float(pixels[:, 1].min()),
                 float(pixels[:, 0].max()), float(pixels[:, 1].max())),
                pixels[0], pixels[9], _LAND_SCALE, _LAND_SHIFT_Y))

        self._rects = surviving
        return _Result(landmark_lists or None, handedness_lists or None)

    def close(self) -> None:
        """Match MediaPipe's API. The infer requests are freed with the object."""
        if self._pending is not None:     # never free an in-flight input buffer
            self._palm.wait()
            self._pending = None
        self._rects = []


def build(device: str | None = None, **kwargs) -> OpenVINOHands | None:
    """
    An `OpenVINOHands` on the resolved device, or None to stay on MediaPipe.

    Never raises: an accelerator that cannot be reached is a fallback the
    caller reports, not a crash on startup.
    """
    resolved = accel.resolve("hand", explicit=device)
    if not resolved:
        return None
    try:
        return OpenVINOHands(device=resolved, **kwargs)
    except OpenVINOHandsUnavailable as exc:
        print(f"[OpenVINOHands] {exc}")
        return None
    except Exception as exc:                                   # pragma: no cover
        print(f"[OpenVINOHands] unexpected failure on {resolved}: {exc}")
        return None
