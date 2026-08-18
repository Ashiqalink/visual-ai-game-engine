"""
bench_resolution.py — Virtual test for MediaPipe input-resolution / frame-skip
trade-offs, against a recorded clip instead of a live camera.

``pipeline.py`` resizes every frame to ``width x height`` *before* handing it to
MediaPipe Hands + FaceDetection (see ``_process_frame``), and runs both models on
every single frame — there is no skip-frame path today. Profiling showed those
two ``.process()`` calls are essentially the entire per-frame cost. This bench
answers "how many ms would downscaling or frame-skipping actually save, and
what detection accuracy would that cost" *before* changing ``pipeline.py``,
using ``fixtures/hand_motion.mp4`` (a real recorded clip — MediaPipe accuracy
can't be synthesized the way filter streams can) as a fixed input replayed at
each candidate config.

Ground truth is MediaPipe Hands run once at the clip's native resolution, every
frame. Every other config is scored against that reference index-fingertip
(landmark 8) trajectory — frames where the reference itself found no hand are
excluded from scoring on both sides.

Run it::

    python benchmarks/bench_resolution.py
    python benchmarks/bench_resolution.py --html report.html
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import BenchResult, Scenario  # noqa: E402
import metrics  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hand_motion.mp4"
INDEX_FINGERTIP = 8  # MediaPipe Hands landmark id

#: (label, target_width_or_None, skip_n). target_width=None keeps native res.
#: skip_n=1 means every frame is detected (no skip).
_CONFIGS = [
    ("native 1280x720",   None, 1),
    ("downscale 960x540", 960,  1),
    ("downscale 640x360", 640,  1),
    ("downscale 480x270", 480,  1),
    ("skip-2 + interp (native)", None, 2),
    ("skip-3 + interp (native)", None, 3),
]

_REFERENCE_LABEL = "native 1280x720"


def _load_frames() -> list[np.ndarray]:
    if not FIXTURE.exists():
        raise FileNotFoundError(
            f"{FIXTURE} not found — run benchmarks/fixtures/record_clip.py first")
    cap = cv2.VideoCapture(str(FIXTURE))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _resize(frame: np.ndarray, target_width: int | None) -> tuple[np.ndarray, float]:
    """Returns the (possibly) resized frame and the scale factor to map its
    landmark coordinates back to the native frame size."""
    if target_width is None or target_width >= frame.shape[1]:
        return frame, 1.0
    scale = frame.shape[1] / target_width
    target_height = int(round(frame.shape[0] / scale))
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def _detect_fingertip(hands, frame: np.ndarray) -> tuple[float, float] | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark[INDEX_FINGERTIP]
    h, w = frame.shape[:2]
    return (lm.x * w, lm.y * h)


def _interpolate_gaps(points: list[tuple[float, float] | None]) -> list[tuple[float, float] | None]:
    """Linear interpolation across skipped frames, between the two nearest
    detections. Leading/trailing gaps (before the first / after the last
    detection) stay ``None`` — there is nothing to interpolate from."""
    out = list(points)
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        start = i - 1
        j = i
        while j < n and out[j] is None:
            j += 1
        if start < 0 or j >= n:
            i = j
            continue
        x0, y0 = out[start]
        x1, y1 = out[j]
        span = j - start
        for k in range(start + 1, j):
            t = (k - start) / span
            out[k] = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        i = j
    return out


def _run_config(frames: list[np.ndarray], target_width: int | None, skip_n: int) -> dict:
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    raw_points: list[tuple[float, float] | None] = [None] * len(frames)
    process_us: list[float] = []

    for i, frame in enumerate(frames):
        if i % skip_n != 0:
            continue
        resized, scale = _resize(frame, target_width)
        start = time.perf_counter()
        pt = _detect_fingertip(hands, resized)
        process_us.append((time.perf_counter() - start) * 1e6)
        if pt is not None:
            pt = (pt[0] * scale, pt[1] * scale)
        raw_points[i] = pt

    hands.close()

    points = _interpolate_gaps(raw_points) if skip_n > 1 else raw_points
    per_call_ms = float(np.median(process_us)) / 1000.0 if process_us else 0.0
    return {
        "points": points,
        # Amortized over every original frame, not just the ones .process() ran
        # on — a skip-2 config pays the full cost on half the frames and zero
        # on the rest, so the effective per-frame budget is roughly halved.
        "ms_per_frame": per_call_ms / skip_n,
        "detected": sum(1 for p in raw_points if p is not None),
        "total": len(frames),
    }


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def run() -> BenchResult:
    result = BenchResult(
        name="MediaPipe resolution / frame-skip trade-off",
        subtitle=f"replaying {FIXTURE.relative_to(FIXTURE.parents[2])} — "
                 "recorded clip, not synthetic (MediaPipe accuracy can't be faked)",
    )

    frames = _load_frames()
    native_w = frames[0].shape[1]
    native_h = frames[0].shape[0]

    runs = {label: _run_config(frames, width, skip)
            for label, width, skip in _CONFIGS}

    reference = runs[_REFERENCE_LABEL]
    ref_points = reference["points"]
    valid_idx = [i for i, p in enumerate(ref_points) if p is not None]

    def series_for(points: list) -> tuple[np.ndarray, np.ndarray]:
        """Aligned (candidate, reference) arrays over frames the reference detected."""
        cand = np.array([points[i] if points[i] is not None else (np.nan, np.nan)
                         for i in valid_idx])
        ref = np.array([ref_points[i] for i in valid_idx])
        mask = ~np.isnan(cand).any(axis=1)
        return cand[mask], ref[mask]

    baseline_ms = reference["ms_per_frame"]
    comparison_rows = []
    scen = Scenario(
        name="fingertip tracking vs native-resolution reference",
        description=(f"{len(frames)} frames, hand detected in {reference['detected']}/"
                     f"{len(frames)} reference frames ({100*reference['detected']/len(frames):.0f}%). "
                     "rmse/lag scored only on frames the reference itself detected."),
        y_label="px",
    )

    for label, width, skip in _CONFIGS:
        run_data = runs[label]
        cand, ref = series_for(run_data["points"])
        coverage = len(cand) / max(1, len(valid_idx)) * 100.0

        if len(cand) >= 8:
            rmse = metrics.rmse(cand, ref)
            lag = metrics.lag_ms(cand, ref, 1.0 / 30.0)
        else:
            rmse, lag = float("nan"), float("nan")

        ms_saved_pct = (1.0 - run_data["ms_per_frame"] / baseline_ms) * 100.0 if baseline_ms else 0.0

        comparison_rows.append([
            label,
            _fmt(run_data["ms_per_frame"]),
            f"{ms_saved_pct:+.0f}%" if label != _REFERENCE_LABEL else "—",
            _fmt(rmse) if np.isfinite(rmse) else "n/a",
            _fmt(lag, 1) if np.isfinite(lag) else "n/a",
            _fmt(coverage, 0),
        ])

    scen.metrics = {"reference_ms_per_frame": baseline_ms}
    scen.traces = {
        label: [p[0] if p is not None else float("nan") for p in runs[label]["points"]]
        for label in runs
    }
    result.scenarios.append(scen)

    result.tables.append({
        "title": "Config comparison (fingertip x/y, single-hand)",
        "note": ("ms/frame = median MediaPipe Hands .process() cost · "
                 "rmse/lag measured only where the native-res reference detected a hand · "
                 "coverage = % of those frames the candidate also produced a point for"),
        "headers": ["config", "ms/frame", "ms saved", "rmse px", "lag ms", "coverage %"],
        "rows": comparison_rows,
        "aligns": "lrrrrr",
    })

    return result


if __name__ == "__main__":
    from run_all import main
    raise SystemExit(main(["--only", "resolution", *sys.argv[1:]]))
