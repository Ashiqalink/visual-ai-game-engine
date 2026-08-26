"""
bench_max_hands.py — what an unused second hand slot costs, against a recorded
clip instead of a live camera.

MediaPipe Hands re-runs *palm detection* on every frame while the number of
hands it is currently tracking is below ``max_num_hands``. A one-hand game that
leaves ``VisionPipeline(max_hands=2)`` at its default therefore pays a palm
detection pass per frame hunting for a hand that is never in shot. This bench
measures that, so "declare your hand budget" is a number rather than an argument
from the MediaPipe source.

The control is what makes the result trustworthy. Frames where *no* hand is in
view run palm detection under either budget, so the two configs must cost the
same there — if they diverge, what is being measured is machine noise or thermal
drift rather than the empty slot, and the ``no-hand parity`` check fails. Frames
with a hand tracked are where the claim lives, and they are timed separately for
exactly that reason. Pooling the two buckets buries the effect: this clip has a
hand in only a quarter of its frames, which drags a pooled median down to a
fraction of the per-frame saving a game would actually feel.

Equivalence is measured alongside cost. A config that is cheap because it
stopped finding the hand is not a saving, so the index-fingertip trajectory and
the detection rate are both scored against the ``max_hands=2`` run.

Backends are measured as ``pipeline.py`` resolves them: OpenVINO if this machine
has it, otherwise MediaPipe's CPU graph. Pass ``--cpu`` to force the CPU path —
worth doing before shipping a change, since it is what a machine without an
Intel accelerator will run, and it is where the effect is largest.

Run it::

    python benchmarks/bench_max_hands.py
    python benchmarks/bench_max_hands.py --cpu
    python benchmarks/run_all.py --only hands
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import BenchResult, Scenario  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hand_motion.mp4"
INDEX_FINGERTIP = 8  # MediaPipe Hands landmark id

#: Frames dropped from the head of every pass. The first detections pay lazy
#: graph setup that has nothing to do with the hand budget.
WARMUP = 15
#: Passes per config, run A/B/A/B so thermal drift lands on both sides equally
#: rather than on whichever config happened to go last.
REPEATS = 3
#: The claim. Below this an empty slot is not worth a per-game override, and
#: this bench should fail rather than quietly report a rounding error.
MIN_SAVING_PCT = 5.0
#: The same fingertip tolerance ``bench_hand_backend`` uses: MediaPipe's own two
#: model tiers differ by ~7.8 px on this clip, so under 20 px is the same path.
MAX_RMSE_PX = 20.0
#: The two budgets must find the hand on essentially the same frames.
MAX_DETECT_DELTA_PP = 5.0
#: No-hand frames must cost the same under either budget — see the docstring.
MAX_PARITY_MS = 1.0

SIZE = (640, 480)


def _load(size=SIZE) -> list[np.ndarray]:
    """Replay the clip as RGB frames, prepared exactly as the pipeline does."""
    cap = cv2.VideoCapture(str(FIXTURE))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        cv2.flip(frame, 1, dst=frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # The pipeline marks its RGB buffer read-only so MediaPipe's wrapper can
        # skip a defensive copy. Timing a writeable frame would measure a copy
        # that never happens in play.
        rgb.flags.writeable = False
        frames.append(rgb)
    cap.release()
    return frames


def _build(max_hands: int, cpu_only: bool):
    """Build a detector the way ``VisionPipeline`` does: OpenVINO, then CPU."""
    from visual_ai import openvino_hands

    det = None if cpu_only else openvino_hands.build(
        device=None, max_num_hands=max_hands, model_complexity=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.45)
    if det is not None:
        return det, "OpenVINO"

    import mediapipe as mp
    return mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=max_hands, model_complexity=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.65), "MediaPipe CPU"


def _run(frames, max_hands: int, cpu_only: bool) -> dict:
    det, backend = _build(max_hands, cpu_only)
    hot, cold, tips = [], [], []
    try:
        for i, rgb in enumerate(frames):
            started = time.perf_counter()
            res = det.process(rgb)
            elapsed = (time.perf_counter() - started) * 1e3
            sets = res.multi_hand_landmarks or []
            if i < WARMUP:
                continue
            if sets:
                hot.append(elapsed)
                lm = sets[0].landmark[INDEX_FINGERTIP]
                tips.append((lm.x * SIZE[0], lm.y * SIZE[1]))
            else:
                cold.append(elapsed)
                tips.append(None)
    finally:
        if hasattr(det, "close"):
            det.close()
    return {"backend": backend, "hot": hot, "cold": cold, "tips": tips,
            "ratio": len(hot) / max(1, len(hot) + len(cold))}


def _rmse(a, b) -> float:
    """Fingertip distance over frames where both runs found a hand."""
    pairs = [(p, q) for p, q in zip(a, b) if p is not None and q is not None]
    if not pairs:
        return float("nan")
    return (sum((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                for p, q in pairs) / len(pairs)) ** 0.5


def run(cpu_only: bool = False) -> BenchResult:
    if not FIXTURE.is_file():
        raise SystemExit(f"missing fixture: {FIXTURE}")

    frames = _load()
    acc = {1: {"hot": [], "cold": []}, 2: {"hot": [], "cold": []}}
    last: dict[int, dict] = {}

    for _ in range(REPEATS):
        for budget in (1, 2):
            res = _run(frames, budget, cpu_only)
            acc[budget]["hot"].extend(res["hot"])
            acc[budget]["cold"].extend(res["cold"])
            last[budget] = res

    for budget in (1, 2):
        if not acc[budget]["hot"] or not acc[budget]["cold"]:
            raise SystemExit(
                f"max_hands={budget} produced no {'hand' if not acc[budget]['hot'] else 'empty'} "
                "frames — the clip cannot score this bench")

    backend = last[1]["backend"]
    hot1, hot2 = statistics.median(acc[1]["hot"]), statistics.median(acc[2]["hot"])
    cold1, cold2 = statistics.median(acc[1]["cold"]), statistics.median(acc[2]["cold"])
    saved_ms = hot2 - hot1
    saved_pct = saved_ms / hot2 * 100.0
    rmse = _rmse(last[1]["tips"], last[2]["tips"])
    detect_delta = (last[1]["ratio"] - last[2]["ratio"]) * 100.0

    result = BenchResult(
        name="Unused hand slot cost",
        subtitle=(f"{FIXTURE.name} · {len(frames)} frames @ {SIZE[0]}x{SIZE[1]} · "
                  f"{REPEATS} alternating A/B passes · {backend}"))

    scen = Scenario(
        name="max_hands=1 vs max_hands=2, one hand in shot",
        description=("MediaPipe re-runs palm detection while tracked hands < "
                     "max_num_hands, so a spare slot costs a detection pass per "
                     "frame. Timed separately for frames with and without a hand."),
        x_label="frame", y_label="ms")
    scen.notes = (f"hand in frame: {hot2:.2f} -> {hot1:.2f} ms · "
                  f"no hand: {cold2:.2f} vs {cold1:.2f} ms · "
                  f"clip has a hand in {last[2]['ratio'] * 100:.0f}% of frames")

    scen.check("per-frame saving, hand in shot", saved_pct, "%",
               min_value=MIN_SAVING_PCT,
               detail=f"{saved_ms:+.2f} ms ({hot2:.2f} -> {hot1:.2f})")
    scen.check("no-hand parity", abs(cold2 - cold1), "ms",
               max_value=MAX_PARITY_MS,
               detail="both budgets run palm detection here, so they must match")
    scen.check("fingertip agreement", rmse, "px",
               max_value=MAX_RMSE_PX,
               detail="index fingertip vs the max_hands=2 trajectory")
    scen.check("detection rate delta", abs(detect_delta), "pp",
               max_value=MAX_DETECT_DELTA_PP,
               detail=f"{last[1]['ratio'] * 100:.1f}% vs {last[2]['ratio'] * 100:.1f}%")

    scen.metrics = {"hot_ms_1": hot1, "hot_ms_2": hot2,
                    "cold_ms_1": cold1, "cold_ms_2": cold2,
                    "saved_ms": saved_ms, "rmse_px": rmse}
    scen.traces = {"max_hands=1": acc[1]["hot"][:len(frames)],
                   "max_hands=2": acc[2]["hot"][:len(frames)]}
    result.scenarios.append(scen)

    def _row(budget: int, hot: float, cold: float) -> list[str]:
        pooled = sorted(acc[budget]["hot"] + acc[budget]["cold"])
        return [f"max_hands={budget}", f"{hot:.2f}", f"{cold:.2f}",
                f"{statistics.median(pooled):.2f}",
                f"{pooled[int(len(pooled) * 0.95)]:.2f}",
                f"{last[budget]['ratio'] * 100:.1f}"]

    result.tables.append({
        "title": f"Per-frame detector cost ({backend})",
        "note": ("hand in frame = the claim · no hand = the control, both budgets "
                 "hunt there · the pooled median is diluted by this clip's low "
                 "hand coverage and is not what a game would feel"),
        "headers": ["config", "hand in frame ms", "no hand ms", "pooled median ms",
                    "pooled p95 ms", "detect %"],
        "rows": [_row(1, hot1, cold1), _row(2, hot2, cold2)],
        "aligns": "lrrrrr",
    })

    return result


if __name__ == "__main__":
    forwarded = [a for a in sys.argv[1:] if a != "--cpu"]
    if "--cpu" in sys.argv[1:]:
        # run_all calls run() with no arguments, so the CPU override has to ride
        # on the module attribute rather than through the call.
        import functools

        sys.modules[__name__].run = functools.partial(run, cpu_only=True)
    from run_all import main

    raise SystemExit(main(["--only", "hands", *forwarded]))
