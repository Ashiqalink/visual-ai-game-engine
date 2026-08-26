"""
bench_stage_costs.py — what each per-frame pipeline stage costs, next to the
inference it rides along with.

``VisionPipeline`` puts ~55 keys on every payload. A given game reads a handful:
Sling reads eight. The recurring question is therefore "should the stages behind
the unread keys be stripped out", and it is a question that answers itself wrong
from source — a class with a deque and a detrend loop in it *looks* expensive
next to a dict literal, and is not.

So this measures rather than argues. It exists to keep that decision anchored:
if the total for the unread stages is a low single-digit percentage of the
detector cost, removing them is a clarity change and must be argued as one, not
as a performance change.

Unlike the rest of ``benchmarks/`` there is no ground truth here and nothing to
pass or fail — it is a cost profile, so it prints a table and exits 0. It is not
part of ``run_all.py`` for that reason, the same way ``bench_hand_backend.py``
is not.

Stages are driven with real landmarks pulled from ``fixtures/hand_motion.mp4``,
not synthetic points: ``_classify_fingers`` branches on finger geometry and the
jitter window's detrend depends on how much the samples actually move, so made
up landmarks would measure the wrong branches.

Run it::

    python benchmarks/bench_stage_costs.py
"""

from __future__ import annotations

import queue
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import bootstrap, dim, header, table  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hand_motion.mp4"
SIZE = (640, 480)
#: Timed iterations per stage. These run in microseconds, so the sample count
#: matters more than it does for the detector benches.
REPS = 400
#: How many real landmark sets to cycle through, so no stage sees the same
#: geometry every iteration and branch prediction flatters it.
LANDMARK_FRAMES = 60

#: Stages whose output no shipped game reads today. Summed at the end against
#: the detector cost, which is the only number that makes the total meaningful.
UNREAD_BY_SLING = ("JitterAnalyzer.update", "DepthStabilizer.tick",
                   "PipelineNoiseFilter (off)")


def _capture_landmarks(limit=LANDMARK_FRAMES):
    """Run MediaPipe over the clip once and keep the landmark sets it found."""
    import mediapipe as mp

    det = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, model_complexity=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.65)
    cap = cv2.VideoCapture(str(FIXTURE))
    landmarks, frames = [], []
    try:
        while len(landmarks) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, SIZE)
            cv2.flip(frame, 1, dst=frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = det.process(rgb)
            if res.multi_hand_landmarks:
                landmarks.append(res.multi_hand_landmarks[0].landmark)
                frames.append(frame)
    finally:
        cap.release()
        det.close()
    return landmarks, frames


def _bench(fn, reps=REPS) -> tuple[float, float]:
    """Median and mean microseconds per call."""
    fn()  # warm: first call pays import-time and allocation costs
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1e6)
    return statistics.median(samples), statistics.mean(samples)


def main() -> int:
    if not FIXTURE.is_file():
        raise SystemExit(f"missing fixture: {FIXTURE}")

    bootstrap()
    from visual_ai.jitter_analyzer import JitterAnalyzer
    from visual_ai.low_light import LowLightBoost
    from visual_ai.noise_filter import OneEuroFilter, PipelineNoiseFilter
    from visual_ai.pipeline import VisionPipeline

    landmarks, frames = _capture_landmarks()
    if not landmarks:
        raise SystemExit("no landmarks found in the fixture — cannot drive the stages")

    print(header("Per-frame stage costs",
                 f"{FIXTURE.name} · {len(landmarks)} real landmark sets · "
                 f"{REPS} timed calls per stage"))

    pipeline = VisionPipeline(result_queue=queue.Queue(maxsize=1),
                              width=SIZE[0], height=SIZE[1],
                              detect_face=False, max_hands=1)
    frame = frames[0]
    cursor = {"i": 0}

    def next_landmarks():
        cursor["i"] = (cursor["i"] + 1) % len(landmarks)
        return landmarks[cursor["i"]]

    jitter = JitterAnalyzer(window_size=30)
    noise = PipelineNoiseFilter(noise_duration=0.0)
    one_euro = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.006)
    low_light = LowLightBoost()
    payload = pipeline._empty_payload(320.0, 240.0, frame)

    stages = [
        ("_extract_gesture (whole)",
         lambda: pipeline._extract_gesture(next_landmarks(), slot=0, now=time.time())),
        ("  _classify_fingers",
         lambda: pipeline._classify_fingers(next_landmarks(), pipeline._gs_slots[0])),
        ("  JitterAnalyzer.update",
         lambda: jitter.update((320.0, 240.0), (321.0, 241.0))),
        ("  OneEuroFilter.filter",
         lambda: one_euro.filter((320.0, 240.0), time.time())),
        ("LowLightBoost.apply (bright)", lambda: low_light.apply(frame)),
        ("DepthStabilizer.tick", pipeline.tof_stabilizer.tick),
        ("PipelineNoiseFilter (off)", lambda: noise.process_payload(payload)),
        ("_empty_payload build",
         lambda: pipeline._empty_payload(320.0, 240.0, frame)),
    ]

    measured = {}
    rows = []
    for label, fn in stages:
        median, mean = _bench(fn)
        measured[label.strip()] = median
        rows.append([label, f"{median:.1f}", f"{mean:.1f}", f"{median / 1000.0:.3f}"])

    print(table(["stage", "median us", "mean us", "ms"], rows, aligns="lrrr"))
    print(dim("  indented rows are components of _extract_gesture above them"))

    unread = sum(measured[name] for name in UNREAD_BY_SLING)
    print(f"\n  stages no shipped game reads : {unread:.1f} us "
          f"= {unread / 1000.0:.3f} ms/frame")
    print(dim("  (JitterAnalyzer's output is read by the tracker_report tools, "
              "just not in play)"))
    print("\n  Run bench_max_hands.py for the detector cost to weigh this against —\n"
          "  it is measured on this machine, and it is the number that dominates:\n"
          "  the detector is timed in milliseconds, everything above in microseconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
