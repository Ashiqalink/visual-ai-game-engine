"""
bench_hand_backend.py — MediaPipe Hands against the OpenVINO backend, on every
device this machine has.

The question is not "is the NPU fast". It is whether `visual_ai.openvino_hands`
tracks the same hand MediaPipe does while being cheaper, so the answer has to
carry both numbers: per-frame time *and* agreement with MediaPipe's own
fingertip trajectory on `fixtures/hand_motion.mp4`. A backend that is fast
because it stopped detecting shows up here as a detection-rate collapse.

Unlike the rest of `benchmarks/`, this one needs hardware — an Intel iGPU or
NPU and the `openvino` package — so it is not part of `run_all.py`. Without
them it reports what is missing and exits 0.

Run it::

    python benchmarks/bench_hand_backend.py
    python benchmarks/bench_hand_backend.py --tier lite --hands 2

Exit code 1 means a device disagreed with MediaPipe beyond tolerance.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visual_ai import accel, openvino_hands  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hand_motion.mp4"

#: MediaPipe's own two model tiers differ by ~7.8 px rmse on this clip at
#: 640x480, so anything under 20 px is the same trajectory with different
#: rounding; past that, the ROI geometry is wrong.
MAX_RMSE_PX = 20.0
#: A backend that finds the hand on far fewer frames is not tracking, however
#: fast it is.
MIN_DETECTION_RATIO = 0.8


def load(size=(640, 480)) -> list[np.ndarray]:
    if not FIXTURE.is_file():
        raise SystemExit(f"missing fixture: {FIXTURE}")
    capture = cv2.VideoCapture(str(FIXTURE))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def trajectory(tracker, frames: list[np.ndarray], warmup: int = 10):
    """Index-fingertip pixel position per frame (None where no hand), and times."""
    for frame in frames[:warmup]:
        tracker.process(frame)
    height, width = frames[0].shape[:2]
    tips: list[tuple[float, float] | None] = []
    times: list[float] = []
    for frame in frames:
        started = time.perf_counter()
        result = tracker.process(frame)
        times.append((time.perf_counter() - started) * 1e3)
        if result.multi_hand_landmarks:
            landmark = result.multi_hand_landmarks[0].landmark[8]
            tips.append((landmark.x * width, landmark.y * height))
        else:
            tips.append(None)
    return tips, times


def compare(reference, candidate) -> tuple[float, float, int]:
    """rmse, median error and sample count over frames where both found a hand."""
    both = [(a, b) for a, b in zip(reference, candidate) if a and b]
    if not both:
        return float("nan"), float("nan"), 0
    errors = [float(np.hypot(a[0] - b[0], a[1] - b[1])) for a, b in both]
    return (float(np.sqrt(np.mean(np.square(errors)))),
            float(np.median(errors)), len(both))


def report(label: str, times: list[float], detected: int, total: int, extra: str = "") -> None:
    print(f"  {label:24} mean={statistics.mean(times):6.2f}ms  "
          f"p95={sorted(times)[int(0.95 * len(times))]:6.2f}ms  "
          f"det={detected / total:5.1%}{extra}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--tier", choices=("full", "lite"), default="full",
                        help="model tier: full is model_complexity=1")
    parser.add_argument("--hands", type=int, default=1, help="max hands to track")
    args = parser.parse_args()

    devices = accel.available_devices()
    if not devices:
        print("no OpenVINO devices (is `openvino` installed?) — nothing to compare")
        return 0

    frames = load()
    print(f"clip {FIXTURE.name}: {len(frames)} frames at "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}, tier={args.tier}")
    print(f"devices: {', '.join(f'{d} ({accel.device_name(d)})' for d in devices)}\n")

    complexity = 1 if args.tier == "full" else 0
    try:
        from mediapipe.python.solutions import hands as mp_hands
    except Exception as exc:
        print(f"mediapipe is not importable, so there is nothing to compare against: {exc}")
        return 0

    reference_tracker = mp_hands.Hands(
        static_image_mode=False, max_num_hands=args.hands,
        model_complexity=complexity, min_detection_confidence=0.7,
        min_tracking_confidence=0.65)
    reference, mp_times = trajectory(reference_tracker, frames)
    reference_detected = sum(1 for tip in reference if tip)
    report("mediapipe (CPU)", mp_times, reference_detected, len(frames))

    failures: list[str] = []
    for device in devices:
        try:
            tracker = openvino_hands.OpenVINOHands(
                device=device, max_num_hands=args.hands,
                model_complexity=complexity, min_detection_confidence=0.7,
                min_tracking_confidence=0.65)
        except openvino_hands.OpenVINOHandsUnavailable as exc:
            print(f"  {('openvino ' + device):24} unavailable: {exc}")
            continue
        tips, times = trajectory(tracker, frames)
        detected = sum(1 for tip in tips if tip)
        rmse, median, samples = compare(reference, tips)
        report(f"openvino {device}", times, detected, len(frames),
               extra=f"  rmse={rmse:6.2f}px  median={median:5.2f}px  n={samples}")
        if detected < MIN_DETECTION_RATIO * reference_detected:
            failures.append(f"{device}: found a hand on {detected} frames against "
                            f"mediapipe's {reference_detected}")
        if not samples or rmse > MAX_RMSE_PX:
            failures.append(f"{device}: fingertip rmse {rmse:.2f}px exceeds {MAX_RMSE_PX}px")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nevery device tracks within tolerance of MediaPipe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
