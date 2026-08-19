"""Does the low-light boost actually recover hand tracking in the dark?

Synthetic frames cannot answer that -- MediaPipe either finds a real hand or
it does not -- so this captures your own camera and dims it in software:

    python benchmarks/low_light_bench.py               # capture and run
    python benchmarks/low_light_bench.py --save room.npz
    python benchmarks/low_light_bench.py --load room.npz   # repeatable

Hold one hand up, filling a reasonable part of the frame, while it captures.
The bright capture is the reference: its landmarks are treated as ground
truth, and each dimmed condition is scored on how often a hand is found at
all and how far the index fingertip lands from that reference.

Dimming multiplies the frame, which reproduces lost exposure but not the
sensor noise a real dark room adds -- so treat the numbers as an upper bound
on what the boost recovers, and the ordering between conditions as the point.

Exit code is 0 if the boost never made a condition worse.
"""

import os
import sys
import time
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from visual_ai.capture import default_backend                 # noqa: E402
from visual_ai.low_light import LowLightBoost, measure_luma   # noqa: E402

DIM_FACTORS = (1.0, 0.5, 0.35, 0.22, 0.14, 0.09)


def capture(n_frames, camera_index=0, width=800, height=600, countdown=4):
    cap = cv2.VideoCapture(camera_index, default_backend())
    if not cap.isOpened():
        print(f"camera {camera_index} would not open")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frames = []
    # Count down out loud. The capture lasts a couple of seconds, and a
    # reference pass with no hand in it invalidates every row below it --
    # which is worth more warning than a single printed line.
    print("hold ONE HAND up in view of the camera")
    for remaining in range(countdown, 0, -1):
        print(f"  capturing in {remaining}...")
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            cap.read()          # keeps auto-exposure settling meanwhile
    print(f"capturing {n_frames} frames -- hold still")
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.flip(cv2.resize(frame, (width, height)), 1))
        if len(frames) % 15 == 0:
            print(f"  {len(frames)}/{n_frames}")
    cap.release()
    return np.array(frames) if frames else None


def make_hands():
    import mediapipe as mp
    return mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def index_tip(result, w, h):
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark[8]
    return (lm.x * w, lm.y * h)


def run_condition(frames, factor, boost_on):
    """Detection rate, fingertip positions and cost for one dim factor."""
    h, w = frames.shape[1:3]
    hands = make_hands()
    boost = LowLightBoost() if boost_on else None
    tips, hits, luma_in, luma_out, cost_ms = [], 0, [], [], []

    for frame in frames:
        dim = frame if factor >= 1.0 else np.clip(
            frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        luma_in.append(measure_luma(dim))

        t0 = time.perf_counter()
        if boost is not None:
            dim = boost.apply(dim)
        cost_ms.append((time.perf_counter() - t0) * 1000.0)
        luma_out.append(measure_luma(dim))

        rgb = cv2.cvtColor(dim, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        tip = index_tip(hands.process(rgb), w, h)
        tips.append(tip)
        hits += tip is not None

    hands.close()
    return {
        "rate": hits / float(len(frames)),
        "tips": tips,
        "luma_in": float(np.mean(luma_in)),
        "luma_out": float(np.mean(luma_out)),
        "cost_ms": float(np.mean(cost_ms)),
        "gain": boost.gain if boost else 1.0,
    }


def tip_error(reference, tips):
    """Mean fingertip distance, over frames where both found a hand."""
    pairs = [(r, t) for r, t in zip(reference, tips) if r and t]
    if not pairs:
        return None
    return float(np.mean([np.hypot(r[0] - t[0], r[1] - t[1])
                          for r, t in pairs]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--save")
    ap.add_argument("--load")
    ap.add_argument("--countdown", type=int, default=4,
                    help="seconds of warning before capture starts")
    args = ap.parse_args(argv)

    if args.load:
        frames = np.load(args.load)["frames"]
        print(f"loaded {len(frames)} frames from {args.load}")
    else:
        frames = capture(args.frames, args.camera, countdown=args.countdown)
        if frames is None:
            return 2
        if args.save:
            np.savez_compressed(args.save, frames=frames)
            print(f"saved to {args.save}")

    print("\nreference pass (full brightness, no boost)")
    reference = run_condition(frames, 1.0, boost_on=False)
    print(f"  hand found in {reference['rate']*100:.0f}% of frames, "
          f"mean luma {reference['luma_in']:.0f}")
    if reference["rate"] < 0.5:
        print("\n  The bright reference barely found a hand, so nothing below\n"
              "  means anything. Re-run with your hand clearly in frame.")
        return 2

    print(f"\n{'dim':>5s} {'luma':>6s} | {'off: found':>11s} {'err px':>7s} | "
          f"{'on: found':>10s} {'err px':>7s} {'luma':>6s} {'gain':>5s} "
          f"{'cost ms':>8s}")
    regressed = False
    for factor in DIM_FACTORS:
        off = run_condition(frames, factor, boost_on=False)
        on = run_condition(frames, factor, boost_on=True)
        e_off = tip_error(reference["tips"], off["tips"])
        e_on = tip_error(reference["tips"], on["tips"])
        print(f"{factor:5.2f} {off['luma_in']:6.1f} | "
              f"{off['rate']*100:10.0f}% "
              f"{'  n/a' if e_off is None else format(e_off, '7.1f')} | "
              f"{on['rate']*100:9.0f}% "
              f"{'  n/a' if e_on is None else format(e_on, '7.1f')} "
              f"{on['luma_out']:6.1f} {on['gain']:5.2f} {on['cost_ms']:8.2f}")
        # Losing more than a couple of frames' worth of detections is a real
        # regression; a hair either way is MediaPipe's own run-to-run noise.
        if on["rate"] < off["rate"] - 0.05:
            regressed = True

    print("\nboost never lost detections" if not regressed
          else "\nWARNING: the boost lost detections in at least one condition")
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
