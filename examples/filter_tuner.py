"""
filter_tuner.py — tune the One-Euro landmark filter against your own hand.

Draws the raw MediaPipe track (grey) and the smoothed track (green) of your
pinch point side by side while you retune `min_cutoff` and `beta` live, with
the pipeline's own jitter statistics on screen. The benchmarks score filters
against synthetic streams; this shows what a tuning feels like on a real hand,
and prints the numbers to back it up.

Two axes, two failure modes:
    min_cutoff too high  -> jitter at rest      (lower it)
    beta too low         -> lag on fast swipes  (raise it)

Keys
    1 / 2   min_cutoff  down / up   (x1.25 steps)
    3 / 4   beta        down / up   (x1.5 steps)
    K       toggle smoothing entirely (compare against raw)
    R       reset the trails
    D       dump current tuning + jitter stats to the console
    Q/ESC   quit

Start from the defaults, wave fast until the green trail stops cutting
corners (raise beta), then hold still until the green dot freezes (lower
min_cutoff). `play conductor --beta ...` accepts what you land on.
"""

from __future__ import annotations

import argparse
import collections
import queue
import sys
import time
from pathlib import Path

try:
    import visual_ai  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from visual_ai import VisionPipeline

FONT = cv2.FONT_HERSHEY_SIMPLEX
TRAIL = 120                      # points kept per trail (~4 s at 30 fps)

RAW_COLOR = (150, 150, 150)
SMOOTH_COLOR = (80, 220, 120)
LAG_COLOR = (60, 160, 245)


def draw_trail(frame, points, color, thickness):
    pts = list(points)
    for a, b in zip(pts, pts[1:]):
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 color, thickness, cv2.LINE_AA)


def text(frame, s, pos, color=(235, 235, 235), scale=0.5, thick=1):
    cv2.putText(frame, s, pos, FONT, scale, color, thick, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min-cutoff", type=float, default=None,
                        help="starting min_cutoff in Hz (default: pipeline's)")
    parser.add_argument("--beta", type=float, default=None,
                        help="starting beta (default: pipeline's)")
    args = parser.parse_args()

    ai_queue: queue.Queue = queue.Queue(maxsize=1)
    pipeline = VisionPipeline(result_queue=ai_queue, width=args.width,
                              height=args.height, max_hands=1)
    if args.min_cutoff is not None or args.beta is not None:
        pipeline.set_filter_tuning(min_cutoff=args.min_cutoff, beta=args.beta)
    pipeline.start()

    raw_trail = collections.deque(maxlen=TRAIL)
    smooth_trail = collections.deque(maxlen=TRAIL)
    lag_px = 0.0                 # distance raw->smoothed, EMA'd for readability
    payload = None

    try:
        while True:
            try:
                while True:
                    payload = ai_queue.get_nowait()
            except queue.Empty:
                pass
            if payload is None:
                time.sleep(0.01)
                continue

            frame = payload["frame"]
            if frame is None:
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            frame = frame.copy()

            if payload["hand_visible"]:
                raw = payload["pinch_pos_raw"]
                smooth = payload["pinch_pos"]
                raw_trail.append(raw)
                smooth_trail.append(smooth)
                d = ((raw[0] - smooth[0]) ** 2 + (raw[1] - smooth[1]) ** 2) ** 0.5
                lag_px = lag_px * 0.9 + d * 0.1

            draw_trail(frame, raw_trail, RAW_COLOR, 1)
            draw_trail(frame, smooth_trail, SMOOTH_COLOR, 2)
            if payload["hand_visible"]:
                rx, ry = (int(v) for v in payload["pinch_pos_raw"])
                sx, sy = (int(v) for v in payload["pinch_pos"])
                cv2.circle(frame, (rx, ry), 5, RAW_COLOR, -1)
                cv2.circle(frame, (sx, sy), 7, SMOOTH_COLOR, -1)
                cv2.line(frame, (rx, ry), (sx, sy), LAG_COLOR, 1, cv2.LINE_AA)

            # ── HUD ───────────────────────────────────────────────────────────
            jitter = payload["jitter"]
            smoothing = payload["smoothing_enabled"]
            text(frame, "ONE-EURO FILTER TUNER", (15, 28), (220, 200, 60), 0.6, 2)
            text(frame, f"min_cutoff {pipeline.filter_min_cutoff:.3f} Hz   "
                        f"beta {pipeline.filter_beta:.4f}", (15, 54))
            text(frame, f"smoothing {'ON' if smoothing else 'OFF (raw passthrough)'}",
                 (15, 76), SMOOTH_COLOR if smoothing else LAG_COLOR)
            text(frame, f"jitter raw {jitter['raw_jitter_std']:.2f} px   "
                        f"smoothed {jitter['smoothed_jitter_std']:.2f} px   "
                        f"reduction {jitter['jitter_reduction_pct']:.0f}%", (15, 98))
            text(frame, f"lag (raw->smoothed dist) {lag_px:.1f} px", (15, 120),
                 LAG_COLOR)
            text(frame, "1/2 min_cutoff  3/4 beta  K raw  R reset  D dump  Q quit",
                 (15, frame.shape[0] - 15), (150, 150, 150), 0.42)

            cv2.imshow("visual_ai filter tuner", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("1"):
                pipeline.set_filter_tuning(min_cutoff=pipeline.filter_min_cutoff / 1.25)
            elif key == ord("2"):
                pipeline.set_filter_tuning(min_cutoff=pipeline.filter_min_cutoff * 1.25)
            elif key == ord("3"):
                pipeline.set_filter_tuning(beta=pipeline.filter_beta / 1.5)
            elif key == ord("4"):
                pipeline.set_filter_tuning(beta=max(pipeline.filter_beta, 1e-4) * 1.5)
            elif key == ord("k"):
                pipeline.toggle_smoothing()
            elif key == ord("r"):
                raw_trail.clear()
                smooth_trail.clear()
                lag_px = 0.0
            elif key == ord("d"):
                print(f"min_cutoff={pipeline.filter_min_cutoff:.4f} "
                      f"beta={pipeline.filter_beta:.5f} "
                      f"raw_std={jitter['raw_jitter_std']:.2f}px "
                      f"smoothed_std={jitter['smoothed_jitter_std']:.2f}px "
                      f"reduction={jitter['jitter_reduction_pct']:.0f}% "
                      f"lag={lag_px:.1f}px")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
