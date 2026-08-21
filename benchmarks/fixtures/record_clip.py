"""One-off recorder for benchmarks/fixtures/hand_motion.mp4. Not part of the bench suite."""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "src"))
from visual_ai.capture import default_backend

OUT = "hand_motion.mp4"
DURATION_S = 12
WIDTH, HEIGHT, FPS = 1280, 720, 30


def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    # Platform backend: CAP_DSHOW is Windows-only, and this is the last
    # place in the repo that asked for it unconditionally.
    cap = cv2.VideoCapture(cam_index, default_backend())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {cam_index}")
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT, fourcc, FPS, (WIDTH, HEIGHT))

    print(f"Recording {DURATION_S}s from camera {cam_index} to {OUT} — "
          "move a hand around in frame now.")
    start = time.time()
    frames = 0
    while time.time() - start < DURATION_S:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frames += 1

    cap.release()
    writer.release()
    elapsed = time.time() - start
    print(f"Wrote {frames} frames in {elapsed:.1f}s ({frames/elapsed:.1f} fps) to {OUT}")


if __name__ == "__main__":
    main()
