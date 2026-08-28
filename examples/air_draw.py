"""
air_draw.py — draw in the air with a pinch.

A drawing canvas layered over the camera. Pinch (thumb + index) to put ink at
the pinch point, open your palm to erase around your hand, point to move
without drawing. Doubles as a pinch-reliability test: broken strokes mean the
pinch threshold is flickering, and wobbly lines show the current smoothing
tuning directly (tune it in filter_tuner.py, keyed off the same pinch point).

Keys
    1-6     ink colour
    + / -   brush size
    E       toggle eraser lock (erase with the fingertip, no palm needed)
    U       undo the last stroke
    C       clear the canvas
    S       save the drawing as PNG (canvas only, no camera frame)
    Q/ESC   quit
"""

from __future__ import annotations

import argparse
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

from visual_ai import VisionPipeline, display

FONT = cv2.FONT_HERSHEY_SIMPLEX

PALETTE = [                       # BGR
    (80, 220, 120),   # green
    (60, 160, 245),   # orange
    (235, 90, 80),    # blue
    (90, 90, 235),    # red
    (220, 200, 60),   # cyan
    (235, 235, 235),  # white
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    ai_queue: queue.Queue = queue.Queue(maxsize=1)
    pipeline = VisionPipeline(result_queue=ai_queue, width=args.width,
                              height=args.height, max_hands=1)
    pipeline.start()

    canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    strokes: list[np.ndarray] = []       # canvas snapshots for undo
    color_index = 0
    brush = 6
    eraser_lock = False
    last_point: tuple[int, int] | None = None
    drawing = False                      # were we inking last frame?
    payload = None
    flash = ""                           # transient status line
    flash_until = 0.0

    def note(msg: str) -> None:
        nonlocal flash, flash_until
        flash, flash_until = msg, time.time() + 1.5

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

            if payload["hand_visible"]:
                px, py = (int(v) for v in payload["pinch_pos"])
                ink = payload["is_pinching"] and not eraser_lock
                erase = payload["is_open_palm"] or (eraser_lock and payload["is_pinching"])

                if ink:
                    if not drawing:
                        strokes.append(canvas.copy())   # undo point at stroke start
                        if len(strokes) > 20:
                            strokes.pop(0)
                        last_point = (px, py)
                    cv2.line(canvas, last_point, (px, py),
                             PALETTE[color_index], brush, cv2.LINE_AA)
                    last_point = (px, py)
                elif erase:
                    cv2.circle(canvas, (px, py), brush * 5, (0, 0, 0), -1)
                    last_point = None
                else:
                    last_point = None
                drawing = ink
            else:
                drawing, last_point = False, None

            # Composite: camera dimmed under the ink so strokes stay readable.
            view = cv2.addWeighted(frame, 0.55, canvas, 1.0, 0)

            # Cursor
            if payload["hand_visible"]:
                px, py = (int(v) for v in payload["pinch_pos"])
                if payload["is_open_palm"] or (eraser_lock and payload["is_pinching"]):
                    cv2.circle(view, (px, py), brush * 5, (120, 120, 120), 2)
                else:
                    cv2.circle(view, (px, py), max(brush, 4),
                               PALETTE[color_index],
                               -1 if payload["is_pinching"] else 2)

            # HUD: palette swatches + status
            for i, col in enumerate(PALETTE):
                x = 15 + i * 34
                cv2.rectangle(view, (x, 12), (x + 26, 38), col, -1)
                if i == color_index:
                    cv2.rectangle(view, (x - 2, 10), (x + 28, 40), (255, 255, 255), 2)
            cv2.putText(view, f"brush {brush}px" + ("  ERASER" if eraser_lock else ""),
                        (15 + len(PALETTE) * 34 + 10, 32), FONT, 0.5,
                        (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(view, "pinch draw   open palm erase   1-6 colour  +/- size  "
                              "E eraser  U undo  C clear  S save  Q quit",
                        (15, view.shape[0] - 15), FONT, 0.4,
                        (150, 150, 150), 1, cv2.LINE_AA)
            if time.time() < flash_until:
                cv2.putText(view, flash, (15, 64), FONT, 0.55,
                            (220, 200, 60), 2, cv2.LINE_AA)

            # `27` is also what closing the window reports, so the X button now
            # leaves through the same path - and unsaved ink is still lost, as
            # it was before.
            key = display.show("visual_ai air draw", view)
            if key in (27, ord("q")):
                break
            elif ord("1") <= key <= ord("6"):
                color_index = key - ord("1")
            elif key in (ord("+"), ord("=")):
                brush = min(40, brush + 2)
            elif key == ord("-"):
                brush = max(2, brush - 2)
            elif key == ord("e"):
                eraser_lock = not eraser_lock
                note("eraser " + ("locked" if eraser_lock else "off"))
            elif key == ord("u"):
                if strokes:
                    canvas = strokes.pop()
                    note("undo")
            elif key == ord("c"):
                strokes.append(canvas.copy())
                canvas = np.zeros_like(canvas)
                note("cleared")
            elif key == ord("s"):
                path = Path(f"air_draw_{time.strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(str(path), canvas)
                note(f"saved {path.name}")
                print(f"[air_draw] saved {path.resolve()}")
    finally:
        pipeline.stop()
        display.close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
