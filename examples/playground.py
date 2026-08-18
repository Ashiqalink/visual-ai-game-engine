"""
playground.py — live VisionPipeline payload inspector and recorder.

Everything the pipeline emits, on screen at once: hand signs, finger states,
pinch, velocities, depth, jitter stats, face box, both-hand slots. Use it to
answer "is the SDK seeing what I think it's seeing?" before blaming a game.

    python playground.py                       inspect live
    python playground.py --record out.jsonl    also log every payload
    python playground.py --max-hands 1

Keys
    K       toggle One-Euro landmark smoothing (X/Y)
    L       toggle ToF depth stabilizer (3 s hold-still calibration)
    X       cancel an in-progress calibration
    R       start / stop recording payloads to a JSONL file
    SPACE   pretty-print the current payload to the console
    Q/ESC   quit

Recordings strip the camera frame (and depth grid, if enabled) and add a
`_t` wall-clock timestamp per line, so a session replays as data, not video.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

# Runs standalone too: fall back to ../src when the launcher hasn't set PYTHONPATH.
try:
    import visual_ai  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from visual_ai import VisionPipeline

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 360

WHITE = (235, 235, 235)
GREY = (150, 150, 150)
GREEN = (80, 220, 120)
RED = (80, 80, 235)
CYAN = (220, 200, 60)
ORANGE = (60, 160, 245)


def sanitize(payload: dict) -> dict:
    """Payload minus the unserializable bulk, ready for json.dumps."""
    out = {}
    for key, value in payload.items():
        if key in ("frame", "depth_grid"):
            continue
        if isinstance(value, np.ndarray):
            continue
        if key == "hands":
            value = [sanitize(h) for h in value]
        elif key in ("hand_left", "hand_right"):
            value = sanitize(value) if value is not None else None
        elif isinstance(value, (np.floating, np.integer)):
            value = value.item()
        elif isinstance(value, tuple):
            value = [v.item() if isinstance(v, (np.floating, np.integer)) else v
                     for v in value]
        out[key] = value
    return out


class Recorder:
    def __init__(self, path: Path | None):
        self.path = path
        self._fh = None
        self.lines = 0

    @property
    def active(self) -> bool:
        return self._fh is not None

    def toggle(self) -> None:
        if self.active:
            self.stop()
            return
        path = self.path or Path(f"payload_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        self.path = path
        self._fh = path.open("w", encoding="utf-8")
        self.lines = 0
        print(f"[playground] recording -> {path.resolve()}")

    def write(self, payload: dict) -> None:
        if not self.active:
            return
        row = sanitize(payload)
        row["_t"] = time.time()
        self._fh.write(json.dumps(row) + "\n")
        self.lines += 1

    def stop(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
            print(f"[playground] wrote {self.lines} payloads to {self.path.resolve()}")


def draw_text(img, text, pos, color=WHITE, scale=0.45, thick=1):
    cv2.putText(img, text, pos, FONT, scale, color, thick, cv2.LINE_AA)


def draw_bar(img, x, y, w, fraction, color, label=""):
    fraction = max(0.0, min(1.0, fraction))
    cv2.rectangle(img, (x, y), (x + w, y + 10), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * fraction), y + 10), color, -1)
    if label:
        draw_text(img, label, (x + w + 8, y + 9), GREY, 0.4)


def draw_panel(panel, payload, fps, recorder):
    """The left column: every interesting payload key as text, dots and bars."""
    y = 24
    def line(label, value, color=WHITE):
        nonlocal y
        draw_text(panel, label, (12, y), GREY, 0.42)
        draw_text(panel, str(value), (150, y), color, 0.42)
        y += 20

    draw_text(panel, "PAYLOAD PLAYGROUND", (12, y), CYAN, 0.55, 2)
    y += 26

    line("fps", f"{fps:.1f}")
    line("hand_visible", payload["hand_visible"],
         GREEN if payload["hand_visible"] else GREY)
    line("hand_count", payload.get("hand_count", 0))
    line("handedness", f'{payload["handedness"]} '
                       f'({payload["handedness_score"]:.2f})')
    line("hand_sign", payload["hand_sign"],
         GREEN if payload["hand_sign"] != "unknown" else GREY)

    # Finger dots: thumb..pinky
    draw_text(panel, "fingers", (12, y), GREY, 0.42)
    for i, extended in enumerate(payload["fingers_extended"]):
        cv2.circle(panel, (156 + i * 22, y - 4), 7,
                   GREEN if extended else (70, 70, 70), -1)
    y += 22

    draw_text(panel, "grip_openness", (12, y), GREY, 0.42)
    draw_bar(panel, 150, y - 9, 120, payload["grip_openness"],
             ORANGE, f'{payload["grip_openness"]:.2f}')
    y += 22

    line("is_pinching", payload["is_pinching"],
         GREEN if payload["is_pinching"] else GREY)
    line("3_finger_pinch", payload["is_3_finger_pinching"],
         GREEN if payload["is_3_finger_pinching"] else GREY)
    line("index_isolated", payload["is_index_isolated"],
         GREEN if payload["is_index_isolated"] else GREY)
    line("click_fired", payload["click_just_fired"],
         RED if payload["click_just_fired"] else GREY)
    y += 6

    line("index_pos", tuple(int(v) for v in payload["index_pos"]))
    line("index_speed", f'{payload["index_speed"]:.0f} px/s')
    line("index_accel", f'{payload["index_accel"]:.0f} px/s2')
    line("z_delta", f'{payload["z_delta"]:+.4f}')
    y += 6

    line("smoothing", "ON" if payload["smoothing_enabled"] else "OFF (raw)",
         GREEN if payload["smoothing_enabled"] else ORANGE)
    jitter = payload["jitter"]
    line("jitter raw/sm", f'{jitter["raw_jitter_std"]:.2f} / '
                          f'{jitter["smoothed_jitter_std"]:.2f} px')
    line("reduction", f'{jitter["jitter_reduction_pct"]:.0f} %')
    y += 6

    line("depth_source", payload["depth_source"])
    line("tof_active", payload["tof_active"],
         GREEN if payload["tof_active"] else GREY)
    line("tof_z", f'{payload["tof_z_m"]:.3f} m  (raw {payload["tof_z_raw"]:.3f})')
    line("stabilizer", f'{payload["stabilizer_state"]} '
                       f'{payload["stabilizer_progress"] * 100:.0f}%')
    y += 6

    line("face_visible", payload["face_visible"],
         GREEN if payload["face_visible"] else GREY)
    line("face_count", payload.get("face_count", 0))
    y += 10

    if recorder.active:
        draw_text(panel, f"REC {recorder.lines} payloads", (12, y), RED, 0.5, 2)
    y += 24
    draw_text(panel, "K smooth  L stabilize  X cancel  R record  SPACE dump  Q quit",
              (12, panel.shape[0] - 12), GREY, 0.38)


def draw_overlay(frame, payload):
    """Markers on the camera frame for every position the payload carries."""
    for hand in payload.get("hands", ()) or (payload,):
        if not hand.get("hand_visible"):
            continue
        ix, iy = (int(v) for v in hand["index_pos"])
        tx_, ty_ = (int(v) for v in hand["thumb_pos"])
        mx, my = (int(v) for v in hand["middle_pos"])
        cv2.circle(frame, (ix, iy), 10, GREEN, 2)
        cv2.circle(frame, (tx_, ty_), 7, CYAN, 2)
        cv2.circle(frame, (mx, my), 7, ORANGE, 2)
        if hand["is_pinching"]:
            px, py = (int(v) for v in hand["pinch_pos"])
            cv2.circle(frame, (px, py), 14, RED, -1)
        # velocity vector off the fingertip
        vx, vy = hand["index_velocity"]
        cv2.arrowedLine(frame, (ix, iy),
                        (int(ix + vx * 0.15), int(iy + vy * 0.15)), GREEN, 2)
        label = hand.get("handedness", "?")
        draw_text(frame, f'{label} {hand["hand_sign"]}', (ix + 14, iy - 14), GREEN, 0.5, 2)

    if payload["face_visible"]:
        fx, fy, fw, fh = (int(v) for v in payload["face_box"])
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), CYAN, 2)

    tx_, ty_ = int(payload["target_x"]), int(payload["target_y"])
    cv2.drawMarker(frame, (tx_, ty_), WHITE, cv2.MARKER_CROSS, 16, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--record", type=Path, metavar="FILE",
                        help="record payloads to this JSONL file from the start")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    ai_queue: queue.Queue = queue.Queue(maxsize=1)
    pipeline = VisionPipeline(result_queue=ai_queue, width=args.width,
                              height=args.height, max_hands=args.max_hands)
    pipeline.start()

    recorder = Recorder(args.record)
    if args.record:
        recorder.toggle()

    payload = None
    tof_stab_on = False
    fps, frames, fps_timer = 0.0, 0, time.time()

    try:
        while True:
            # Drain to the freshest payload (the queue contract).
            try:
                while True:
                    payload = ai_queue.get_nowait()
            except queue.Empty:
                pass
            if payload is None:
                time.sleep(0.01)
                continue

            recorder.write(payload)

            frames += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps, frames, fps_timer = frames / (now - fps_timer), 0, now

            frame = payload["frame"]
            if frame is None:
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            canvas = np.zeros((frame.shape[0], frame.shape[1] + PANEL_W, 3),
                              dtype=np.uint8)
            canvas[:, PANEL_W:] = frame
            draw_overlay(canvas[:, PANEL_W:], payload)
            draw_panel(canvas[:, :PANEL_W], payload, fps, recorder)

            cv2.imshow("visual_ai payload playground", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("k"):
                state = pipeline.toggle_smoothing()
                print(f"[playground] smoothing {'ON' if state else 'OFF'}")
            elif key == ord("l"):
                tof_stab_on = not tof_stab_on
                if tof_stab_on:
                    pipeline.begin_stabilization(3.0)
                    print("[playground] ToF stabilizer: calibrating — hold still")
                else:
                    pipeline.disable_stabilization()
                    print("[playground] ToF stabilizer: OFF")
            elif key == ord("x"):
                pipeline.cancel_stabilization()
                tof_stab_on = False
            elif key == ord("r"):
                recorder.toggle()
            elif key == ord(" "):
                print(json.dumps(sanitize(payload), indent=2))
    finally:
        recorder.stop()
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
