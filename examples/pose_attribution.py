"""
pose_attribution.py — "whose hand is this?", answered by the arm, not by proximity.

A test rig for multiplayer hand ownership. `VisionPipeline` tells you *what* each
hand is doing but not *who* it belongs to: with two people in frame, a hand that
wanders into a free slot drives the game exactly like the player's own. Guessing
by nearest-face-in-x breaks the moment two players reach across each other.

This playground attributes hands anatomically instead. MediaPipe's Tasks
`PoseLandmarker` returns one skeleton *per person*, already grouped — so the
wrist -> elbow -> shoulder -> head chain you would otherwise have to trace is
simply the structure of the result. Each `Hands` detection is matched to the pose
whose own hand landmarks sit closest to it, which is a hand-to-hand match of a
few tens of pixels rather than a torso-scale guess.

Nothing here touches the SDK. Pose runs on `payload["frame"]` (already mirrored
by the pipeline, so the coordinates share display space with `payload["hands"]`)
and the attribution is layered on top. Read it as a proposal you can watch fail.

    play pose                             (once registered in play.py)
    python pose_attribution.py            live, two players
    python pose_attribution.py --download first run: fetch the ~5.5 MB model
    python pose_attribution.py --pose-every 5 --max-hands 4

What to watch
    * The chain readout per hand — W-E-S-H means wrist, elbow, shoulder and head
      were all confidently visible, so the attribution rests on the full arm.
      Dashes mark occluded links; the deepest visible one still decides.
    * `pose ms` against `loop fps`. Pose is the expensive half by a mile:
      measured here at 720p it costs ~31 ms for one player and ~65 ms for two,
      and downscaling the input does nothing because MediaPipe resizes to the
      model's own input size anyway. It therefore runs on its own thread; use
      --sync to watch the render loop collapse to ~15 fps without it.
    * Raise --pose-every and confirm attribution survives on sticky slots
      between refreshes, because who-owns-what only changes when a hand enters
      or leaves frame, not while it moves.
    * Cross your arms with a second player. Face-proximity would swap you here.

Keys
    P       pose on / off (off = every hand falls back to sticky, then unowned)
    [ / ]   pose decimation down / up (run pose every N frames)
    S       cycle strictness of the match radius
    D       dump the current attribution table to the console
    Q/ESC   quit
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Runs standalone too: fall back to ../src when the launcher hasn't set PYTHONPATH.
try:
    import visual_ai  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from visual_ai import VisionPipeline

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent / "assets" / "pose_landmarker_lite.task"
)

# BlazePose 33-landmark indices. Only the upper body matters here.
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22

# Per side: the four hand-ish points, and the chain from hand back to head.
SIDES = {
    "L": {
        "hand": (L_WRIST, L_PINKY, L_INDEX, L_THUMB),
        "chain": (("W", L_WRIST), ("E", L_ELBOW), ("S", L_SHOULDER), ("H", NOSE)),
    },
    "R": {
        "hand": (R_WRIST, R_PINKY, R_INDEX, R_THUMB),
        "chain": (("W", R_WRIST), ("E", R_ELBOW), ("S", R_SHOULDER), ("H", NOSE)),
    },
}

VIS_MIN = 0.5          # landmark visibility below this counts as occluded
STRICTNESS = (0.9, 0.6, 0.4)   # match radius as a multiple of shoulder width

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (235, 235, 235)
GREY = (150, 150, 150)
DIM = (90, 90, 90)
RED = (80, 80, 235)
# One colour per player slot; anything unowned is drawn grey.
PLAYER_COLOURS = ((220, 200, 60), (60, 160, 245), (120, 220, 120), (200, 120, 220))


# ── Model asset ───────────────────────────────────────────────────────────────

def resolve_model(path: Path, allow_download: bool) -> Path:
    """
    Locate the pose_landmarker .task bundle, fetching it only when asked.

    MediaPipe ships the legacy single-person pose .tflite files inside the wheel
    but no Tasks bundle, and the legacy `solutions.pose` API cannot do more than
    one person — so multiplayer needs this ~5.5 MB download. It is deliberately
    opt-in rather than automatic: it is a network fetch into the engine repo.
    """
    if path.exists():
        return path
    if not allow_download:
        raise SystemExit(
            f"Pose model not found at {path}\n"
            f"  Fetch it with:  python {Path(__file__).name} --download\n"
            f"  Or point elsewhere with --model <path>\n"
            f"  Source: {MODEL_URL}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pose] downloading {MODEL_URL}\n[pose]   -> {path}")
    tmp = path.with_suffix(".partial")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(path)
    print(f"[pose] done ({path.stat().st_size / 1e6:.1f} MB)")
    return path


# ── Attribution ───────────────────────────────────────────────────────────────

class PoseAttributor:
    """
    Maps each tracked hand onto a player, using pose skeletons as the evidence.

    Player slots are held stable the same way `VisionPipeline._assign_slots`
    holds hand slots — nearest-neighbour against the previous frame's anchor —
    because `PoseLandmarker` no more promises a stable result order than `Hands`
    does. Anchoring on the shoulder midpoint rather than a wrist is what makes
    this cheap: torsos barely move between frames, so the match is unambiguous.
    """

    def __init__(self, model_path: Path, max_players: int = 2, strictness: int = 1):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp_vision = vision
        # Handed over as a buffer, not a path: MediaPipe treats a Windows
        # "D:\..." string as relative and joins it onto its own package
        # directory, which fails with a baffling errno=22 on a file that
        # plainly exists. Bytes have no such ambiguity.
        self._landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_buffer=Path(model_path).read_bytes()
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=max_players,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.max_players = max_players
        self.strictness = strictness

        self.arms: list[dict] = []          # one per (player, side) with a visible hand
        self.players: list[dict] = []       # slot-indexed pose summaries
        # `arms` and `players` are rebuilt together and read together, and with
        # a PoseWorker the writer is a different thread from the reader.
        self._lock = threading.RLock()
        self._anchor: list[tuple[float, float] | None] = [None] * max_players
        self._sticky: dict[int, int] = {}   # hand slot -> player slot, across pose gaps
        self.last_ms = 0.0
        # VIDEO mode rejects a timestamp that does not strictly increase, and it
        # raises rather than skipping the frame. Owning the counter here means a
        # decimated or paused caller can never stall the landmarker.
        self._ts_ms = 0

    def close(self) -> None:
        self._landmarker.close()

    # -- pose refresh -------------------------------------------------------

    def update(self, frame_bgr: np.ndarray) -> None:
        """Run pose on one frame and rebuild the player/arm tables."""
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._ts_ms += 33
        t0 = time.perf_counter()
        result = self._landmarker.detect_for_video(image, self._ts_ms)
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        raw = list(result.pose_landmarks or [])
        assignment = self._assign_player_slots(raw, w, h)

        players: list[dict | None] = [None] * self.max_players
        arms: list[dict] = []

        for pose_index, slot in assignment.items():
            lms = raw[pose_index]
            px = [(lm.x * w, lm.y * h) for lm in lms]
            vis = [self._visibility(lm) for lm in lms]

            shoulder_w = _dist(px[L_SHOULDER], px[R_SHOULDER])
            # A person turned side-on collapses the shoulder line, which would
            # shrink the match radius to nothing. Ear span is the fallback
            # scale, and a floor keeps the radius usable when both are lost.
            if shoulder_w < 1e-3 or min(vis[L_SHOULDER], vis[R_SHOULDER]) < VIS_MIN:
                shoulder_w = max(_dist(px[L_EAR], px[R_EAR]) * 2.0, w * 0.08)

            players[slot] = {
                "slot": slot,
                "px": px,
                "vis": vis,
                "scale": shoulder_w,
                "head": px[NOSE],
                "head_visible": vis[NOSE] >= VIS_MIN,
            }

            for side, spec in SIDES.items():
                pts = [px[i] for i in spec["hand"] if vis[i] >= VIS_MIN]
                if not pts:
                    continue
                chain = [(tag, vis[i] >= VIS_MIN) for tag, i in spec["chain"]]
                # "As far as we can trace it": the deepest confidently visible
                # link. A shoulder alone already identifies the person; an
                # elbow alone still narrows it to one torso.
                depth = 0
                for _, ok in chain:
                    if not ok:
                        break
                    depth += 1
                arms.append({
                    "player": slot,
                    "side": side,
                    "centroid": (
                        sum(p[0] for p in pts) / len(pts),
                        sum(p[1] for p in pts) / len(pts),
                    ),
                    "chain": chain,
                    "depth": depth,
                    "scale": shoulder_w,
                })

        with self._lock:
            self.players = [p for p in players if p is not None]
            self.arms = arms

    def _visibility(self, lm) -> float:
        v = getattr(lm, "visibility", None)
        return 1.0 if v is None else float(v)

    def _assign_player_slots(self, raw, w: int, h: int) -> dict[int, int]:
        """Nearest-neighbour pose -> player slot on the shoulder midpoint."""
        if not raw:
            return {}

        centres = []
        for lms in raw:
            ls, rs = lms[L_SHOULDER], lms[R_SHOULDER]
            centres.append((((ls.x + rs.x) / 2.0) * w, ((ls.y + rs.y) / 2.0) * h))

        # Torsos are slow; a quarter-frame radius is generous and still refuses
        # to hand a slot to somebody who just walked in on the far side.
        max_match = 0.25 * max(w, h)
        pairs = sorted(
            (_dist(centres[i], self._anchor[s]), i, s)
            for i in range(len(centres))
            for s in range(self.max_players)
            if self._anchor[s] is not None
        )

        assignment: dict[int, int] = {}
        used: set[int] = set()
        for distance, i, s in pairs:
            if i in assignment or s in used or distance > max_match:
                continue
            assignment[i] = s
            used.add(s)

        for i in range(len(centres)):
            if i in assignment:
                continue
            free = next((s for s in range(self.max_players) if s not in used), None)
            if free is None:
                break
            assignment[i] = free
            used.add(free)

        for i, s in assignment.items():
            self._anchor[s] = centres[i]
        return assignment

    # -- hand -> player -----------------------------------------------------

    def attribute(self, hands) -> list[dict]:
        """
        Match every tracked hand to a player. One row per hand, in slot order.

        Falls back to the previous verdict when no arm matches — a hand behind
        the player's own torso loses its chain for a few frames, and "keep what
        it was" degrades far better than re-guessing from scratch.
        """
        radius_k = STRICTNESS[self.strictness]
        taken: set[tuple[int, str]] = set()
        rows: list[dict] = []
        with self._lock:
            arms = list(self.arms)

        for hand in hands:
            slot = hand.get("slot", 0)
            point = hand.get("pinch_pos_raw") or hand.get("index_pos")
            best, best_d = None, float("inf")

            for arm in arms:
                key = (arm["player"], arm["side"])
                if key in taken:
                    continue          # one pose arm owns at most one hand
                d = _dist(point, arm["centroid"])
                if d < best_d:
                    best, best_d = arm, d

            if best is not None and best_d <= radius_k * best["scale"]:
                taken.add((best["player"], best["side"]))
                self._sticky[slot] = best["player"]
                rows.append({
                    "slot": slot, "hand": hand, "point": point,
                    "player": best["player"], "arm": best,
                    "dist": best_d, "source": "arm",
                })
            elif slot in self._sticky:
                rows.append({
                    "slot": slot, "hand": hand, "point": point,
                    "player": self._sticky[slot], "arm": None,
                    "dist": best_d, "source": "sticky",
                })
            else:
                rows.append({
                    "slot": slot, "hand": hand, "point": point,
                    "player": None, "arm": None,
                    "dist": best_d, "source": "none",
                })

        live = {h.get("slot", 0) for h in hands}
        for slot in [s for s in self._sticky if s not in live]:
            del self._sticky[slot]
        return rows


def _dist(a, b) -> float:
    if a is None or b is None:
        return float("inf")
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class PoseWorker(threading.Thread):
    """
    Runs pose off the render loop, on whatever the newest frame happens to be.

    Measured on this machine at 720p: pose lite costs ~31 ms at ``num_poses=1``
    and ~65 ms at 2 — the second person is a whole extra inference pass, and
    downscaling the input does not help because MediaPipe resizes to the model's
    own fixed size internally. Run synchronously that caps the game at ~15 fps.

    Off-thread it costs the render loop nothing, and attribution simply runs
    against the most recent verdict. That is sound because ownership is nearly
    static: it changes when a hand enters or leaves frame, not while it moves.
    The price is latency on a *new* hand — it stays unowned for up to one pose
    interval — which is a few hundred ms, once, per hand.
    """

    def __init__(self, attributor: PoseAttributor):
        super().__init__(daemon=True)
        self.att = attributor
        self.running = True
        self._pending: np.ndarray | None = None
        self._wake = threading.Event()
        self._lock = threading.Lock()

    def submit(self, frame: np.ndarray) -> None:
        """Offer a frame. Never blocks; a newer frame simply replaces an older."""
        with self._lock:
            self._pending = frame
        self._wake.set()

    def run(self) -> None:
        while self.running:
            self._wake.wait(0.05)
            self._wake.clear()
            with self._lock:
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            try:
                self.att.update(frame)
            except Exception as exc:            # keep the demo alive to show it
                print(f"[pose] worker error: {exc}")

    def stop(self) -> None:
        self.running = False
        self._wake.set()
        self.join(timeout=2.0)


# ── Drawing ───────────────────────────────────────────────────────────────────

def colour_for(player) -> tuple[int, int, int]:
    return GREY if player is None else PLAYER_COLOURS[player % len(PLAYER_COLOURS)]


def draw_pose(frame, player: dict) -> None:
    """Arm chains and head only — the evidence the attribution actually uses."""
    col = colour_for(player["slot"])
    px, vis = player["px"], player["vis"]

    for spec in SIDES.values():
        chain = [i for _, i in spec["chain"]]
        for a, b in zip(chain, chain[1:]):
            ok = vis[a] >= VIS_MIN and vis[b] >= VIS_MIN
            cv2.line(frame, _i(px[a]), _i(px[b]), col if ok else DIM, 3 if ok else 1)
        for i in chain:
            if vis[i] >= VIS_MIN:
                cv2.circle(frame, _i(px[i]), 5, col, -1)

    if vis[L_SHOULDER] >= VIS_MIN and vis[R_SHOULDER] >= VIS_MIN:
        cv2.line(frame, _i(px[L_SHOULDER]), _i(px[R_SHOULDER]), col, 2)
    if player["head_visible"]:
        cv2.circle(frame, _i(player["head"]), 14, col, 2)
        cv2.putText(frame, f"P{player['slot']}",
                    (int(player["head"][0]) - 14, int(player["head"][1]) - 20),
                    FONT, 0.6, col, 2)


def draw_attribution(frame, rows: list[dict]) -> None:
    for row in rows:
        col = colour_for(row["player"])
        point = _i(row["point"])
        cv2.circle(frame, point, 11, col, 2)

        if row["arm"] is not None:
            # The link that carries the claim: hand -> the pose arm it matched.
            cv2.line(frame, point, _i(row["arm"]["centroid"]), col, 1)

        owner = "?" if row["player"] is None else f"P{row['player']}"
        chain = "".join(t if ok else "-" for t, ok in row["arm"]["chain"]) \
            if row["arm"] else "...."
        tag = f"{owner} h{row['slot']} {chain}"
        if row["source"] == "sticky":
            tag += " sticky"
        cv2.putText(frame, tag, (point[0] + 14, point[1] - 10), FONT, 0.5, col, 1)


def draw_hud(frame, att: PoseAttributor, rows, pose_on, pose_every, fps, worker) -> None:
    h = frame.shape[0]
    mode = "sync" if worker is None else "thread"
    lines = [
        (f"pose {'ON' if pose_on else 'OFF'} [{mode}]  every {pose_every}f"
         f"  {att.last_ms:5.1f} ms   loop {fps:4.1f} fps", WHITE),
        (f"players {len(att.players)}   arms {len(att.arms)}   hands {len(rows)}"
         f"   radius x{STRICTNESS[att.strictness]}", GREY),
    ]
    owned = sum(1 for r in rows if r["source"] == "arm")
    sticky = sum(1 for r in rows if r["source"] == "sticky")
    orphan = sum(1 for r in rows if r["source"] == "none")
    lines.append((f"by arm {owned}   sticky {sticky}   unowned {orphan}",
                  RED if orphan else GREY))

    cv2.rectangle(frame, (0, h - 76), (frame.shape[1], h), (18, 18, 18), -1)
    for i, (text, col) in enumerate(lines):
        cv2.putText(frame, text, (12, h - 52 + i * 20), FONT, 0.5, col, 1)


def _i(p) -> tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max-hands", type=int, default=4,
                    help="two players need four (default: 4)")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--pose-every", type=int, default=2,
                    help="submit a frame to pose every N frames (default: 2)")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--download", action="store_true",
                    help="fetch the pose model if missing (~5.5 MB)")
    ap.add_argument("--sync", action="store_true",
                    help="run pose on the render loop — slow, but honest timing")
    args = ap.parse_args()

    model = resolve_model(args.model, args.download)

    q: queue.Queue = queue.Queue(maxsize=1)
    pipe = VisionPipeline(
        result_queue=q,
        width=args.width, height=args.height,
        camera_index=args.camera,
        max_hands=args.max_hands,
        noise_duration=0.0,
    )
    att = PoseAttributor(model, max_players=args.players)
    worker = None if args.sync else PoseWorker(att)

    pipe.start()
    if worker is not None:
        worker.start()
    print(f"[pose] {args.players} players, {args.max_hands} hand slots, "
          f"pose every {args.pose_every} frames, "
          f"{'synchronous' if args.sync else 'threaded'}")

    pose_on = True
    pose_every = max(1, args.pose_every)
    frame_i = 0
    fps, last_t = 0.0, time.perf_counter()
    rows: list[dict] = []

    try:
        while True:
            payload = None
            while True:
                try:
                    payload = q.get_nowait()
                except queue.Empty:
                    break
            if payload is None:
                time.sleep(0.002)
                continue

            frame = payload["frame"]
            if frame is None:
                continue
            frame = frame.copy()
            hands = payload.get("hands", ())

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 1e-6:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            # Pose is the expensive half, so it runs at a fraction of the hand
            # rate. Ownership only changes when a hand enters or leaves frame,
            # and `attribute()` holds the last verdict across the gap.
            if pose_on and frame_i % pose_every == 0:
                if worker is not None:
                    worker.submit(frame)
                else:
                    att.update(frame)
            frame_i += 1

            rows = att.attribute(hands)

            if pose_on:
                for player in att.players:
                    draw_pose(frame, player)
            draw_attribution(frame, rows)
            draw_hud(frame, att, rows, pose_on, pose_every, fps, worker)

            cv2.imshow("pose attribution", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                pose_on = not pose_on
            elif key == ord("["):
                pose_every = max(1, pose_every - 1)
            elif key == ord("]"):
                pose_every = min(30, pose_every + 1)
            elif key == ord("s"):
                att.strictness = (att.strictness + 1) % len(STRICTNESS)
            elif key == ord("d"):
                print(f"\n-- frame {frame_i}  pose {att.last_ms:.1f} ms "
                      f"players={len(att.players)} arms={len(att.arms)}")
                for row in rows:
                    chain = "".join(t if ok else "-" for t, ok in row["arm"]["chain"]) \
                        if row["arm"] else "...."
                    print(f"   hand {row['slot']}  -> "
                          f"{'P' + str(row['player']) if row['player'] is not None else 'unowned'}"
                          f"  via {row['source']:6s} chain {chain}  "
                          f"d={row['dist']:.0f}px  sign={row['hand'].get('hand_sign')}")
    finally:
        if worker is not None:
            worker.stop()
        pipe.stop()
        att.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
