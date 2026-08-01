"""
pipeline.py — VisionPipeline: background webcam thread for the Visual AI Game Engine.

Detects BOTH:
  • Face coordinates  (MediaPipe FaceDetection or fallback centre)
  • Hand gestures     (MediaPipe Hands)
      – index-fingertip position       → index_pos
      – pinch gesture (thumb+index)    → is_pinching  (debounced)
      – Z-push click (forward finger)  → click_just_fired
      – index-finger isolation         → is_index_isolated

Queue payload (dict)
--------------------
{
  # Face
  "target_x"  : float,   # face centre X (px), or frame centre if no face
  "target_y"  : float,   # face centre Y (px), or frame centre if no face
  "frame"     : ndarray, # BGR camera frame (already flipped & resized)

  # Hand
  "hand_visible"      : bool,
  "index_pos"         : (int, int),   # index fingertip (px)
  "pinch_pos"         : (int, int),   # thumb-index midpoint (px)
  "is_pinching"       : bool,
  "click_just_fired"  : bool,         # True for EXACTLY ONE frame
  "is_index_isolated" : bool,
  "z_delta"           : float,        # raw Z push magnitude (debug)
}
"""

import math
import queue
import threading
import time

import cv2
import numpy as np

# ── Optional MediaPipe imports ────────────────────────────────────────────────
HAS_MEDIAPIPE = False
mp_face_detection_module = None
mp_hands_module = None

try:
    import mediapipe as mp

    # Face detection
    try:
        import mediapipe.solutions.face_detection as mp_face_detection_module
        HAS_MEDIAPIPE = True
    except (ImportError, AttributeError):
        try:
            from mediapipe.python.solutions import face_detection as mp_face_detection_module
            HAS_MEDIAPIPE = True
        except (ImportError, AttributeError):
            pass

    # Hands
    try:
        import mediapipe.solutions.hands as mp_hands_module
    except (ImportError, AttributeError):
        try:
            from mediapipe.python.solutions import hands as mp_hands_module
        except (ImportError, AttributeError):
            mp_hands_module = None

except (ImportError, AttributeError):
    pass

# ── Gesture detection constants ───────────────────────────────────────────────
# Pinch
_PINCH_ENTER_THRESHOLD = 0.06   # Start pinching if closer than this
_PINCH_EXIT_THRESHOLD  = 0.09   # Release pinch if further than this
_PINCH_DEBOUNCE       = 3      # consecutive frames needed to toggle pinch state

# Z-push click
_Z_HISTORY_LEN       = 10      # rolling window size (frames)
_Z_CLICK_THRESHOLD   = 0.055   # minimum Z delta to count as a push (~2 inch deliberate push)
_Z_CLICK_XY_MAX_PX   = 30      # max lateral drift allowed during push (px)
_Z_COOLDOWN_FRAMES   = 40      # frames before another click can fire (~1.3 s at 30 fps)


def _dist2d(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ── EMA helper ────────────────────────────────────────────────────────────────
def _ema(current: float, previous: float, alpha: float = 0.25) -> float:
    return alpha * current + (1.0 - alpha) * previous


# ── Gesture state (per-pipeline instance) ─────────────────────────────────────
class _GestureState:
    """Mutable gesture tracking state — isolated from the thread so it can be
    reset cleanly when the hand disappears."""

    def __init__(self):
        # EMA smoothing
        self.smooth_ix: float = -1.0
        self.smooth_iy: float = -1.0
        self.smooth_px: float = -1.0
        self.smooth_py: float = -1.0

        # Pinch debounce
        self.pinch_consec: int   = 0
        self.release_consec: int = 0
        self.pinch_active: bool  = False

        # Z-push click
        self.z_history: list[float] = []
        self.z_cooldown: int        = 0
        self.z_start_xy             = None   # type: tuple[int,int] | None

        # Finger lock state
        self.locked_pos: tuple[float, float] | None = None
        self.lost_frames: int = 0

    def reset(self):
        self.smooth_ix = self.smooth_iy = -1.0
        self.smooth_px = self.smooth_py = -1.0
        self.pinch_consec = self.release_consec = 0
        self.pinch_active  = False
        self.z_history.clear()
        self.z_cooldown = 0
        self.z_start_xy = None
        self.locked_pos = None
        self.lost_frames = 0


# ── VisionPipeline ─────────────────────────────────────────────────────────────
class VisionPipeline(threading.Thread):
    """
    Background vision thread: captures webcam frames and detects both
    face coordinates and hand gestures.

    Parameters
    ----------
    result_queue : queue.Queue
        Thread-safe queue that receives one dict per processed frame.
    width, height : int
        Target frame dimensions.
    camera_index : int
        OpenCV camera index (default 0).
    smooth_alpha : float
        EMA smoothing factor for landmark positions (0 < α ≤ 1).
        Lower = smoother but laggier; 0.25 is a good default.
    """

    def __init__(
        self,
        result_queue: queue.Queue,
        width: int = 800,
        height: int = 600,
        camera_index: int = 0,
        smooth_alpha: float = 0.20,
    ):
        super().__init__(daemon=True)
        self.result_queue  = result_queue
        self.width         = width
        self.height        = height
        self.camera_index  = camera_index
        self.smooth_alpha  = smooth_alpha
        self.running       = False

        # ── ToF (Time-of-Flight) Depth Sensor State ───────────────────────────
        self.tof_active: bool       = False
        self.tof_simulated: bool    = False
        self.tof_device_name: str   = "None"
        self.depth_map: np.ndarray | None = None

        # ── MediaPipe: Face Detection ─────────────────────────────────────────
        self._mp_face = None
        if HAS_MEDIAPIPE and mp_face_detection_module is not None:
            try:
                self._mp_face = mp_face_detection_module.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
            except Exception as e:
                print(f"[VisionPipeline] FaceDetection init warning: {e}")

        # ── MediaPipe: Hands ──────────────────────────────────────────────────
        self._mp_hands = None
        if mp_hands_module is not None:
            try:
                self._mp_hands = mp_hands_module.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.65,
                )
            except Exception as e:
                print(f"[VisionPipeline] Hands init warning: {e}")

        # ── Gesture state ─────────────────────────────────────────────────────
        self._gs = _GestureState()

    # ── Thread entry ──────────────────────────────────────────────────────────
    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)
        camera_available = cap.isOpened()
        if not camera_available:
            print(
                f"[VisionPipeline] Camera {self.camera_index} not available. "
                "Running simulated vision target."
            )

        sim_angle = 0.0

        while self.running:
            if camera_available:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.flip(frame, 1)   # mirror for natural interaction

                payload = self._process_frame(frame)

            else:
                # ── Simulated mode (no camera) ────────────────────────────────
                sim_angle += 0.05
                target_x = self.width  / 2.0 + math.cos(sim_angle) * 200.0
                target_y = self.height / 2.0 + math.sin(sim_angle) * 150.0

                dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(
                    dummy,
                    "Simulated Vision Mode (No Camera)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                payload = self._empty_payload(target_x, target_y, dummy)
                time.sleep(0.033)

            if not self.result_queue.full():
                self.result_queue.put(payload)

        if cap and cap.isOpened():
            cap.release()

    def stop(self):
        self.running = False

    # ── Frame processing ──────────────────────────────────────────────────────
    def _process_frame(self, bgr_frame: np.ndarray) -> dict:
        """Run face + hand detection on one BGR frame. Returns full payload."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # ── Face detection ────────────────────────────────────────────────────
        target_x = self.width  / 2.0
        target_y = self.height / 2.0

        if self._mp_face is not None:
            face_results = self._mp_face.process(rgb)
            if face_results.detections:
                det  = face_results.detections[0]
                bbox = det.location_data.relative_bounding_box
                target_x = (bbox.xmin + bbox.width  / 2.0) * self.width
                target_y = (bbox.ymin + bbox.height / 2.0) * self.height

        # ── Hand gesture detection ────────────────────────────────────────────
        gesture = self._empty_gesture()

        if self._mp_hands is not None:
            hand_results = self._mp_hands.process(rgb)

            if hand_results.multi_hand_landmarks:
                selected_lm = self._select_locked_hand(hand_results.multi_hand_landmarks)
                if selected_lm is not None:
                    gesture = self._extract_gesture(selected_lm)
                else:
                    self._gs.lost_frames += 1
                    if self._gs.lost_frames > 12:
                        self._gs.reset()
            else:
                self._gs.lost_frames += 1
                if self._gs.lost_frames > 12:
                    self._gs.reset()

        payload = {
            # Face
            "target_x": target_x,
            "target_y": target_y,
            "frame":    bgr_frame,
            # Hand (merged in from gesture dict)
            **gesture,
        }
        return payload

    def _select_locked_hand(self, multi_hand_landmarks):
        """
        Locks onto a single index finger. If multiple hands/fingers are detected,
        returns the landmark set whose index tip is closest to the locked position.
        If locked hand is not found within MAX_LOCK_DIST_PX, returns None until lost_frames expires.
        """
        gs = self._gs
        W, H = self.width, self.height
        MAX_LOCK_DIST_PX = 250.0  # max spatial jump allowed between consecutive frames

        candidates = []
        for hand_lms in multi_hand_landmarks:
            lm = hand_lms.landmark
            ix = lm[8].x * W
            iy = lm[8].y * H
            candidates.append((ix, iy, lm))

        if not candidates:
            return None

        if gs.locked_pos is None:
            # First lock: take first detected hand
            best_ix, best_iy, best_lm = candidates[0]
            gs.locked_pos = (best_ix, best_iy)
            gs.lost_frames = 0
            return best_lm

        # Find candidate closest to current locked position
        lx, ly = gs.locked_pos
        best_lm = None
        best_dist = float('inf')
        best_pos = None

        for ix, iy, lm in candidates:
            dist = math.sqrt((ix - lx) ** 2 + (iy - ly) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_lm = lm
                best_pos = (ix, iy)

        if best_dist <= MAX_LOCK_DIST_PX:
            gs.locked_pos = best_pos
            gs.lost_frames = 0
            return best_lm

        # All candidates are too far from locked position -> likely secondary finger / noise
        return None

    # ── Gesture extraction ────────────────────────────────────────────────────
    def _extract_gesture(self, lm) -> dict:
        """
        Given MediaPipe hand landmarks (already mirrored via frame flip),
        compute and return the full gesture dict.
        """
        gs = self._gs
        W, H = self.width, self.height

        # Raw pixel positions (frame is already flipped, so no extra mirror needed)
        raw_ix = lm[8].x * W   # index tip
        raw_iy = lm[8].y * H
        raw_tx = lm[4].x * W   # thumb tip
        raw_ty = lm[4].y * H

        # EMA smoothing (bootstrap on first frame)
        if gs.smooth_ix < 0:
            gs.smooth_ix = raw_ix;  gs.smooth_iy = raw_iy
            gs.smooth_px = (raw_ix + raw_tx) / 2
            gs.smooth_py = (raw_iy + raw_ty) / 2

        # ── 2. Z-Push Click ───────────────────────────────────────────────────
        z_val = lm[8].z   # MediaPipe Z: more negative = closer to camera
        click_fired, z_delta, xy_drift = self._detect_z_click(z_val, (round(gs.smooth_ix), round(gs.smooth_iy)))

        # Dynamic adaptive EMA alpha: scale based on movement velocity to filter micro jitter while keeping speed
        d_dist = math.sqrt((raw_ix - gs.smooth_ix) ** 2 + (raw_iy - gs.smooth_iy) ** 2)
        scale = max(0.0, min(1.0, (d_dist - 2.0) / 20.0))
        min_alpha = min(0.10, self.smooth_alpha)
        a = min_alpha + scale * (self.smooth_alpha - min_alpha)

        # Heavily dampen X/Y updates if z_delta > 0.015 (active forward push)
        if z_delta > 0.015:
            a = 0.02

        gs.smooth_ix = _ema(raw_ix, gs.smooth_ix, a)
        gs.smooth_iy = _ema(raw_iy, gs.smooth_iy, a)
        raw_px = (raw_ix + raw_tx) / 2
        raw_py = (raw_iy + raw_ty) / 2
        gs.smooth_px = _ema(raw_px, gs.smooth_px, a)
        gs.smooth_py = _ema(raw_py, gs.smooth_py, a)

        index_pos = (round(gs.smooth_ix), round(gs.smooth_iy))
        pinch_pos = (round(gs.smooth_px), round(gs.smooth_py))

        # ── 1. Pinch Detection ────────────────────────────────────────────────
        # Normalised 3-D distance — invariant to hand-camera distance
        pinch_dist_norm = math.sqrt(
            (lm[4].x - lm[8].x) ** 2 +
            (lm[4].y - lm[8].y) ** 2 +
            (lm[4].z - lm[8].z) ** 2 * 0.5   # Z weighted down (noisier)
        )
        
        # Hysteresis: use different thresholds depending on current state
        if gs.pinch_active:
            raw_pinching = pinch_dist_norm < _PINCH_EXIT_THRESHOLD
        else:
            raw_pinching = pinch_dist_norm < _PINCH_ENTER_THRESHOLD

        # Debounce: require N consecutive frames before toggling
        if raw_pinching:
            gs.pinch_consec += 1
            gs.release_consec = 0
            if gs.pinch_consec >= _PINCH_DEBOUNCE:
                gs.pinch_active = True
        else:
            gs.release_consec += 1
            gs.pinch_consec = 0
            if gs.release_consec >= _PINCH_DEBOUNCE:
                gs.pinch_active = False

        # ── 3. Index Isolation ────────────────────────────────────────────────
        # Uses landmark 9 (middle MCP) as a stable palm anchor.
        def dist3d(a_id: int, b_id: int) -> float:
            return math.sqrt(
                (lm[a_id].x - lm[b_id].x) ** 2 +
                (lm[a_id].y - lm[b_id].y) ** 2 +
                (lm[a_id].z - lm[b_id].z) ** 2
            )

        def extended(tip: int, _pip: int, mcp: int) -> bool:
            return dist3d(tip, 9) > dist3d(mcp, 9) * 0.85

        index_ext  = extended(8,  6,  5)
        middle_ext = extended(12, 10, 9)
        ring_ext   = extended(16, 14, 13)
        pinky_ext  = extended(20, 18, 17)
        # Relaxed index isolation: allow middle finger co-extension (natural tendon attachment),
        # requiring only ring and pinky to remain non-extended.
        is_isolated = index_ext and not (ring_ext or pinky_ext)

        # ── 4. ToF Depth Lookup ───────────────────────────────────────────────
        tof_active, tof_z_m, depth_source = self.sample_tof_depth(index_pos[0], index_pos[1], z_val)

        return {
            "hand_visible":      True,
            "index_pos":         index_pos,
            "pinch_pos":         pinch_pos,
            "is_pinching":       gs.pinch_active,
            "click_just_fired":  click_fired,
            "is_index_isolated": is_isolated,
            "z_delta":           z_delta,
            "xy_drift":          xy_drift,
            "tof_active":        tof_active,
            "tof_z_m":           tof_z_m,
            "depth_source":      depth_source,
        }

    # ── ToF Depth Probe & Sampling ─────────────────────────────────────────────
    def sample_tof_depth(self, px: int, py: int, lm_z: float = 0.0) -> tuple[bool, float, str]:
        """
        Samples physical depth (in meters) at 2D (X, Y) pixel coordinates from ToF camera frame.
        Falls back to relative estimation if ToF is inactive.
        """
        if self.tof_active or self.tof_simulated:
            if self.depth_map is not None and 0 <= py < self.depth_map.shape[0] and 0 <= px < self.depth_map.shape[1]:
                # 16-bit depth values in mm -> converted to meters
                raw_depth = self.depth_map[py, px]
                z_m = float(raw_depth) / 1000.0 if raw_depth > 0 else 0.45 + (lm_z * 0.5)
            else:
                # Simulated hardware ToF depth reading centered around 0.45m calibrated baseline
                z_m = round(max(0.15, 0.45 + (lm_z * 0.6)), 3)
            src_label = "ToF IR Hardware" if self.tof_active else "ToF Hardware (Simulated)"
            return True, z_m, src_label

        return False, 0.0, "RGB MediaPipe Estimate"

    # ── Z-push click algorithm ────────────────────────────────────────────────
    def _detect_z_click(self, z_now: float, xy_now: tuple) -> tuple:
        """
        Returns (fired: bool, delta_z: float, drift: float).

        Algorithm
        ---------
        Keep a rolling window of _Z_HISTORY_LEN Z values.
        A click fires when:
          delta_z = z_history[0] - z_now  >= _Z_CLICK_THRESHOLD
          (more negative → moving toward camera)
        AND lateral XY drift since push started < _Z_CLICK_XY_MAX_PX.
        A cooldown prevents double-firing.
        """
        gs = self._gs
        drift = 0.0

        if gs.z_cooldown > 0:
            gs.z_cooldown -= 1
            gs.z_history.append(z_now)
            if len(gs.z_history) > _Z_HISTORY_LEN:
                gs.z_history.pop(0)
            return False, 0.0, 0.0

        gs.z_history.append(z_now)
        if len(gs.z_history) > _Z_HISTORY_LEN:
            gs.z_history.pop(0)
        if len(gs.z_history) < _Z_HISTORY_LEN:
            return False, 0.0, 0.0

        z_baseline = gs.z_history[0]
        delta_z    = z_baseline - z_now    # positive = moving toward camera

        if gs.z_start_xy is not None:
            drift = _dist2d(gs.z_start_xy, xy_now)

        if delta_z >= _Z_CLICK_THRESHOLD:
            if gs.z_start_xy is None:
                gs.z_start_xy = xy_now
                drift = 0.0
            else:
                drift = _dist2d(gs.z_start_xy, xy_now)

            if drift < _Z_CLICK_XY_MAX_PX:
                # ✓ Valid Z-push click
                gs.z_history.clear()
                gs.z_start_xy = None
                gs.z_cooldown = _Z_COOLDOWN_FRAMES
                return True, delta_z, drift
            else:
                # Too much lateral movement — reset
                gs.z_history.clear()
                gs.z_start_xy = None
        else:
            # Push hasn't started — keep refreshing baseline XY
            gs.z_start_xy = xy_now

        return False, delta_z, drift

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _empty_gesture() -> dict:
        return {
            "hand_visible":      False,
            "index_pos":         (0, 0),
            "pinch_pos":         (0, 0),
            "is_pinching":       False,
            "click_just_fired":  False,
            "is_index_isolated": False,
            "z_delta":           0.0,
            "xy_drift":          0.0,
            "tof_active":        False,
            "tof_z_m":           0.0,
            "depth_source":      "RGB MediaPipe Estimate",
        }

    def _empty_payload(self, tx: float, ty: float, frame: np.ndarray) -> dict:
        return {
            "target_x": tx,
            "target_y": ty,
            "frame":    frame,
            **self._empty_gesture(),
        }
