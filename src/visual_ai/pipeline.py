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
  "index_pos"         : (int, int),   # index fingertip, smoothed (px)
  "thumb_pos"         : (int, int),   # thumb tip, raw (px)
  "middle_pos"        : (int, int),   # middle tip, raw (px)
  "pinch_pos"         : (int, int),   # 3-finger centroid, smoothed (px)
  "pinch_pos_raw"     : (int, int),   # 3-finger centroid, unsmoothed (px)
  "is_pinching"       : bool,
  "click_just_fired"  : bool,         # True for EXACTLY ONE frame
  "is_index_isolated" : bool,

  # Hand signs (see classify_hand_sign)
  "hand_sign"         : str,          # "fist" | "open_palm" | "point" | "peace" | "unknown"
  "fingers_extended"  : (bool,) * 5,  # thumb, index, middle, ring, pinky
  "is_fist"           : bool,
  "is_open_palm"      : bool,
  "grip_openness"     : float,        # 0.0 curled fist … 1.0 fingers straight

  "smoothing_enabled" : bool,         # One-Euro landmark smoothing currently on?
  "z_delta"           : float,        # raw Z push magnitude (debug)
  "xy_drift"          : float,        # lateral drift during a Z push (px)

  # 3-finger pinch
  "is_3_finger_pinching" : bool,

  # Depth (see tof_stabilizer.py)
  "tof_active"           : bool,
  "tof_z_m"              : float,     # stabilized depth (m)
  "tof_z_raw"            : float,     # depth before stabilization (m)
  "depth_source"         : str,
  "stabilizer_state"     : str,       # "inactive" | "sampling" | "active"
  "stabilizer_progress"  : float,
  "stabilizer_noise_amp" : float,     # measured vibration std-dev (m)
  "stabilizer_gate"      : float,     # current noise-gate half-width (m)

  # Jitter metrics (see jitter_analyzer.py) — always a full stat dict
  "jitter"               : dict,
}

Every key above is present on EVERY payload, including frames with no hand and
simulated-camera frames, so consumers can index directly.
"""

import math
import queue
import threading
import time

import cv2
import numpy as np

from visual_ai.noise_filter import (
    OneEuroFilter,
    PipelineNoiseFilter,
    ema_alpha_to_cutoff,
)
from visual_ai.jitter_analyzer import JitterAnalyzer
from visual_ai.tof_stabilizer import ToFStabilizer

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
_Z_MIN_HISTORY       = 4       # frames needed before the window can fire
_Z_CLICK_THRESHOLD   = 0.012   # minimum MediaPipe-Z delta to count as a push
_Z_CLICK_THRESHOLD_M = 0.030   # minimum ToF depth delta to count as a push (m)
_Z_CLICK_XY_MAX_PX   = 30      # max lateral drift allowed during push (px)
_Z_COOLDOWN_FRAMES   = 25      # frames before another click can fire (~0.8 s at 30 fps)

# ToF depth sampling
_TOF_PATCH_RADIUS    = 2       # median filter half-width (px) around the sample point

# ── Hand signs ────────────────────────────────────────────────────────────────
SIGN_FIST      = "fist"
SIGN_OPEN_PALM = "open_palm"
SIGN_POINT     = "point"
SIGN_PEACE     = "peace"
SIGN_UNKNOWN   = "unknown"

# Frames a new sign must hold before it replaces the reported one. The hand
# passes through ambiguous shapes while opening and closing, and without this a
# grab-then-release would emit a burst of spurious signs in between.
_SIGN_DEBOUNCE = 3


def classify_hand_sign(fingers_extended) -> str:
    """
    Map five finger-extension booleans to a named sign.

    Parameters
    ----------
    fingers_extended : tuple[bool, bool, bool, bool, bool]
        (thumb, index, middle, ring, pinky)

    The thumb is deliberately ignored for `fist`: plenty of people close a fist
    with the thumb resting alongside the fingers rather than across them, which
    is a genuinely abducted thumb, so requiring a tucked thumb would make fists
    unreliable. It is still required for `open_palm`, where it is unambiguous.
    """
    thumb, index, middle, ring, pinky = fingers_extended
    four = (index, middle, ring, pinky)
    count = sum(four)

    if count == 0:
        return SIGN_FIST
    if count == 4 and thumb:
        return SIGN_OPEN_PALM
    if count == 1 and index:
        return SIGN_POINT
    if count == 2 and index and middle:
        return SIGN_PEACE
    return SIGN_UNKNOWN


def _dist2d(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ── EMA helper ────────────────────────────────────────────────────────────────
def _ema(current: float, previous: float, alpha: float = 0.25) -> float:
    return alpha * current + (1.0 - alpha) * previous


# ── Gesture state (per-pipeline instance) ─────────────────────────────────────
class _GestureState:
    """Mutable gesture tracking state — isolated from the thread so it can be
    reset cleanly when the hand disappears."""

    def __init__(self, make_filter=None):
        # Position smoothing. `make_filter` supplies a fresh OneEuroFilter so
        # tuning lives on the pipeline while the state object stays reusable.
        self._make_filter = make_filter
        self.index_filter = make_filter() if make_filter else None
        self.centroid_filter = make_filter() if make_filter else None

        # Latest smoothed values. `smoothed` is an explicit initialisation flag:
        # the old code bootstrapped on `smooth_ix < 0`, but MediaPipe emits
        # negative landmark coordinates whenever a fingertip leaves the frame,
        # which silently re-seeded the filter mid-gesture and produced a snap.
        self.smoothed: bool = False
        self.smooth_ix: float = 0.0
        self.smooth_iy: float = 0.0
        self.smooth_px: float = 0.0
        self.smooth_py: float = 0.0

        # Pinch debounce
        self.pinch_consec: int   = 0
        self.release_consec: int = 0
        self.pinch_active: bool  = False

        # Z-push click — entries are (z_value, (x, y)) so each sample keeps the
        # position it was taken at, for the lateral-drift guard.
        self.z_history: list[tuple[float, tuple[int, int]]] = []
        self.z_cooldown: int        = 0
        self.z_start_xy             = None   # type: tuple[int,int] | None

        # Frames since a hand was last seen (drives the reset below)
        self.lost_frames: int = 0

        # 3-Finger Pinch state
        self.was_3_pinching: bool = False        # Edge trigger for 3-finger pinch

        # Hand sign debounce
        self.sign: str = SIGN_UNKNOWN            # currently reported sign
        self.sign_candidate: str = SIGN_UNKNOWN  # sign awaiting confirmation
        self.sign_consec: int = 0                # frames the candidate has held

    def reset(self):
        self.smoothed = False
        self.smooth_ix = self.smooth_iy = 0.0
        self.smooth_px = self.smooth_py = 0.0
        if self.index_filter is not None:
            self.index_filter.reset()
        if self.centroid_filter is not None:
            self.centroid_filter.reset()
        self.pinch_consec = self.release_consec = 0
        self.pinch_active  = False
        self.z_history.clear()
        self.z_cooldown = 0
        self.z_start_xy = None
        self.lost_frames = 0
        self.was_3_pinching = False
        self.sign = SIGN_UNKNOWN
        self.sign_candidate = SIGN_UNKNOWN
        self.sign_consec = 0


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
        movement_magnification: float = 2.0,
        noise_duration: float = 0.0,
        filter_beta: float = 0.006,
        capture_fps: float = 30.0,
        enable_z_click: bool = False,
    ):
        super().__init__(daemon=True)
        self.result_queue  = result_queue
        self.width         = width
        self.height        = height
        self.camera_index  = camera_index
        self.smooth_alpha  = smooth_alpha
        self.movement_magnification = movement_magnification
        self.noise_filter  = PipelineNoiseFilter(noise_duration=noise_duration)
        self.tof_stabilizer = ToFStabilizer()
        self.running       = False

        # ── Landmark smoothing (One-Euro) ─────────────────────────────────────
        # `smooth_alpha` is reinterpreted as the *resting* smoothness: it maps to
        # the One-Euro min_cutoff, and `filter_beta` opens the cutoff up as the
        # hand speeds up. The previous adaptive-EMA ramp topped out at
        # smooth_alpha itself (0.10 → 0.20), so even a 150 px/frame swipe was
        # filtered with a ~5-frame time constant and lagged visibly.
        self.capture_fps  = max(1.0, float(capture_fps))
        self.filter_beta  = max(0.0, float(filter_beta))
        self.filter_min_cutoff = ema_alpha_to_cutoff(smooth_alpha, self.capture_fps)

        # Depth-push click is opt-in; see `_detect_z_click`.
        self.enable_z_click = bool(enable_z_click)

        # One-Euro landmark smoothing can be bypassed at runtime so raw
        # MediaPipe positions reach the consumer. This is separate from the ToF
        # stabilizer below — that one gates depth, this one gates X/Y.
        self.smoothing_enabled: bool = True

        # ── ToF (Time-of-Flight) Depth Sensor State ───────────────────────────
        self.tof_active: bool       = False
        self.tof_simulated: bool    = False
        self.tof_device_name: str   = "None"
        self.depth_map: np.ndarray | None = None
        
        self.disable_camera: bool   = False

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

        # ── Jitter Analyzer ───────────────────────────────────────────────────
        self.jitter_analyzer = JitterAnalyzer(window_size=30)

        # ── Gesture state ─────────────────────────────────────────────────────
        self._gs = _GestureState(make_filter=self._make_position_filter)

        self.last_error: str | None = None
        self.camera_available: bool = False

    def get_status(self) -> dict:
        """Get diagnostic status of the pipeline thread and hardware sources."""
        return {
            "running": self.running,
            "camera_available": self.camera_available,
            "disable_camera": getattr(self, "disable_camera", False),
            "has_mediapipe": HAS_MEDIAPIPE,
            "face_tracking": self._mp_face is not None,
            "hand_tracking": self._mp_hands is not None,
            "last_error": self.last_error,
        }

    # ── Landmark smoothing ────────────────────────────────────────────────────

    def _make_position_filter(self) -> OneEuroFilter:
        """Build a One-Euro filter for a 2D landmark stream using current tuning."""
        return OneEuroFilter(
            freq=self.capture_fps,
            min_cutoff=self.filter_min_cutoff,
            beta=self.filter_beta,
            d_cutoff=1.0,
        )

    def _apply_filter_tuning(self) -> None:
        """Push current tuning onto the live filters without dropping their state."""
        for f in (self._gs.index_filter, self._gs.centroid_filter):
            if f is not None:
                f.min_cutoff = self.filter_min_cutoff
                f.beta = self.filter_beta

    def set_smooth_alpha(self, alpha: float) -> None:
        """
        Update resting smoothness. Lower = steadier when the hand is still,
        at the cost of a little more lag. Takes effect immediately.
        """
        self.smooth_alpha = max(0.01, min(0.99, float(alpha)))
        self.filter_min_cutoff = ema_alpha_to_cutoff(self.smooth_alpha, self.capture_fps)
        self._apply_filter_tuning()

    def set_filter_tuning(self, min_cutoff: float | None = None,
                          beta: float | None = None) -> None:
        """
        Tune the One-Euro landmark filter directly.

        Parameters
        ----------
        min_cutoff : float, optional
            Cutoff in Hz at zero speed. Lower = less resting jitter, more lag.
        beta : float, optional
            Speed coupling in Hz per (px/s). Higher = snappier on fast motion.
            Raise this first if tracking feels laggy while swiping.
        """
        if min_cutoff is not None:
            self.filter_min_cutoff = max(0.01, float(min_cutoff))
        if beta is not None:
            self.filter_beta = max(0.0, float(beta))
        self._apply_filter_tuning()

    def set_smoothing_enabled(self, enabled: bool) -> None:
        """
        Turn One-Euro landmark smoothing on or off.

        When off, `index_pos` and `pinch_pos` carry raw MediaPipe positions.
        Input becomes maximally responsive but visibly jittery at rest. The
        filters keep running underneath so re-enabling does not cause a jump.
        """
        self.smoothing_enabled = bool(enabled)

    def toggle_smoothing(self) -> bool:
        """Flip landmark smoothing and return the new state."""
        self.smoothing_enabled = not self.smoothing_enabled
        return self.smoothing_enabled

    def set_movement_magnification(self, mag: float):
        """Dynamically update movement magnification factor for input tracking and gesture pinch scaling."""
        self.movement_magnification = max(0.5, float(mag))

    def set_noise_duration(self, duration: float):
        """Dynamically update noise filter window duration in seconds."""
        self.noise_filter.filter.set_duration(duration)

    def reset_noise_filter(self):
        """Reset noise filter timing window for state changes / restarts."""
        self.noise_filter.reset()

    # ── ToF Stabilization public API ──────────────────────────────────────────

    def begin_stabilization(self, duration: float = 3.0) -> None:
        """
        Start a ToF lid-shake calibration run.

        While active, a warning overlay ("Please do not move or change position")
        is drawn directly onto every outgoing frame for ``duration`` seconds.
        After calibration, ``tof_z_m`` in the queue payload is automatically
        corrected by the measured ambient offset.

        Parameters
        ----------
        duration : float
            Calibration window in seconds. 3.0 or 5.0 recommended.
        """
        if not (self.tof_active or self.tof_simulated):
            print(
                "[VisionPipeline] begin_stabilization() called with no ToF source — "
                "set pipeline.tof_simulated = True or attach a sensor first. "
                "Calibration will collect no samples and abort."
            )
        self.tof_stabilizer.begin(duration)

    def cancel_stabilization(self) -> None:
        """Abort an in-progress calibration (the overlay's 'Press X to cancel')."""
        self.tof_stabilizer.cancel()

    def disable_stabilization(self) -> None:
        """Disable ToF stabilization and clear the calibrated offset."""
        self.tof_stabilizer.disable()

    # ── Thread entry ──────────────────────────────────────────────────────────
    def run(self):
        self.running = True
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_index)
            self.camera_available = cap.isOpened()
        except Exception as e:
            self.last_error = f"Camera initialization failed: {e}"
            self.camera_available = False

        if not self.camera_available:
            msg = f"[VisionPipeline] Camera index {self.camera_index} not available."
            if self.last_error:
                msg += f" Details: {self.last_error}"
            print(f"{msg} Running simulated vision target.")

        sim_angle = 0.0

        try:
            while self.running:
                if self.camera_available and cap is not None and not getattr(self, 'disable_camera', False):
                    try:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            time.sleep(0.01)
                            continue

                        frame = cv2.resize(frame, (self.width, self.height))
                        frame = cv2.flip(frame, 1)   # mirror for natural interaction

                        payload = self._process_frame(frame)
                    except Exception as frame_err:
                        self.last_error = f"Frame read/process error: {frame_err}"
                        time.sleep(0.01)
                        continue
                else:
                    if self.camera_available and cap is not None:
                        cap.grab() # Keep buffer drained while disabled
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

                # Advance the stabilization clock from the capture loop, not from
                # gesture extraction. Samples only arrive while a hand is in view,
                # so a calibration started with no hand present used to hang in
                # "sampling" forever with the warning overlay stuck on screen.
                self.tof_stabilizer.tick()

                # Apply Noise Filter to suppress transient clicks/pinches during warmup window
                payload = self.noise_filter.process_payload(payload)

                # ── ToF Stabilizer: draw warning overlay during calibration ──────
                if (
                    self.tof_stabilizer.state == ToFStabilizer.STATE_SAMPLING
                    and payload is not None
                    and "frame" in payload
                    and payload["frame"] is not None
                ):
                    payload["frame"] = self._draw_stabilizer_warning(
                        payload["frame"], self.tof_stabilizer.progress
                    )

                # Push to queue (drop oldest frame if consumer is lagging)
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put_nowait(payload)
                time.sleep(0.01)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.running = False

    def stop(self):
        """Stop the background pipeline thread."""
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
                self._gs.lost_frames = 0
                gesture = self._extract_gesture(hand_results.multi_hand_landmarks[0].landmark)
            else:
                self._gs.lost_frames += 1
                if self._gs.lost_frames > 12:
                    self._gs.reset()
                    self.jitter_analyzer.reset()

        payload = {
            # Face
            "target_x": target_x,
            "target_y": target_y,
            "frame":    bgr_frame,
            # Hand (merged in from gesture dict)
            **gesture,
        }
        return payload

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
        raw_mx = lm[12].x * W  # middle tip
        raw_my = lm[12].y * H

        thumb_pos  = (round(raw_tx), round(raw_ty))
        index_pos  = (round(raw_ix), round(raw_iy))
        middle_pos = (round(raw_mx), round(raw_my))

        # Centroid of the 3 fingers
        c_x = (raw_tx + raw_ix + raw_mx) / 3.0
        c_y = (raw_ty + raw_iy + raw_my) / 3.0
        centroid_pos = (round(c_x), round(c_y))

        z_val = lm[8].z   # MediaPipe Z

        # ── Landmark smoothing (One-Euro, velocity-adaptive) ──────────────────
        # Heavy smoothing at rest kills fingertip jitter; the cutoff opens up
        # with hand speed so fast motion stays lag-free. A fixed-alpha EMA
        # cannot do both, which is why tracking previously felt either shaky
        # or laggy depending on which way `smooth_alpha` was pushed.
        now = time.time()
        gs.smooth_ix, gs.smooth_iy = gs.index_filter.filter((raw_ix, raw_iy), now)
        gs.smooth_px, gs.smooth_py = gs.centroid_filter.filter((c_x, c_y), now)
        gs.smoothed = True

        if self.smoothing_enabled:
            index_pos = (round(gs.smooth_ix), round(gs.smooth_iy))
            pinch_pos = (round(gs.smooth_px), round(gs.smooth_py))
        else:
            # Filters above still ran, so re-enabling resumes from current state
            # instead of snapping from a stale one.
            index_pos = (round(raw_ix), round(raw_iy))
            pinch_pos = (round(c_x), round(c_y))

        # ── 1. 2-Finger Pinch (DISABLED — superseded by the 3-finger lock below)
        # The thumb/index distance test fired constantly at frame edges where
        # MediaPipe compresses the hand, so the 3-finger centroid test replaced
        # it. Kept out deliberately rather than left half-wired.

        # Z-push click is evaluated after the ToF section below, since it wants
        # the stabilized depth when a sensor is present.
        click_fired = False
        z_delta = 0.0
        xy_drift = 0.0

        # ── 3-Finger Pinch Trigger ───────────────────────────────────────────
        # Distance from each fingertip to centroid
        dist_t_c = math.sqrt((raw_tx - c_x)**2 + (raw_ty - c_y)**2)
        dist_i_c = math.sqrt((raw_ix - c_x)**2 + (raw_iy - c_y)**2)
        dist_m_c = math.sqrt((raw_mx - c_x)**2 + (raw_my - c_y)**2)
        max_dist_to_centroid = max(dist_t_c, dist_i_c, dist_m_c)

        base_pinch_radius = 55.0
        effective_pinch_radius = base_pinch_radius * max(1.0, float(self.movement_magnification))
        is_3_pinching = max_dist_to_centroid < effective_pinch_radius

        # Trigger action on the rising edge of a 3-finger pinch
        if is_3_pinching and not gs.was_3_pinching:
            click_fired = True
        gs.was_3_pinching = is_3_pinching
        gs.pinch_active = is_3_pinching

        # ── 3. Finger extension & hand sign ───────────────────────────────────
        # Anchored at the wrist (landmark 0): a finger counts as extended when
        # its tip sits further from the wrist than its own PIP joint. This is
        # tolerant of hand rotation and works for the thumb too, which the
        # previous landmark-9 anchor could not measure meaningfully.
        def dist3d(a_id: int, b_id: int) -> float:
            return math.sqrt(
                (lm[a_id].x - lm[b_id].x) ** 2 +
                (lm[a_id].y - lm[b_id].y) ** 2 +
                (lm[a_id].z - lm[b_id].z) ** 2
            )

        def extended(tip: int, pip: int) -> bool:
            return dist3d(tip, 0) > dist3d(pip, 0)

        # The wrist anchor is degenerate for the thumb: landmark 2 (thumb MCP)
        # sits almost on the wrist, so the tip clears it even with the thumb
        # folded flat across the palm — the T pip read as extended nearly all
        # the time. Measure abduction instead: a tucked thumb travels toward the
        # pinky MCP (17) while an extended one swings away from it, so compare
        # the tip against its own IP joint with a palm-scaled margin so a
        # resting thumb cannot flicker.
        palm_span  = max(dist3d(0, 17), 1e-6)
        thumb_ext  = (dist3d(4, 17) - dist3d(3, 17)) > 0.10 * palm_span
        index_ext  = extended(8,  6)
        middle_ext = extended(12, 10)
        ring_ext   = extended(16, 14)
        pinky_ext  = extended(20, 18)
        fingers_extended = (thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext)

        # Relaxed index isolation: allow middle finger co-extension (natural tendon attachment),
        # requiring only ring and pinky to remain non-extended.
        is_isolated = index_ext and not (ring_ext or pinky_ext)

        # Debounced so a half-closed hand mid-transition cannot flap the sign
        # back and forth and fire a game action on every other frame.
        hand_sign = self._debounce_sign(classify_hand_sign(fingers_extended))

        # Continuous grip measure, deliberately NOT debounced and not
        # thresholded into booleans. `hand_sign` needs _SIGN_DEBOUNCE frames of
        # one steady sign before it changes, and a hand on its way open walks
        # through several signs (unknown → point → peace → …), each one
        # restarting that count — so `is_fist` can keep reading True for a dozen
        # frames after the fingers have visibly started to move. Anything that
        # must react the instant a grip *begins* to open (a slingshot release, a
        # throw) should threshold this instead.
        #   0.0 = every finger curled into the palm, 1.0 = fingers straight out.
        # Per finger, how far the tip reaches past its own PIP joint, measured
        # from the wrist and scaled by the palm so hand size and camera distance
        # drop out.
        def curl(tip: int, pip: int) -> float:
            reach = (dist3d(tip, 0) - dist3d(pip, 0)) / (0.5 * palm_span)
            return min(1.0, max(0.0, reach))

        grip_openness = sum(
            curl(tip, pip) for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
        ) / 4.0

        # ── 4. ToF Depth Lookup ───────────────────────────────────────────────
        tof_active, tof_z_raw, depth_source = self.sample_tof_depth(index_pos[0], index_pos[1], z_val)

        # Feed raw sample to calibrator during sampling window
        if self.tof_stabilizer.state == ToFStabilizer.STATE_SAMPLING:
            self.tof_stabilizer.feed(tof_z_raw)

        # Gate ambient vibration out of Z (no-op when inactive)
        tof_z_m = self.tof_stabilizer.correct(tof_z_raw)

        # ── 5. Z-Push Click (opt-in via enable_z_click) ───────────────────────
        # Driven by stabilized metric depth when a ToF source is present, so the
        # threshold is in real metres and can be floored by the stabilizer's
        # measured noise gate. Falls back to MediaPipe's relative Z otherwise.
        if self.enable_z_click:
            if tof_active:
                click_fired, z_delta, xy_drift = self._detect_z_click(
                    tof_z_m, index_pos,
                    threshold=max(_Z_CLICK_THRESHOLD_M, self.tof_stabilizer.noise_gate * 2.0),
                )
            else:
                # MediaPipe Z decreases toward the camera, same sense as depth.
                click_fired, z_delta, xy_drift = self._detect_z_click(
                    z_val, index_pos, threshold=_Z_CLICK_THRESHOLD,
                )

        # ── 6. Jitter Calculation ──────────────────────────────────────────────
        jitter_stats = self.jitter_analyzer.update((raw_ix, raw_iy), (gs.smooth_ix, gs.smooth_iy))

        return {
            "hand_visible":        True,
            "index_pos":           index_pos,
            "thumb_pos":           thumb_pos,
            "middle_pos":          middle_pos,
            # Smoothed centroid. This previously reported `centroid_pos` — the
            # raw, unfiltered value — so the smoothing computed for it every
            # frame was discarded and consumers got a jittery anchor point.
            "pinch_pos":           pinch_pos,
            "pinch_pos_raw":       centroid_pos,
            "is_pinching":         gs.pinch_active,
            "click_just_fired":    click_fired,
            "is_index_isolated":   is_isolated,
            # Hand signs
            "hand_sign":           hand_sign,
            "fingers_extended":    fingers_extended,
            "is_fist":             hand_sign == SIGN_FIST,
            "is_open_palm":        hand_sign == SIGN_OPEN_PALM,
            "grip_openness":       grip_openness,
            "smoothing_enabled":   self.smoothing_enabled,
            "z_delta":             z_delta,
            "xy_drift":            xy_drift,
            "tof_active":          tof_active,
            "tof_z_m":             tof_z_m,
            "tof_z_raw":           tof_z_raw,
            "depth_source":        depth_source,
            "is_3_finger_pinching":is_3_pinching,
            "jitter":              jitter_stats,
            # Stabilizer telemetry
            "stabilizer_state":     self.tof_stabilizer.state,
            "stabilizer_progress":  self.tof_stabilizer.progress,
            "stabilizer_noise_amp": self.tof_stabilizer.z_noise_amplitude,
            "stabilizer_gate":      self.tof_stabilizer.noise_gate,
        }

    # ── Hand sign debounce ────────────────────────────────────────────────────
    def _debounce_sign(self, raw_sign: str) -> str:
        """
        Only report a sign once it has held for `_SIGN_DEBOUNCE` frames.

        Returns the currently confirmed sign, which may be the previous one
        while a new candidate is still being confirmed.
        """
        gs = self._gs

        if raw_sign == gs.sign:
            gs.sign_candidate = raw_sign
            gs.sign_consec = 0
            return gs.sign

        if raw_sign == gs.sign_candidate:
            gs.sign_consec += 1
        else:
            gs.sign_candidate = raw_sign
            gs.sign_consec = 1

        if gs.sign_consec >= _SIGN_DEBOUNCE:
            gs.sign = raw_sign
            gs.sign_consec = 0

        return gs.sign

    # ── ToF Depth Probe & Sampling ─────────────────────────────────────────────
    def sample_tof_depth(self, px: int, py: int, lm_z: float = 0.0) -> tuple[bool, float, str]:
        """
        Samples physical depth (in meters) at 2D (X, Y) pixel coordinates from ToF camera frame.
        Falls back to relative estimation if ToF is inactive.
        """
        if not (self.tof_active or self.tof_simulated):
            return False, 0.0, "RGB MediaPipe Estimate"

        z_m = None

        if self.depth_map is not None and self.depth_map.size:
            dh, dw = self.depth_map.shape[:2]

            # The depth map rarely matches the RGB frame resolution, and the
            # incoming pixel is in RGB frame space. Sampling it directly read
            # the wrong part of the sensor (or fell out of bounds and silently
            # dropped to the simulated branch) on any sensor whose resolution
            # differs from width x height.
            dx = int(px * dw / max(1, self.width))
            dy = int(py * dh / max(1, self.height))
            dx = max(0, min(dw - 1, dx))
            dy = max(0, min(dh - 1, dy))

            # Median of a small patch. A single ToF pixel is extremely noisy and
            # frequently reads 0 (no return) at object edges — exactly where a
            # fingertip sits.
            r = _TOF_PATCH_RADIUS
            patch = self.depth_map[
                max(0, dy - r): min(dh, dy + r + 1),
                max(0, dx - r): min(dw, dx + r + 1),
            ]
            valid = patch[patch > 0]
            if valid.size:
                # 16-bit depth values in mm -> converted to meters
                z_m = float(np.median(valid)) / 1000.0

        if z_m is None:
            # Simulated hardware ToF depth reading centered around 0.45m calibrated baseline
            z_m = 0.45 + (lm_z * 0.6)

        z_m = round(max(0.15, z_m), 4)
        src_label = "ToF IR Hardware" if self.tof_active else "ToF Hardware (Simulated)"
        return True, z_m, src_label

    # ── Z-push click algorithm ────────────────────────────────────────────────
    def _detect_z_click(self, z_now: float, xy_now: tuple,
                        threshold: float | None = None) -> tuple:
        """
        Returns (fired: bool, delta_z: float, drift: float).

        Algorithm
        ---------
        Keep a rolling window of the last _Z_HISTORY_LEN Z values, paired with
        the XY position each was sampled at. A click fires when

            delta_z = max(window) - z_now  >= threshold

        (Z decreases toward the camera in both the ToF-metres and MediaPipe
        conventions, so a push always yields a positive delta.)

        AND the lateral drift between the frame that set the window maximum and
        now is under _Z_CLICK_XY_MAX_PX. A cooldown prevents double-firing.

        Fixes over the original
        -----------------------
        * Baseline was ``z_history[0]``, the exact 10-frames-ago sample. Any
          slow forward push slid the baseline along with the hand and never
          reached the threshold. The window maximum — the furthest point the
          hand has been recently — is the actual start of the push.
        * ``z_start_xy`` was reassigned to the current position on every
          below-threshold frame, so "drift since the push started" was really
          "drift since last frame" and the lateral-movement guard let swipes
          through. The anchor now travels with the baseline sample.
        * A fired click cleared the whole history, forcing _Z_HISTORY_LEN frames
          of refill *after* the cooldown before another push could register.
        """
        gs = self._gs
        thr = _Z_CLICK_THRESHOLD if threshold is None else float(threshold)

        # History entries are (z, xy) so the baseline keeps its own anchor.
        gs.z_history.append((float(z_now), xy_now))
        if len(gs.z_history) > _Z_HISTORY_LEN:
            gs.z_history.pop(0)

        if gs.z_cooldown > 0:
            gs.z_cooldown -= 1
            return False, 0.0, 0.0

        if len(gs.z_history) < _Z_MIN_HISTORY:
            return False, 0.0, 0.0

        # Furthest recent point = where the push began.
        base_z, base_xy = max(gs.z_history[:-1], key=lambda e: e[0])
        delta_z = base_z - z_now
        drift = _dist2d(base_xy, xy_now)

        if delta_z >= thr:
            if drift < _Z_CLICK_XY_MAX_PX:
                gs.z_cooldown = _Z_COOLDOWN_FRAMES
                # Drop only the pre-push samples; the recent ones stay valid.
                gs.z_history = gs.z_history[-2:]
                return True, delta_z, drift
            # Lateral swipe, not a push — restart the window from here.
            gs.z_history = gs.z_history[-1:]
            return False, delta_z, drift

        return False, delta_z, drift

    # ── Stabilizer Warning Overlay ────────────────────────────────────────────
    def _draw_stabilizer_warning(self, frame: np.ndarray, progress: float) -> np.ndarray:
        """
        Draw a full-screen warning overlay onto the BGR frame during stabilization
        calibration. Called by the run() loop before pushing payload to queue,
        so every consumer game automatically shows this without extra code.

        Parameters
        ----------
        frame    : np.ndarray  BGR camera frame (already resized & flipped)
        progress : float       calibration progress 0.0 → 1.0
        """
        h, w = frame.shape[:2]

        # ── 1. Dark semi-transparent wash ────────────────────────────────────
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:] = (15, 10, 30)                    # deep navy tint
        frame = cv2.addWeighted(frame, 0.22, overlay, 0.78, 0)

        # ── 2. Text blocks ───────────────────────────────────────────────────
        scale_ref = w / 800.0

        header_text  = "!! LID-SHAKE STABILIZATION !!"
        message_text = "Please do not move or change position"
        secs_text    = f"Calibrating...  {self.tof_stabilizer.time_remaining:.1f}s remaining"
        hint_text    = "Press  X  to cancel"

        text_specs = [
            # (text,          y_ratio, font_scale,        color BGR,          thickness)
            (header_text,    0.32,    1.05 * scale_ref,  (0, 140, 255),      3),
            (message_text,   0.46,    0.75 * scale_ref,  (255, 255, 255),    2),
            (secs_text,      0.58,    0.65 * scale_ref,  (0, 220, 200),      2),
            (hint_text,      0.90,    0.50 * scale_ref,  (140, 140, 140),    1),
        ]

        for text, y_ratio, scale, color, thick in text_specs:
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
            tx = (w - tw) // 2
            ty = int(h * y_ratio)
            # Drop-shadow
            cv2.putText(frame, text, (tx + 2, ty + 2),
                        cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 2)
            cv2.putText(frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_DUPLEX, scale, color, thick)

        # ── 3. Progress bar ──────────────────────────────────────────────────
        bar_w  = int(w * 0.60)
        bar_h  = int(14 * scale_ref)
        bar_x  = (w - bar_w) // 2
        bar_y  = int(h * 0.68)
        radius = max(1, bar_h // 2)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 60), -1)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), (0, 180, 255), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (100, 100, 110), 1)
        _ = radius   # suppress unused warning

        # ── 4. Countdown ring ────────────────────────────────────────────────
        cx     = w // 2
        cy     = int(h * 0.80)
        ring_r = int(28 * scale_ref)
        angle_end = int(360 * progress)             # arc sweeps 0 → 360 as time passes

        cv2.circle(frame, (cx, cy), ring_r, (55, 55, 65), 3)
        if angle_end > 0:
            cv2.ellipse(frame, (cx, cy), (ring_r, ring_r),
                        -90, 0, angle_end, (0, 180, 255), 3)

        return frame

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _empty_gesture(self) -> dict:
        """
        No-hand payload. Must carry every key `_extract_gesture` emits — a
        consumer that indexes `payload["is_3_finger_pinching"]` should not
        start raising KeyError the moment the hand leaves the frame.
        """
        return {
            "hand_visible":      False,
            "index_pos":         (0, 0),
            "thumb_pos":         (0, 0),
            "middle_pos":        (0, 0),
            "pinch_pos":         (0, 0),
            "pinch_pos_raw":     (0, 0),
            "is_pinching":       False,
            "click_just_fired":  False,
            "is_index_isolated": False,
            # Hand signs
            "hand_sign":         SIGN_UNKNOWN,
            "fingers_extended":  (False, False, False, False, False),
            "is_fist":           False,
            "is_open_palm":      False,
            "grip_openness":     0.0,
            "smoothing_enabled": self.smoothing_enabled,
            "z_delta":           0.0,
            "xy_drift":          0.0,
            "tof_active":        False,
            "tof_z_m":           0.0,
            "tof_z_raw":         0.0,
            "depth_source":      "RGB MediaPipe Estimate",
            "is_3_finger_pinching": False,
            # Full zeroed stat dict rather than {} — the HUD reads
            # jitter["raw_jitter_std"] and used to see an empty mapping here.
            "jitter":            JitterAnalyzer.empty_stats(),
            # Stabilizer telemetry defaults
            "stabilizer_state":     self.tof_stabilizer.state,
            "stabilizer_progress":  self.tof_stabilizer.progress,
            "stabilizer_noise_amp": self.tof_stabilizer.z_noise_amplitude,
            "stabilizer_gate":      self.tof_stabilizer.noise_gate,
        }

    def _empty_payload(self, tx: float, ty: float, frame: np.ndarray) -> dict:
        return {
            "target_x": tx,
            "target_y": ty,
            "frame":    frame,
            **self._empty_gesture(),
        }
