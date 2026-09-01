"""
pipeline.py — VisionPipeline: background webcam thread for the Visual AI Game Engine.

Detects BOTH:
  • Face coordinates  (BlazeFace on an accelerator, MediaPipe FaceDetection
                       otherwise, or the frame centre when neither found one)
  • Hand gestures     (MediaPipe Hands)
      – index-fingertip position       → index_pos
      – 3-finger pinch (thumb+index+middle within a centroid radius)
                                       → is_pinching  (not debounced)
      – click_just_fired               → rising edge of the 3-finger pinch;
                                         with enable_z_click also a Z push
      – index-finger isolation         → is_index_isolated

Queue payload (dict)
--------------------
{
  # Face
  "target_x"  : float,   # face centre X (px), or frame centre if no face
  "target_y"  : float,   # face centre Y (px), or frame centre if no face
  "face_visible" : bool,
  "face_box"  : (int,)*4,# x, y, w, h in px; (0,0,0,0) when no face. Box width is
                         #   the RGB path's only distance proxy for the head.
  "face_count": int,     # detections found; only the first drives target_x/y
  "face_hands_rejected"       : int,   # hand candidates dropped as being the
                                       #   player's own face, this frame
  "face_hands_rejected_total" : int,   # ...and over the session
  "frame"     : ndarray, # BGR camera frame (already flipped & resized)

  # Hand — these flat keys describe the PRIMARY hand (lowest occupied slot)
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
  "finger_extension"  : (float,) * 5, # what those booleans were thresholded
                                      #   from: tip reach past the PIP joint in
                                      #   palm spans, signed. The thumb's is
                                      #   abduction against a 0.10 threshold,
                                      #   the other four are against 0.0.
  "is_fist"           : bool,
  "is_open_palm"      : bool,
  "grip_openness"     : float,        # 0.0 curled fist … 1.0 fingers straight

  # Motion (see _extract_gesture). px/s, differenced frame to frame.
  "index_velocity"    : (float, float),  # from the SMOOTHED fingertip — use this
  "index_velocity_raw": (float, float),  # from the raw landmark; the difference
                                         #   against the above is the filter's lag
  "index_speed"       : float,        # |index_velocity|
  "index_accel"       : float,        # d|v|/dt, EMA'd (raw double-difference is
                                      #   too noisy to hand to a consumer)

  # Multi-hand
  "hands"      : tuple[dict, ...],    # one gesture dict per tracked hand, slot
                                      #   order, each with the keys above plus
                                      #   "slot" / "handedness" / "handedness_score"
  "hand_count" : int,
  "hand_left"  : dict | None,         # first hand MediaPipe labelled "Left"
  "hand_right" : dict | None,
  "slot"       : int,                 # primary hand's slot
  "handedness" : str,                 # "Left" | "Right" | "unknown"
  "handedness_score" : float,

  # Depth occupancy grid — ONLY when pipeline.emit_depth_grid is True
  "depth_grid" : ndarray | None,      # (rows, cols) float32 metres, 0 = no return

  # Person segmentation mask — ONLY when pipeline.emit_person_mask is True
  "person_mask" : ndarray | None,     # (H, W) uint8, 255 = person, 0 = background

  # Full landmark set, per hand — populated ONLY when pipeline.emit_landmarks
  # is True; an empty tuple otherwise, so indexing it is always safe.
  "landmarks"  : tuple[(float, float, float), ...],  # 21 points, (x px, y px,
                                      #   z px). Z is MediaPipe's relative
                                      #   depth in the same units as x —
                                      #   0 at the wrist, negative toward the
                                      #   camera. It is not metres and is not
                                      #   comparable between hands.

  "smoothing_enabled" : bool,         # One-Euro landmark smoothing currently on?
  "z_delta"           : float,        # raw Z push magnitude (debug)
  "xy_drift"          : float,        # lateral drift during a Z push (px)

  # 3-finger pinch
  "is_3_finger_pinching" : bool,

  # Capture conditions (see low_light.py)
  "scene_luma"           : float,     # mean luma 0-255 before any boost
  "low_light_gain"       : float,     # gain applied; 1.0 = frame untouched

  # Depth (see depth_stabilizer.py)
  "depth_active"         : bool,      # a real depth sensor is supplying this
  "depth_m"              : float,     # stabilized depth (m)
  "depth_m_raw"          : float,     # depth before stabilization (m)
  "depth_source"         : str,
  # Deprecated aliases of the four above, kept so existing games keep running.
  # New code should read the depth_* names: "ToF" named a sensor this engine
  # did not have, and the keys carried an RGB estimate under that name.
  "tof_active"           : bool,
  "tof_z_m"              : float,
  "tof_z_raw"            : float,
  "depth_device"         : str,      # backend name, "" when there is none
  "depth_fps"            : float,    # sensor frames/s, 0.0 without a sensor
  "hand_device"          : str,      # where hand tracking runs: "NPU", "GPU",
                                     # or "CPU (MediaPipe)" — see accel.py. Put
                                     # it on the HUD: a requested accelerator
                                     # that fell back to CPU is otherwise
                                     # indistinguishable in play from one that
                                     # is working, just slower.
  "face_device"          : str,      # the same, for face detection. "" when
                                     # the game asked for detect_face=False,
                                     # so a HUD can leave the line out rather
                                     # than name a device for a graph that is
                                     # not running.
  "stabilizer_state"     : str,       # "inactive" | "sampling" | "active"
  "stabilizer_progress"  : float,
  "stabilizer_noise_amp" : float,     # measured vibration std-dev (m)
  "stabilizer_gate"      : float,     # current noise-gate half-width (m)

  # Jitter metrics (see jitter_analyzer.py) — always a full stat dict
  "jitter"               : dict,
}

Every key above is present on EVERY payload, including frames with no hand and
simulated-camera frames, so consumers can index directly. The exceptions are
"depth_grid" and "person_mask", which are opt-in precisely because they are the
only keys whose size is not O(1) and every game drains a maxsize=1 queue.

Hand slots
----------
MediaPipe does not guarantee a stable order for `multi_hand_landmarks` between
frames. Hands are therefore assigned to persistent slots by nearest-neighbour
against the previous frame's wrist position, and each slot owns its own One-Euro
filters, sign debounce and jitter analyser. Keying that state off the raw list
index instead makes two hands swap filter histories the moment the order flips.

`handedness` is MediaPipe's own label. The frame is mirrored before detection
(selfie view), which is the orientation MediaPipe's handedness model expects, so
the label should match the player's real hand — but treat it as a hint and let
the player rebind rather than hard-coding a hand to a role.
"""

import math
import queue
import threading
import time
import warnings
from collections import namedtuple

import cv2
import numpy as np

from visual_ai import accel, devicelog, openvino_hands, openvino_zoo
from visual_ai.depth_source import DepthStream, open_depth_source
from visual_ai.depth_stabilizer import DepthStabilizer
from visual_ai.gesture_math import get_landmark_velocity
from visual_ai.gesture_mlp import GestureMLP, landmarks_to_features
from visual_ai.jitter_analyzer import JitterAnalyzer
from visual_ai.low_light import LowLightBoost
from visual_ai.noise_filter import (
    OneEuroFilter,
    PipelineNoiseFilter,
    ema_alpha_to_cutoff,
)

# ── Optional MediaPipe imports ────────────────────────────────────────────────
HAS_MEDIAPIPE = False
mp_face_detection_module = None
mp_hands_module = None

try:
    import mediapipe as mp  # noqa: F401 - availability probe; used via the names below

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

# ── Deprecation ───────────────────────────────────────────────────────────────

def _warn_deprecated(old: str, new: str) -> None:
    """
    Point a caller at the depth_* name for an old ToF-era one.

    Left to the warnings module to deduplicate: its default filter shows a
    DeprecationWarning once per call site, and hides it entirely outside
    ``__main__``, so a game that assigns ``tof_simulated`` in its own main
    module gets exactly one line and the SDK's own imports stay silent.
    """
    warnings.warn(
        f"VisionPipeline.{old} is deprecated; use VisionPipeline.{new} instead. "
        f'The "ToF" names describe a time-of-flight sensor this engine never had.',
        DeprecationWarning,
        stacklevel=3,
    )


# ── Gesture detection constants ───────────────────────────────────────────────
# Z-push click
_Z_HISTORY_LEN       = 10      # rolling window size (frames)
_Z_MIN_HISTORY       = 4       # frames needed before the window can fire
_Z_CLICK_THRESHOLD   = 0.012   # minimum MediaPipe-Z delta to count as a push
_Z_CLICK_THRESHOLD_M = 0.030   # minimum ToF depth delta to count as a push (m)

# A depth frame older than this is dropped rather than reused. Sensors stall,
# and a held frame is indistinguishable downstream from a live one -- which is
# how a game ends up acting on where your hand was a second ago.
DEPTH_STALE_S = 0.5
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

# ── Face-as-hand rejection ────────────────────────────────────────────────────
# The palm network fires on faces. Downstream that is not a cosmetic error: a
# false hand takes a slot, and every backend stops looking for more hands once
# it holds `max_hands` of them — so a face eats the slot the player's second
# hand needed and the game simply never sees it. depthpong is where it was
# found; anything with `max_hands=2` has it.
#
# The accelerated backend gates it at the source (see `_FACE_GATE` in
# openvino_hands.py) because only there can the rect be dropped and the
# detector re-run. This is the second net, and it is backend-agnostic: it
# cannot free a slot inside MediaPipe's own tracker, but it does keep the face
# from driving the game and it makes `hand_count` mean what it says.
#
# Two signals have to agree, because the case that must not break is a real
# hand held *over* the face:
#
#   containment — a hand in front of a face is nearer the camera than the face
#     is, so it images *larger* than the head and its landmarks spill out of
#     the box. Landmarks that all fit inside a face-sized box are not a hand
#     in front of that face.
#   handedness  — both backends report which hand it is with a confidence, and
#     a face is a coin flip. A real hand is rarely below 0.9.
#
#: Fraction of the 21 landmarks that must fall inside the dilated face box.
_FACE_HAND_CONTAINED = 0.9
#: The face box MediaPipe returns is tight to the eyes and mouth; the hand the
#: palm network hallucinates out of a face spans the whole head.
_FACE_BOX_DILATE = 1.15
#: Above this handedness confidence the candidate is taken at its word.
_FACE_HAND_HANDEDNESS = 0.9

#: One face, normalised to the frame — what both face backends are reduced to.
#: The field names are MediaPipe's `relative_bounding_box`, because that is the
#: shape the reading code was written against.
_FaceBox = namedtuple("_FaceBox", "xmin ymin width height")

#: How many faces the accelerated backend is asked for. MediaPipe's graph has
#: no cap and `face_count` rides on the payload, so this is not free headroom
#: to cut to one — a game counting people in frame would start seeing fewer.
_MAX_FACES = 4


# ── Duplicate-hand rejection ──────────────────────────────────────────────────
# The palm network can return two boxes for one hand, and nothing downstream
# notices: `_assign_slots` matches on nearest wrist, so the copy takes the
# second slot and the game gets two paddles out of one hand while the player's
# real other hand is never looked for. Same slot-starvation as a face-as-hand,
# different cause. depthpong is where it shows, for the same reason.
#
# The test is proximity: two hands whose palm centres sit within a fraction of
# a palm span of each other are one hand, and the later one is dropped.
#
# It used to be three tests -- centres, then apparent size, then per-landmark
# pose -- so that one real hand held over the other survived, on the argument
# that the nearer hand images larger while two boxes on one hand agree to a
# few percent. That let copies through. The two crops a copy comes from
# differ in scale and position (the detector's palm rect against the
# tracker's landmark rect, or a stale find on a moving hand), so the landmark
# network returns spans that differ by more than the 12% the size test
# allowed, and the game drew two skeletons on one hand. A copy is what the
# player sees every session; two hands overlapping this closely is rare, and
# they come back as two the moment they part. So the rule is the simple one.
# `play <title> --hands 1` removes the question entirely: with one slot the
# backend stops looking once a hand is held, and there is nothing to copy.
#
# Written in palm spans (wrist to middle-finger MCP) so it holds at arm's
# length and up against the lens alike. The accelerated backend applies the
# same test one stage earlier, on its tracking rects (`_DUP_SPAN` in
# openvino_hands.py), because only there can the copy's slot be freed.
_DUP_HAND_SPAN = 0.6


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


# ── Finger extension ──────────────────────────────────────────────────────────
# A finger counts as extended when its tip reaches further from the wrist than
# its own PIP joint. Tested as a bare `>`, that decision has no width: a finger
# resting near the boundary flips on landmark noise alone, and `point` needs
# *three* fingers to stay down at once, so one flickering finger is enough to
# stop the sign ever surviving `_SIGN_DEBOUNCE` frames. That is the "pointing is
# very hard to get right" report, and it got worse when hand tracking moved onto
# the NPU by default — the accelerated landmarks are within 4.4 px median of
# MediaPipe's but that is 4.4 px of extra wobble across a boundary with no
# width. So the boundary gets one: past it by a palm-scaled margin to change,
# and the previous answer inside it.
#
# The margins are fractions of the palm span (wrist to pinky MCP), so they mean
# the same thing at any hand size or camera distance. An extended finger clears
# its PIP by roughly half a palm span and a curled one falls about a third of
# one short of it, so a band of ±0.08 is well inside the real signal while
# being several times the landmark noise.
_FINGER_EXT_MARGIN = 0.08
#: The thumb is measured by abduction against a 0.10 threshold, not against
#: zero, so it gets its own band around that.
_THUMB_EXT_MARGIN = 0.05


def _extended(delta: float, threshold: float, margin: float, previous: bool) -> bool:
    """Threshold with hysteresis: inside the band, keep the previous answer."""
    if delta > threshold + margin:
        return True
    if delta < threshold - margin:
        return False
    return previous


def _dist2d(p1, p2) -> float:
    return math.dist(p1[:2], p2[:2])


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

        self.pinch_active: bool  = False

        # Z-push click — entries are (z_value, (x, y)) so each sample keeps the
        # position it was taken at, for the lateral-drift guard.
        self.z_history: list[tuple[float, tuple[int, int]]] = []
        self.z_cooldown: int        = 0
        self.z_source: str          = "mp"   # unit of z_history: "mp" | "tof"

        # Frames since a hand was last seen (drives the reset below)
        self.lost_frames: int = 0

        # 3-Finger Pinch state
        self.was_3_pinching: bool = False        # Edge trigger for 3-finger pinch

        # Hand sign debounce
        self.sign: str = SIGN_UNKNOWN            # currently reported sign
        self.sign_candidate: str = SIGN_UNKNOWN  # sign awaiting confirmation
        self.sign_consec: int = 0                # frames the candidate has held

        # Last frame's five extension booleans — what the hysteresis band in
        # `_classify_fingers` holds on to while a finger sits on its boundary.
        self.fingers_prev: tuple[bool, bool, bool, bool, bool] = (False,) * 5

        # Motion. Velocity is differenced from the *smoothed* fingertip so it is
        # usable directly; `raw_*` differences the unfiltered landmark so a
        # consumer can measure what the One-Euro filter actually costs in phase
        # lag rather than taking the filter's word for it.
        self.prev_time: float | None = None
        self.prev_xy: tuple[float, float] | None = None
        self.prev_raw_xy: tuple[float, float] | None = None
        self.vel: tuple[float, float] = (0.0, 0.0)
        self.raw_vel: tuple[float, float] = (0.0, 0.0)
        self.speed: float = 0.0
        self.accel: float = 0.0

    def reset(self):
        self.smoothed = False
        self.smooth_ix = self.smooth_iy = 0.0
        self.smooth_px = self.smooth_py = 0.0
        if self.index_filter is not None:
            self.index_filter.reset()
        if self.centroid_filter is not None:
            self.centroid_filter.reset()
        self.pinch_active  = False
        self.z_history.clear()
        self.z_cooldown = 0
        self.z_source = "mp"
        self.lost_frames = 0
        self.was_3_pinching = False
        self.sign = SIGN_UNKNOWN
        self.sign_candidate = SIGN_UNKNOWN
        self.sign_consec = 0
        self.fingers_prev = (False,) * 5
        self.prev_time = None
        self.prev_xy = None
        self.prev_raw_xy = None
        self.vel = (0.0, 0.0)
        self.raw_vel = (0.0, 0.0)
        self.speed = 0.0
        self.accel = 0.0


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
        Resting smoothness (0 < α ≤ 1), mapped to the One-Euro filter's
        min_cutoff via `ema_alpha_to_cutoff` — it is no longer a literal EMA
        factor. Lower = steadier at rest but laggier; default 0.20.
    max_hands : int
        How many hands MediaPipe tracks, and how many gesture slots the payload
        carries. 1 is measurably cheaper per frame; 2 is the default because it
        is what the detector was already configured with.
    detection_stride : int
        Run MediaPipe Hands + FaceDetection every Nth frame instead of every
        frame; the frames in between hold the last detection's landmarks
        (One-Euro smoothing still runs every frame, so held positions don't
        look frozen at rest). 1 (default) detects every frame — unchanged
        behaviour. benchmarks/bench_resolution.py measured stride=2 against a
        recorded clip at ~50% less MediaPipe time for ~22px of fingertip rmse
        and negligible added lag; stride=3 saves more but starts costing real
        accuracy. Tune per game, not here.
    model_complexity : int
        MediaPipe Hands model tier: 1 (default) is the full landmark model,
        0 is the lite model — noticeably cheaper per frame at a small cost in
        fingertip accuracy. The speed knob to reach for on low-end machines.
    depth_source : str, optional
        Where metric depth comes from. None (default) keeps the historic
        behaviour: no sensor, and `tof_z_m` is a MediaPipe estimate unless
        `depth_simulated` is set. "auto" opens the first depth backend that
        responds; "synthetic" and "replay:PATH" need no hardware. See
        depth_source.py for the full spec list.
    low_light_boost : bool
        Lift underexposed frames before detection (default True). The boost is
        adaptive and one-sided: a well-lit frame solves to a gain of 1.0 and is
        passed through untouched, so leaving this on costs one subsampled mean
        per frame. See low_light.py.
    detect_face : bool
        Run the FaceDetection graph. True (default) is the long-standing
        behaviour. False skips a whole second inference — 3.1 ms/frame, 15% of
        per-frame cost at 1280x720 — for games that never read a face key. The
        payload schema does not change either way: with it off, ``target_x`` /
        ``target_y`` hold frame centre, ``face_visible`` is False, ``face_box``
        is zeros and ``face_count`` is 0, exactly as when no face is in view. A
        consumer that reads those keys will not raise, it will simply never see
        a face — so turn it off only for games you know do not use them.
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
        gesture_mlp: GestureMLP | None = None,
        max_hands: int = 2,
        detection_stride: int = 1,
        model_complexity: int = 1,
        detect_face: bool = True,
        low_light_boost: bool = True,
        depth_source: str | None = None,
        hand_device: str | None = None,
        face_device: str | None = None,
    ):
        super().__init__(daemon=True)
        self.result_queue  = result_queue
        self.width         = width
        self.height        = height
        self.camera_index  = camera_index
        self.smooth_alpha  = smooth_alpha
        self.movement_magnification = movement_magnification
        self.noise_filter  = PipelineNoiseFilter(noise_duration=noise_duration)
        self.tof_stabilizer = DepthStabilizer()

        # Underexposure gate. MediaPipe stops returning a hand well before a
        # room looks dark to a person, and a dropout is indistinguishable
        # downstream from a hand that left the frame -- so the fix belongs
        # here, ahead of detection, rather than in each game.
        # Depth producer. Opened in start() rather than here, so constructing
        # a pipeline never grabs a capture device -- the same reason the camera
        # is not opened until the thread runs.
        self.depth_source_spec = depth_source
        self.depth_stream: DepthStream | None = None
        self.depth_fps: float = 0.0

        self.low_light_boost = LowLightBoost() if low_light_boost else None
        self.scene_luma: float = 0.0
        self.low_light_gain: float = 1.0
        self.running       = False
        self._stop_requested = False   # latched by stop(); run() never re-arms past it

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

        # Hand-sign classification is `classify_hand_sign()`'s fixed geometric
        # lookup table by default. Passing a trained `GestureMLP` here swaps
        # `hand_sign` over to it instead — opt-in, so payload behavior for
        # existing consumers is unchanged unless a caller supplies one.
        self.gesture_mlp: GestureMLP | None = gesture_mlp

        # One-Euro landmark smoothing can be bypassed at runtime so raw
        # MediaPipe positions reach the consumer. This is separate from the ToF
        # stabilizer below — that one gates depth, this one gates X/Y.
        self.smoothing_enabled: bool = True

        # ── ToF (Time-of-Flight) Depth Sensor State ───────────────────────────
        # Canonical names. `depth_active` means a real sensor is feeding
        # depth_map; `depth_simulated` means the numbers are synthesised and
        # must never be presented as measurements.
        self.depth_active: bool       = False
        self.depth_simulated: bool    = False
        self.depth_device_name: str   = "None"
        self.depth_map: np.ndarray | None = None

        # ── Depth occupancy grid (opt-in) ─────────────────────────────────────
        # A whole-scene depth read, downsampled, for consumers that want the
        # silhouette rather than a single fingertip sample. Off by default: it
        # is the only payload key whose size is not O(1), and every existing
        # game drains a maxsize=1 queue that would then carry it every frame.
        self.emit_depth_grid: bool = False
        self.depth_grid_size: tuple[int, int] = (32, 24)   # (cols, rows)

        # ── Person segmentation mask (opt-in) ─────────────────────────────────
        # Same reasoning as emit_depth_grid: off by default, since the mask is
        # O(frame) and every existing game drains a maxsize=1 queue that would
        # otherwise carry it every frame whether a consumer wants it or not.
        # The model is built lazily on first use — see _get_person_mask — so
        # turning this on is the only thing that pays its download/init cost.
        self.emit_person_mask: bool = False
        self._person_segmenter = None

        # ── Full landmark set (opt-in) ────────────────────────────────────────
        # The 21 points per hand, in pixels, with MediaPipe's relative Z
        # carried in the same units. Every existing consumer wants a fingertip
        # and a sign, not a skeleton, so this is off by default — but a game
        # that draws the hand into a 3D scene needs the whole hand and needs
        # its depth, or the fingers cannot pass behind anything.
        self.emit_landmarks: bool = False

        self.disable_camera: bool   = False

        # Scratch buffer reused every frame by the capture loop, so the
        # per-frame full-frame conversion stops paying for an allocation each.
        # Only safe because this one never leaves the capture thread — see the
        # resize below for the buffer that could not stay reused.
        self._rgb_buf: np.ndarray | None = None

        # ── Detection frame-skip (opt-in) ─────────────────────────────────────
        # Held frames reuse the last MediaPipe detection rather than an
        # interpolated one: a live pipeline has no future detection to
        # interpolate towards, only past ones.
        self.detection_stride: int = max(1, int(detection_stride))
        self._detection_frame_count: int = 0
        self._cached_face_detections = None
        self._cached_landmark_sets: list = []
        self._cached_handedness: list = []

        # ── Face-as-hand rejection ────────────────────────────────────────────
        # Candidates dropped as the face on the last detection frame, and over
        # the session. Both travel on the payload: a player whose second hand
        # will not track has no other way to find out that their face is what
        # took the slot.
        self.face_hand_filter = True
        self._face_hands_rejected = 0
        self.face_hands_rejected_total = 0
        # Not a payload key: adding one is a queue-schema change. Read it off
        # the pipeline object if a game wants to show it.
        self.duplicate_hands_rejected_total = 0

        # ── Face detection ────────────────────────────────────────────────────
        # A whole second inference graph, measured at 3.1 ms/frame (15% of
        # _process_frame) at 1280x720. It is opt-*out* rather than opt-in
        # because the face keys have been on every payload since the pipeline
        # was written and duckhunt, avatarcatch and labkit read them — flipping
        # the default would break those silently, one KeyError-free frame of
        # wrong behaviour at a time. Hand-only games pass detect_face=False and
        # stop paying for a detection nothing consumes.
        #
        # BlazeFace through OpenVINO first, the way hands do below. It is the
        # same network MediaPipe's FaceDetection runs, so this is a change of
        # device and not of model, and on a frame whose hands are already on
        # the NPU it costs 0.50 ms where MediaPipe's graph costs 9.59 (the
        # measurements are in accel.py).
        #
        # The boxes are not MediaPipe's to the pixel: on the 720p fixture the
        # OpenVINO decode sits ~4 px left of MediaPipe's and ~6 px above it,
        # and is 1% smaller. That is the letterbox and NMS convention rather
        # than the device — NPU against OpenVINO's own CPU plugin is IoU
        # 0.994 — but it does move `face_box` and `target_x/y` by a few pixels
        # for every game that reads them.
        self.detect_face = bool(detect_face)
        self._face_net = None
        self._mp_face = None
        resolved_face_device = (accel.resolve("face", explicit=face_device)
                                if self.detect_face else None)
        if resolved_face_device:
            try:
                net = openvino_zoo.FaceDetector(resolved_face_device,
                                                max_faces=_MAX_FACES)
            except Exception as e:
                print(f"[VisionPipeline] FaceDetector on {resolved_face_device} "
                      f"unavailable: {e}")
                net = None
            # The zoo answers a device that refuses the model with its own CPU
            # plugin. That is not the fallback wanted here — it is four cores
            # of inference against MediaPipe's one — so keep the accelerated
            # path only when the device asked for is the device that answered.
            if net is not None and net.device != resolved_face_device:
                print(f"[VisionPipeline] {resolved_face_device} refused BlazeFace "
                      f"({net.device_name}); using MediaPipe")
                net.close()
                net = None
            self._face_net = net
            if net is None:
                resolved_face_device = None
        if (self.detect_face and self._face_net is None
                and HAS_MEDIAPIPE and mp_face_detection_module is not None):
            try:
                self._mp_face = mp_face_detection_module.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
            except Exception as e:
                print(f"[VisionPipeline] FaceDetection init warning: {e}")
        # Empty rather than "CPU (MediaPipe)" for a game that turned the graph
        # off: nothing is detecting faces, and a HUD should say nothing at all.
        self.face_device_name = (
            accel.describe("face", resolved_face_device, "CPU (MediaPipe)")
            if self.detect_face else "")

        # ── MediaPipe: Hands ──────────────────────────────────────────────────
        self._mp_hands = None
        self.max_hands = max(1, int(max_hands))
        # 1 is MediaPipe's full landmark model; 0 trades a little fingertip
        # accuracy for a substantially cheaper per-frame inference — worth
        # exposing so a low-end machine can hold its frame rate.
        self.model_complexity = 0 if int(model_complexity) <= 0 else 1

        # The same two networks can run on the NPU or the iGPU through
        # OpenVINO instead, at a third of the latency and a third of the CPU
        # (see accel.py for the measurements). Tried by default now
        # (`accel.DEFAULT_PRESET`), and overridable per-consumer with
        # `hand_device=` or per-run with `play <title> --accel ...`. It is tried
        # *first* so a machine that has it never pays to build MediaPipe's CPU
        # graph as well. `build` returns None rather than raising when the
        # device is absent, which is what keeps this a fallback and not a hard
        # dependency -- and what makes defaulting it on safe.
        resolved_hand_device = accel.resolve("hand", explicit=hand_device)
        self._mp_hands = openvino_hands.build(
            device=hand_device,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.45,
        )
        if self._mp_hands is None:
            resolved_hand_device = None
        # Which gate a session ran under is not recoverable from a file mtime —
        # see the 2026-08-25 entry in MISTAKES-FROM-CLAUDE.md.  0.45 for
        # OpenVINO (NPU and iGPU both need the looser gate; see the note on
        # OpenVINOHands.__init__), 0.65 for MediaPipe CPU.  dGPU is untested.
        self.hand_gate = 0.45 if self._mp_hands is not None else 0.65
        self.hand_device_name = accel.describe(
            "hand", resolved_hand_device, "CPU (MediaPipe)")

        if self._mp_hands is None and mp_hands_module is not None:
            try:
                self._mp_hands = mp_hands_module.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_hands,
                    model_complexity=self.model_complexity,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=self.hand_gate,
                )
            except Exception as e:
                print(f"[VisionPipeline] Hands init warning: {e}")

        # Real play is the only place the accelerator numbers can be confirmed
        # (see devicelog.py). Off unless switched on, and built here rather
        # than in `run()` so the header names the devices even for a session
        # that never gets a camera.
        self.device_log = devicelog.DeviceLog()
        self.device_log.session(
            hand_device=self.hand_device_name,
            hand_backend=type(self._mp_hands).__name__ if self._mp_hands else "none",
            face_device=self.face_device_name,
            accel_preset=accel.preset(),
            devices=accel.available_devices(),
            model_complexity=self.model_complexity,
            max_hands=self.max_hands,
            width=self.width, height=self.height,
            capture_fps=self.capture_fps,
        )

        # ── Per-hand gesture state ────────────────────────────────────────────
        # One slot per tracked hand. Slot 0 stays reachable as `self._gs` and
        # `self.jitter_analyzer` because that is what every existing consumer
        # (and test) reads; the extra slots are new. Hands are assigned to slots
        # by nearest-neighbour against the previous frame rather than by
        # MediaPipe's list order, which is not stable frame to frame — without
        # that, two hands crossing would swap their One-Euro filter histories
        # and both would snap.
        self._gs_slots = [
            _GestureState(make_filter=self._make_position_filter)
            for _ in range(self.max_hands)
        ]
        self._jitter_slots = [JitterAnalyzer(window_size=30) for _ in range(self.max_hands)]
        self._slot_anchor: list[tuple[float, float] | None] = [None] * self.max_hands

        self._gs = self._gs_slots[0]
        self.jitter_analyzer = self._jitter_slots[0]

        # Size-dependent parts of the calibration overlay, built on first use.
        # See _calibration_chrome().
        self._calib_chrome: dict | None = None

        self.last_error: str | None = None
        self.camera_available: bool = False

    def get_status(self) -> dict:
        """Get diagnostic status of the pipeline thread and hardware sources."""
        return {
            "running": self.running,
            "camera_available": self.camera_available,
            "disable_camera": getattr(self, "disable_camera", False),
            "has_mediapipe": HAS_MEDIAPIPE,
            "face_tracking": self._face_net is not None or self._mp_face is not None,
            "face_device": self.face_device_name,
            "hand_tracking": self._mp_hands is not None,
            "hand_device": self.hand_device_name,
            "device_log": self.device_log.status(),
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
        """
        Push current tuning onto the live filters without dropping their state.

        Every slot, not just `self._gs`. Retuning slot 0 alone left the second
        hand running the settings it was built with, so `set_smooth_alpha` moved
        one hand and not the other.
        """
        for gs in self._gs_slots:
            for f in (gs.index_filter, gs.centroid_filter):
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
        """Update the movement magnification for tracking and pinch scaling."""
        self.movement_magnification = max(0.5, float(mag))

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
        if not (self.depth_active or self.depth_simulated):
            print(
                "[VisionPipeline] begin_stabilization() called with no ToF source — "
                "attach a sensor, or set pipeline.depth_simulated = True with a real "
                "camera and a tracked hand (samples are only fed from gesture "
                "extraction). Calibration will collect no samples and abort."
            )
        self.tof_stabilizer.begin(duration)

    def cancel_stabilization(self) -> None:
        """Abort an in-progress calibration (the overlay's 'Press X to cancel')."""
        self.tof_stabilizer.cancel()

    def disable_stabilization(self) -> None:
        """Disable ToF stabilization and clear the calibrated offset."""
        self.tof_stabilizer.disable()

    # ── Thread entry ──────────────────────────────────────────────────────────
    def start(self):
        """Arm the loop flag *before* the thread exists. `run()` used to set
        `running = True` as its first statement, so a `start(); stop()` pair
        racing thread scheduling could have `run()` re-arm the flag after
        `stop()` cleared it — a daemon loop holding the camera forever."""
        self.running = True
        super().start()

    def run(self):
        if not self._stop_requested:
            self.running = True   # direct run() callers never went through start()
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_index)
            self.camera_available = cap.isOpened()
            if self.camera_available:
                # Ask the driver for the target format up front. Every set()
                # is a request the driver may ignore — the resize below stays
                # as the fallback — but when it complies the per-frame
                # cv2.resize disappears and, more importantly for input feel,
                # BUFFERSIZE=1 stops the backend queueing frames we would
                # only ever read late.
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.capture_fps)
                except Exception:
                    pass
        except Exception as e:
            self.last_error = f"Camera initialization failed: {e}"
            self.camera_available = False

        if not self.camera_available:
            msg = f"[VisionPipeline] Camera index {self.camera_index} not available."
            if self.last_error:
                msg += f" Details: {self.last_error}"
            print(f"{msg} Running simulated vision target.")

        # ── Depth sensor ──────────────────────────────────────────────────
        # Runs on its own thread: depth and colour have independent frame
        # rates, and a blocking sensor read inside this loop would pace the
        # whole RGB path off the slower of the two.
        if self.depth_source_spec:
            source = open_depth_source(self.depth_source_spec, quiet=False)
            if source is None:
                print(f"[VisionPipeline] No depth source for "
                      f"{self.depth_source_spec!r} — staying on the RGB path.")
            else:
                self.depth_stream = DepthStream(source)
                self.depth_stream.start()
                # `depth_active` means measured depth from a real device. A
                # synthetic or replayed source is depth, but it is not a
                # sensor, and labelling it as one is the misreporting this
                # whole path was built to end.
                self.depth_active = not source.synthetic
                self.depth_simulated = self.depth_simulated or source.synthetic
                self.depth_device_name = source.name
                print(f"[VisionPipeline] Depth source: {source.name}"
                      f"{' (synthetic)' if source.synthetic else ''}")

        sim_angle = 0.0
        sim_template: np.ndarray | None = None
        # Set by a real captured frame and cleared once logged, so a simulated
        # frame never lands in the device log as if it had been tracked.
        frame_ms: float | None = None

        try:
            while self.running:
                # Newest depth frame, if a sensor is attached. Pulled before
                # the camera branch so depth reaches consumers in simulated
                # mode too -- a depth rig with no webcam is a valid setup, and
                # it is the one a headless test uses. A frame older than
                # DEPTH_STALE_S is dropped rather than reused: a held map is
                # indistinguishable downstream from a live one.
                # Snapshot the reference: stop() nulls self.depth_stream from
                # the game thread, and reading the attribute once means a
                # mid-iteration shutdown can't turn the None-check into an
                # AttributeError on the dereferences below.
                depth_stream = self.depth_stream
                if depth_stream is not None:
                    self.depth_map = (
                        depth_stream.latest()
                        if depth_stream.age_s() < DEPTH_STALE_S
                        else None)
                    self.depth_fps = depth_stream.source.fps

                if (self.camera_available and cap is not None
                        and not getattr(self, 'disable_camera', False)):
                    try:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            time.sleep(0.01)
                            continue

                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            # Fresh allocation, deliberately. This is the
                            # fallback path for a driver that ignored the size
                            # request, so it runs every frame when it runs at
                            # all, and a reused `dst=` buffer did save 0.2 ms of
                            # its 0.36 — but this array becomes payload["frame"]
                            # unchanged (`low_light_boost.apply` returns its
                            # argument whenever the scene is bright enough to
                            # need no gain), and the payload contract lets the
                            # game thread read and draw on it. Reusing the
                            # buffer meant the capture thread resized into a
                            # frame a consumer was still holding, and every
                            # queued payload aliased the same one.
                            frame = cv2.resize(frame, (self.width, self.height))
                        # Mirror for natural interaction. In place: a horizontal
                        # flip swaps column pairs, so src and dst may be the same
                        # buffer, and the capture buffer is rewritten by the next
                        # read either way.
                        cv2.flip(frame, 1, dst=frame)

                        # Boost before detection, and hand the boosted frame on
                        # to consumers as well: a player in a dim room needs to
                        # see what the tracker is seeing, and a preview that
                        # stays dark while detection improves is a preview that
                        # lies about why tracking works.
                        if self.low_light_boost is not None:
                            frame = self.low_light_boost.apply(frame)
                            self.scene_luma = self.low_light_boost.luma
                            self.low_light_gain = self.low_light_boost.gain

                        frame_started = time.perf_counter()
                        payload = self._process_frame(frame)
                        frame_ms = (time.perf_counter() - frame_started) * 1e3
                    except Exception as frame_err:
                        self.last_error = f"Frame read/process error: {frame_err}"
                        time.sleep(0.01)
                        continue
                else:
                    if self.camera_available and cap is not None:
                        cap.grab() # Keep buffer drained while disabled
                    # ── Simulated mode (no camera) ────────────────────────────────
                    # 5 rad/s regardless of loop rate — the old fixed 0.05/tick
                    # tied the orbit speed to how fast the loop happened to spin.
                    sim_angle += 5.0 / self.capture_fps
                    target_x = self.width  / 2.0 + math.cos(sim_angle) * 200.0
                    target_y = self.height / 2.0 + math.sin(sim_angle) * 150.0

                    # The banner never changes; paint it once and hand each frame
                    # out as a copy (consumers draw HUDs onto payload["frame"]).
                    if sim_template is None:
                        sim_template = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        cv2.putText(
                            sim_template,
                            "Simulated Vision Mode (No Camera)",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                    payload = self._empty_payload(target_x, target_y, sim_template.copy())

                # Advance the stabilization clock from the capture loop, not from
                # gesture extraction. Samples only arrive while a hand is in view,
                # so a calibration started with no hand present used to hang in
                # "sampling" forever with the warning overlay stuck on screen.
                self.tof_stabilizer.tick()

                # Apply Noise Filter to suppress transient clicks/pinches during warmup window
                payload = self.noise_filter.process_payload(payload)

                # ── ToF Stabilizer: draw warning overlay during calibration ──────
                if (
                    self.tof_stabilizer.state == DepthStabilizer.STATE_SAMPLING
                    and payload is not None
                    and "frame" in payload
                    and payload["frame"] is not None
                ):
                    payload["frame"] = self._draw_stabilizer_warning(
                        payload["frame"], self.tof_stabilizer.progress
                    )

                # One line per window, not per frame — see devicelog.py.
                if self.device_log.enabled and frame_ms is not None:
                    self.device_log.frame(
                        latency_ms=frame_ms,
                        hands=len(payload.get("hands") or ()) if payload else 0,
                        counters=getattr(self._mp_hands, "timings", None),
                    )
                    frame_ms = None

                # Push to queue (drop oldest frame if consumer is lagging)
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put_nowait(payload)
                # Real-camera pacing comes from the blocking cap.read() itself;
                # the old unconditional 10 ms sleep here added a fixed frame of
                # latency at 60 fps for nothing. Only the camera-less path needs
                # a governor, and it now runs at capture_fps instead of ~100 Hz.
                if not self.camera_available or getattr(self, 'disable_camera', False):
                    time.sleep(1.0 / self.capture_fps)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.running = False

    def stop(self, join_timeout: float = 2.0):
        """
        Stop the background pipeline thread and wait for the camera handle to
        be released. Without the join, a game restarting the pipeline on the
        same camera index raced the old thread's `cap.release()` and silently
        fell back to simulated mode.
        """
        self._stop_requested = True
        self.running = False
        self.device_log.close(hand_device=self.hand_device_name)
        if self.depth_stream is not None:
            self.depth_stream.stop()
            self.depth_stream = None
            self.depth_map = None
            self.depth_active = False
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=join_timeout)

    # ── Frame processing ──────────────────────────────────────────────────────
    def _face_boxes(self, rgb: np.ndarray) -> list:
        """
        Faces in `rgb` as :class:`_FaceBox` values, best score first.

        The two backends do not agree on units: MediaPipe reports a box already
        normalised to the frame, while the OpenVINO wrapper reports pixels of
        the frame it was handed. They are reconciled here, once, rather than at
        each place downstream that reads a face.
        """
        if self._face_net is not None:
            height, width = rgb.shape[:2]
            return [_FaceBox(d.box[0] / width, d.box[1] / height,
                             (d.box[2] - d.box[0]) / width,
                             (d.box[3] - d.box[1]) / height)
                    for d in self._face_net.detect(rgb)]
        if self._mp_face is None:
            return []
        found = self._mp_face.process(rgb).detections
        return [_FaceBox(b.xmin, b.ymin, b.width, b.height)
                for b in (d.location_data.relative_bounding_box for d in (found or ()))]

    def _process_frame(self, bgr_frame: np.ndarray) -> dict:
        """Run face + hand detection on one BGR frame. Returns full payload."""
        run_detection = (self._detection_frame_count % self.detection_stride == 0)
        self._detection_frame_count += 1

        # The RGB copy exists only for the two `.process()` calls below, both of
        # which are already gated on `run_detection` — so converting before this
        # check meant a held frame (detection_stride > 1) paid a full-frame
        # allocate and colour convert that nothing then read.
        #
        # `writeable = False` is what lets MediaPipe's Python wrapper pass the
        # buffer straight to the graph: a writeable array is defensively copied
        # on entry to every process() call, and with both a face and a hand
        # graph we were paying that copy twice per frame. Nothing downstream
        # mutates `rgb` — the frame handed to consumers is `bgr_frame`.
        # The buffer is reused across frames rather than reallocated: nothing
        # keeps a reference to it past the process() calls below, and the
        # allocation was the larger half of the convert's cost.
        rgb = None
        if run_detection:
            buf = self._rgb_buf
            if buf is None or buf.shape != bgr_frame.shape or buf.dtype != bgr_frame.dtype:
                buf = self._rgb_buf = np.empty_like(bgr_frame)
            buf.flags.writeable = True
            cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB, dst=buf)
            buf.flags.writeable = False
            rgb = buf

        # ── Face detection ────────────────────────────────────────────────────
        target_x = self.width  / 2.0
        target_y = self.height / 2.0
        face_visible = False
        face_box = (0, 0, 0, 0)
        face_count = 0

        if self._face_net is not None or self._mp_face is not None:
            if run_detection:
                self._cached_face_detections = self._face_boxes(rgb)
            detections = self._cached_face_detections
            if detections:
                face_count = len(detections)
                bbox = detections[0]
                target_x = (bbox.xmin + bbox.width  / 2.0) * self.width
                target_y = (bbox.ymin + bbox.height / 2.0) * self.height
                # Box width is the only cheap proxy the RGB path has for how far
                # away the head is, which is what a game needs to scale anything
                # that should feel anchored to the player rather than the frame.
                face_visible = True
                face_box = (
                    round(bbox.xmin * self.width),
                    round(bbox.ymin * self.height),
                    round(bbox.width * self.width),
                    round(bbox.height * self.height),
                )

        # ── Hand gesture detection ────────────────────────────────────────────
        hands: list[dict] = []

        if self._mp_hands is not None:
            if run_detection:
                gate = face_box if (face_visible and self.face_hand_filter) else None
                before = getattr(self._mp_hands, "face_rejects", 0)
                copies_before = getattr(self._mp_hands, "duplicate_rejects", 0)
                # Only the accelerated backend takes a face box; MediaPipe's own
                # `process()` has no such argument, and neither does a test
                # double — so the backend is asked rather than assumed.
                if getattr(self._mp_hands, "accepts_face_box", False):
                    hand_results = self._mp_hands.process(rgb, gate)
                else:
                    hand_results = self._mp_hands.process(rgb)
                sets = list(hand_results.multi_hand_landmarks or [])
                scores = list(getattr(hand_results, "multi_handedness", None) or [])
                # What the backend threw out at the source, plus what it did not.
                rejected = getattr(self._mp_hands, "face_rejects", 0) - before
                if gate is not None:
                    sets, scores, dropped = self._drop_face_hands(sets, scores, gate)
                    rejected += dropped
                sets, scores, duplicates = self._drop_duplicate_hands(sets, scores)
                # ...plus the copies the accelerated backend dropped at the source.
                duplicates += (getattr(self._mp_hands, "duplicate_rejects", 0)
                               - copies_before)
                self.duplicate_hands_rejected_total += duplicates
                self._cached_landmark_sets = sets
                self._cached_handedness = scores
                self._face_hands_rejected = rejected
                self.face_hands_rejected_total += rejected
            landmark_sets = self._cached_landmark_sets
            handedness = self._cached_handedness

            assignment = self._assign_slots(landmark_sets)

            # One timestamp for the whole frame: both hands were captured in the
            # same exposure, so giving slot 1 a later `now` than slot 0 would put
            # a few hundred microseconds of made-up dt into its filter and its
            # velocity difference.
            frame_time = time.time()

            for mp_index, slot in sorted(assignment.items(), key=lambda kv: kv[1]):
                lm = landmark_sets[mp_index].landmark
                entry = self._extract_gesture(lm, slot=slot, now=frame_time)
                entry["slot"] = slot
                label, score = "unknown", 0.0
                if mp_index < len(handedness):
                    try:
                        cls = handedness[mp_index].classification[0]
                        label, score = cls.label, float(cls.score)
                    except (AttributeError, IndexError):
                        pass
                entry["handedness"] = label
                entry["handedness_score"] = score
                hands.append(entry)

            occupied = set(assignment.values())
            for slot in range(self.max_hands):
                if slot in occupied:
                    continue
                gs = self._gs_slots[slot]
                gs.lost_frames += 1
                if gs.lost_frames > 12:
                    gs.reset()
                    self._jitter_slots[slot].reset()
                    self._slot_anchor[slot] = None

        # The flat gesture keys describe the lowest occupied slot. With one hand
        # in view that is the same hand every frame, which is what every existing
        # game assumes; the difference only shows with two, where slot order is
        # stable and MediaPipe's list order is not.
        gesture = hands[0] if hands else self._empty_gesture()

        by_hand = {"Left": None, "Right": None}
        for entry in hands:
            if entry.get("handedness") in by_hand and by_hand[entry["handedness"]] is None:
                by_hand[entry["handedness"]] = entry

        payload = {
            # Face
            "target_x": target_x,
            "target_y": target_y,
            "face_visible": face_visible,
            "face_box": face_box,
            "face_count": face_count,
            # Hand candidates dropped as the player's own face, this frame and
            # over the session. See the _FACE_HAND_* notes.
            "face_hands_rejected": self._face_hands_rejected,
            "face_hands_rejected_total": self.face_hands_rejected_total,
            "frame":    bgr_frame,
            "scene_luma":     self.scene_luma,
            "low_light_gain": self.low_light_gain,
            "depth_device":   self.depth_device_name if self.depth_stream else "",
            "depth_fps":      self.depth_fps,
            "hand_device":    self.hand_device_name,
            "face_device":    self.face_device_name,
            # Hand (merged in from the primary hand's gesture dict)
            **gesture,
            # Every tracked hand, slot-ordered, plus handedness shortcuts.
            "hands":      tuple(hands),
            "hand_count": len(hands),
            "hand_left":  by_hand["Left"],
            "hand_right": by_hand["Right"],
        }

        if self.emit_depth_grid:
            payload["depth_grid"] = self._build_depth_grid(
                hands, face_box if face_visible else None)

        if self.emit_person_mask:
            payload["person_mask"] = self._get_person_mask(bgr_frame)

        return payload

    # ── Face-as-hand rejection ────────────────────────────────────────────────
    def _drop_face_hands(self, landmark_sets, handedness, face_box):
        """
        Drop the candidates that are the player's face, not a hand.

        Returns ``(landmark_sets, handedness, dropped)`` with the two lists
        still parallel — `_process_frame` indexes handedness by the landmark
        set's position, so filtering one without the other would hand slot 0
        the other hand's label.

        Both tests have to agree before anything is dropped; see the
        `_FACE_HAND_*` constants for why each one is here and what the case is
        that must survive (a real hand held over the face).
        """
        x, y, w, h = (float(v) for v in face_box)
        if w <= 0.0 or h <= 0.0:
            return landmark_sets, handedness, 0

        grow_x = w * (_FACE_BOX_DILATE - 1.0) / 2.0
        grow_y = h * (_FACE_BOX_DILATE - 1.0) / 2.0
        x0, y0 = x - grow_x, y - grow_y
        x1, y1 = x + w + grow_x, y + h + grow_y

        kept_sets, kept_scores, dropped = [], [], 0
        for i, lm_set in enumerate(landmark_sets):
            score = 1.0
            if i < len(handedness):
                try:
                    score = float(handedness[i].classification[0].score)
                except (AttributeError, IndexError):
                    score = 1.0

            points = lm_set.landmark
            inside = sum(
                1 for p in points
                if x0 <= p.x * self.width <= x1 and y0 <= p.y * self.height <= y1
            )
            is_face = (score < _FACE_HAND_HANDEDNESS
                       and inside >= _FACE_HAND_CONTAINED * len(points))
            if is_face:
                dropped += 1
                continue
            kept_sets.append(lm_set)
            # A placeholder rather than a skip: the lists are positional, so
            # leaving a gap here would give the next hand this one's label.
            # `_process_frame` already tolerates an entry it cannot read.
            kept_scores.append(handedness[i] if i < len(handedness) else None)
        return kept_sets, kept_scores, dropped

    def _drop_duplicate_hands(self, landmark_sets, handedness):
        """
        Drop the second copy of a hand the palm network found twice.

        Returns ``(landmark_sets, handedness, dropped)``, the two lists still
        parallel for the same reason as in `_drop_face_hands`. See the
        `_DUP_HAND_SPAN` note for why the test is proximity alone and why it
        is written in palm spans.
        """
        if len(landmark_sets) < 2:
            return landmark_sets, handedness, 0

        def measure(lm_set):
            pts = [(p.x * self.width, p.y * self.height) for p in lm_set.landmark]
            centre = ((pts[0][0] + pts[5][0] + pts[17][0]) / 3.0,
                      (pts[0][1] + pts[5][1] + pts[17][1]) / 3.0)
            # Wrist to middle-finger MCP: the one length that scales with the
            # hand and not with which fingers happen to be extended.
            return centre, _dist2d(pts[0], pts[9])

        def is_copy(a, b):
            (centre_a, span_a), (centre_b, span_b) = a, b
            return _dist2d(centre_a, centre_b) < _DUP_HAND_SPAN * min(span_a, span_b)

        kept_sets, kept_scores, kept, dropped = [], [], [], 0
        for i, lm_set in enumerate(landmark_sets):
            measured = measure(lm_set)
            # A degenerate span cannot be measured against, so such a set is
            # kept rather than guessed about.
            if measured[1] > 0.0 and any(
                is_copy(measured, other) for other in kept if other[1] > 0.0
            ):
                dropped += 1
                continue
            kept_sets.append(lm_set)
            kept_scores.append(handedness[i] if i < len(handedness) else None)
            kept.append(measured)
        return kept_sets, kept_scores, dropped

    # ── Hand → slot assignment ────────────────────────────────────────────────
    def _assign_slots(self, landmark_sets) -> dict[int, int]:
        """
        Map each detected hand (by MediaPipe list index) to a persistent slot.

        MediaPipe does not promise a stable ordering for ``multi_hand_landmarks``
        across frames, so keying per-hand filter state off that index makes two
        hands trade One-Euro histories whenever the order flips — both outputs
        snap on the same frame. Matching each hand to the slot whose last known
        wrist position is nearest keeps a hand on its own filter.
        """
        if not landmark_sets:
            return {}

        wrists = [
            (lm.landmark[0].x * self.width, lm.landmark[0].y * self.height)
            for lm in landmark_sets
        ]

        # A hand can move a long way between frames; the radius only has to be
        # tight enough to prefer the right slot when two candidates exist.
        max_match = 0.5 * max(self.width, self.height)

        pairs = sorted(
            (
                (_dist2d(wrists[i], self._slot_anchor[s]), i, s)
                for i in range(len(wrists))
                for s in range(self.max_hands)
                if self._slot_anchor[s] is not None
            ),
            key=lambda t: t[0],
        )

        assignment: dict[int, int] = {}
        used_slots: set[int] = set()
        for distance, i, s in pairs:
            if i in assignment or s in used_slots or distance > max_match:
                continue
            assignment[i] = s
            used_slots.add(s)

        for i in range(len(wrists)):
            if i in assignment:
                continue
            free = next((s for s in range(self.max_hands) if s not in used_slots), None)
            if free is None:
                break                      # more hands than slots — drop the extras
            assignment[i] = free
            used_slots.add(free)

        for i, s in assignment.items():
            self._slot_anchor[s] = wrists[i]
            self._gs_slots[s].lost_frames = 0
        return assignment

    # ── Person segmentation mask ──────────────────────────────────────────────
    def _get_person_mask(self, bgr_frame: np.ndarray) -> np.ndarray | None:
        """
        uint8 mask, same (height, width) as ``bgr_frame``, 255 = person.

        Builds :class:`~visual_ai.segment.PersonSegmenter` on first call —
        which downloads its model weights on first use — rather than in
        ``__init__``, so enabling ``emit_person_mask`` is the only thing that
        pays that cost. A failure here (no network on first use, a missing
        model file) disables the flag rather than raising every frame, the
        same way a missing MediaPipe module leaves ``self._mp_face`` as None
        instead of crashing construction.
        """
        if self._person_segmenter is None:
            try:
                from visual_ai.segment import PersonSegmenter
                self._person_segmenter = PersonSegmenter()
            except Exception as e:
                print(f"[VisionPipeline] PersonSegmenter init warning: {e}")
                self.emit_person_mask = False
                return None

        try:
            return self._person_segmenter.segment(bgr_frame)
        except Exception as e:
            print(f"[VisionPipeline] PersonSegmenter frame error: {e}")
            return None

    # ── Depth occupancy grid ──────────────────────────────────────────────────
    def _build_depth_grid(self, hands: list[dict], face_box) -> np.ndarray | None:
        """
        Whole-scene depth, downsampled to ``depth_grid_size``, in metres.
        Zero means "no return" — the same convention the raw sensor uses.

        With a real sensor this is a straight resize of ``depth_map``. Without
        one it is *synthesised* from the tracked hands and face so consumers can
        be developed and profiled against the real payload shape; the values are
        plausible, not measured, and ``depth_source`` still says so.
        """
        cols, rows = self.depth_grid_size

        if self.depth_map is not None and self.depth_map.size:
            grid = cv2.resize(
                self.depth_map.astype(np.float32), (cols, rows),
                interpolation=cv2.INTER_NEAREST,
            )
            return grid / 1000.0

        if not self.depth_simulated:
            return None

        grid = np.full((rows, cols), 2.0, dtype=np.float32)   # far wall at 2 m
        ys, xs = np.mgrid[0:rows, 0:cols]

        def blob(cx_px, cy_px, radius_px, depth_m):
            cx = cx_px * cols / max(1, self.width)
            cy = cy_px * rows / max(1, self.height)
            r = max(1.0, radius_px * cols / max(1, self.width))
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            mask = d2 <= r * r
            np.minimum(grid, np.where(mask, depth_m, np.inf), out=grid)

        if face_box and face_box[2] > 0:
            blob(face_box[0] + face_box[2] / 2.0, face_box[1] + face_box[3] / 2.0,
                 face_box[2] * 0.6, 0.9)
        for entry in hands:
            px, py = entry.get("pinch_pos", (0, 0))
            blob(px, py, 70.0, max(0.15, float(entry.get("tof_z_m", 0.45))))

        return grid

    # ── Gesture extraction ────────────────────────────────────────────────────
    def _smooth_and_motion(self, gs, raw_ix: float, raw_iy: float,
                           c_x: float, c_y: float, now: float) -> tuple:
        """
        One-Euro smoothing, plus the fingertip motion differenced from it.

        Mutates ``gs`` — filter state, velocity, acceleration, previous frame —
        and returns the two display positions the payload carries,
        ``(index_pos, pinch_pos)``.
        """
        # ── Landmark smoothing (One-Euro, velocity-adaptive) ──────────────────
        # Heavy smoothing at rest kills fingertip jitter; the cutoff opens up
        # with hand speed so fast motion stays lag-free. A fixed-alpha EMA
        # cannot do both, which is why tracking previously felt either shaky
        # or laggy depending on which way `smooth_alpha` was pushed.
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

        # ── Fingertip motion ──────────────────────────────────────────────────
        # Differenced from the smoothed fingertip, so a consumer gets a velocity
        # it can use directly rather than one it has to filter again. `raw_vel`
        # differences the unfiltered landmark alongside it: the gap between the
        # two is the phase lag One-Euro is currently costing, which is the only
        # honest way to tune `filter_beta` against a fast reversal.
        dt = 0.0 if gs.prev_time is None else (now - gs.prev_time)
        if dt > 1e-4:
            gs.vel = get_landmark_velocity(gs.prev_xy, (gs.smooth_ix, gs.smooth_iy), dt)
            gs.raw_vel = get_landmark_velocity(gs.prev_raw_xy, (raw_ix, raw_iy), dt)
            speed = math.hypot(*gs.vel)
            # Acceleration is differenced from an already-differenced signal, so
            # it is noisy by construction — EMA it rather than handing a
            # consumer a value that swings by 10x between adjacent frames.
            gs.accel = _ema((speed - gs.speed) / dt, gs.accel, alpha=0.3)
            gs.speed = speed
        gs.prev_time = now
        gs.prev_xy = (gs.smooth_ix, gs.smooth_iy)
        gs.prev_raw_xy = (raw_ix, raw_iy)

        return index_pos, pinch_pos

    def _classify_fingers(self, lm, gs) -> tuple:
        """
        Finger extension, the debounced hand sign, and the continuous grip.

        Returns ``(fingers_extended, is_isolated, hand_sign, grip_openness,
        finger_extension)``.

        Extension and grip both want each fingertip's and each PIP joint's
        distance from the wrist, and used to measure them separately — sixteen
        square roots per hand per frame where eight will do. ``reach`` holds
        those eight, keyed by landmark id.
        """
        def xyz(i: int) -> tuple:
            p = lm[i]
            return (p.x, p.y, p.z)

        # ── 3. Finger extension & hand sign ───────────────────────────────────
        # Anchored at the wrist (landmark 0): a finger counts as extended when
        # its tip sits further from the wrist than its own PIP joint. This is
        # tolerant of hand rotation and works for the thumb too, which the
        # previous landmark-9 anchor could not measure meaningfully.
        wrist = xyz(0)
        joints = ((8, 6), (12, 10), (16, 14), (20, 18))   # tip, pip per finger
        reach = {i: math.dist(xyz(i), wrist) for pair in joints for i in pair}

        # The wrist anchor is degenerate for the thumb: landmark 2 (thumb MCP)
        # sits almost on the wrist, so the tip clears it even with the thumb
        # folded flat across the palm — the T pip read as extended nearly all
        # the time. Measure abduction instead: a tucked thumb travels toward the
        # pinky MCP (17) while an extended one swings away from it, so compare
        # the tip against its own IP joint with a palm-scaled margin so a
        # resting thumb cannot flicker.
        pinky_mcp  = xyz(17)
        palm_span  = max(math.dist(wrist, pinky_mcp), 1e-6)
        was = gs.fingers_prev

        # Every test below is a threshold with hysteresis; see `_extended`. The
        # per-finger reach differences are kept as `finger_extension` — in palm
        # spans, signed, positive meaning extended — because "which finger is
        # the tracker unsure about" is not answerable from five booleans, and a
        # sign that will not settle is exactly the question a player asks.
        thumb_delta = (math.dist(xyz(4), pinky_mcp)
                       - math.dist(xyz(3), pinky_mcp)) / palm_span
        thumb_ext = _extended(thumb_delta, 0.10, _THUMB_EXT_MARGIN, was[0])
        deltas = [(reach[tip] - reach[pip]) / palm_span for tip, pip in joints]
        index_ext, middle_ext, ring_ext, pinky_ext = (
            _extended(delta, 0.0, _FINGER_EXT_MARGIN, previous)
            for delta, previous in zip(deltas, was[1:]))
        fingers_extended = (thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext)
        gs.fingers_prev = fingers_extended
        finger_extension = (round(thumb_delta, 4), *(round(d, 4) for d in deltas))

        # Relaxed index isolation: allow middle finger co-extension (natural tendon attachment),
        # requiring only ring and pinky to remain non-extended.
        is_isolated = index_ext and not (ring_ext or pinky_ext)

        # Debounced so a half-closed hand mid-transition cannot flap the sign
        # back and forth and fire a game action on every other frame.
        if self.gesture_mlp is not None:
            raw_sign, _ = self.gesture_mlp.predict(landmarks_to_features(lm))
        else:
            raw_sign = classify_hand_sign(fingers_extended)
        hand_sign = self._debounce_sign(raw_sign, gs)

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
        half_span = 0.5 * palm_span
        grip_openness = sum(
            min(1.0, max(0.0, (reach[tip] - reach[pip]) / half_span))
            for tip, pip in joints
        ) / 4.0

        return (fingers_extended, is_isolated, hand_sign, grip_openness,
                finger_extension)

    def _depth_and_click(self, gs, index_pos: tuple, z_val: float) -> tuple:
        """
        Depth sample, stabilizer gating, then the opt-in Z-push click.

        Returns ``(depth_active, depth_m, depth_raw_m, depth_source, z_click,
        z_delta, xy_drift)``. The last three stay inert unless `enable_z_click`.
        """
        # ── 4. Depth lookup ───────────────────────────────────────────────────
        depth_active, depth_raw_m, depth_source = self.sample_depth(
            index_pos[0], index_pos[1], z_val)

        # Feed raw sample to calibrator during sampling window
        if self.tof_stabilizer.state == DepthStabilizer.STATE_SAMPLING:
            self.tof_stabilizer.feed(depth_raw_m)

        # Gate ambient vibration out of Z (no-op when inactive)
        depth_m = self.tof_stabilizer.correct(depth_raw_m)

        # ── 5. Z-Push Click (opt-in via enable_z_click) ───────────────────────
        # Driven by stabilized metric depth when a sensor is present, so the
        # threshold is in real metres and can be floored by the stabilizer's
        # measured noise gate. Falls back to MediaPipe's relative Z otherwise.
        z_click, z_delta, xy_drift = False, 0.0, 0.0
        if self.enable_z_click:
            if depth_active:
                z_click, z_delta, xy_drift = self._detect_z_click(
                    depth_m, index_pos,
                    threshold=max(_Z_CLICK_THRESHOLD_M, self.tof_stabilizer.noise_gate * 2.0),
                    gs=gs, source="tof",
                )
            else:
                # MediaPipe Z decreases toward the camera, same sense as depth.
                z_click, z_delta, xy_drift = self._detect_z_click(
                    z_val, index_pos, threshold=_Z_CLICK_THRESHOLD,
                    gs=gs, source="mp",
                )

        return (depth_active, depth_m, depth_raw_m, depth_source,
                z_click, z_delta, xy_drift)

    def _extract_gesture(self, lm, slot: int = 0, now: float | None = None) -> dict:
        """
        Given MediaPipe hand landmarks (already mirrored via frame flip),
        compute and return the full gesture dict.

        The work is split across `_smooth_and_motion`, `_classify_fingers` and
        `_depth_and_click`; what is left here is fingertip positioning, the
        3-finger pinch edge, and assembling the payload. The keys below are the
        schema every game indexes — `_empty_gesture` must carry every one.

        ``slot`` selects which hand's filter / debounce / jitter state to use.
        Slot 0 is the default and is the state every single-hand consumer has
        always been driving.

        ``now`` overrides the timestamp fed to the One-Euro filters and the
        velocity difference. It exists because both are dt-dependent and
        wall-clock dt is whatever the machine happened to do: a test that steps
        landmarks in a tight loop sees a sub-100 us dt, which is below the guard
        on the velocity difference, so the motion keys never move off zero and
        the filter behaves nothing like it does at 30 fps.
        """
        gs = self._gs_slots[slot]
        W, H = self.width, self.height

        # Raw pixel positions (frame is already flipped, so no extra mirror needed)
        raw_ix = lm[8].x * W   # index tip
        raw_iy = lm[8].y * H
        raw_tx = lm[4].x * W   # thumb tip
        raw_ty = lm[4].y * H
        raw_mx = lm[12].x * W  # middle tip
        raw_my = lm[12].y * H

        thumb_pos  = (round(raw_tx), round(raw_ty))
        middle_pos = (round(raw_mx), round(raw_my))

        # Centroid of the 3 fingers
        c_x = (raw_tx + raw_ix + raw_mx) / 3.0
        c_y = (raw_ty + raw_iy + raw_my) / 3.0
        centroid_pos = (round(c_x), round(c_y))

        z_val = lm[8].z   # MediaPipe Z

        now = time.time() if now is None else float(now)
        index_pos, pinch_pos = self._smooth_and_motion(gs, raw_ix, raw_iy, c_x, c_y, now)

        # ── 1. 2-Finger Pinch (DISABLED — superseded by the 3-finger lock below)
        # The thumb/index distance test fired constantly at frame edges where
        # MediaPipe compresses the hand, so the 3-finger centroid test replaced
        # it. Kept out deliberately rather than left half-wired.

        # ── 3-Finger Pinch Trigger ───────────────────────────────────────────
        # Distance from each fingertip to the centroid.
        centroid = (c_x, c_y)
        max_dist_to_centroid = max(math.dist((raw_tx, raw_ty), centroid),
                                   math.dist((raw_ix, raw_iy), centroid),
                                   math.dist((raw_mx, raw_my), centroid))

        base_pinch_radius = 55.0
        effective_pinch_radius = base_pinch_radius * max(1.0, float(self.movement_magnification))
        is_3_pinching = max_dist_to_centroid < effective_pinch_radius

        # Trigger action on the rising edge of a 3-finger pinch
        click_fired = is_3_pinching and not gs.was_3_pinching
        gs.was_3_pinching = is_3_pinching
        gs.pinch_active = is_3_pinching

        (fingers_extended, is_isolated, hand_sign, grip_openness,
         finger_extension) = self._classify_fingers(lm, gs)

        # The Z-push click is evaluated after the depth section rather than
        # alongside the pinch above, because it wants the stabilized depth when
        # a sensor is present.
        (depth_active, depth_m, depth_raw_m, depth_source,
         z_click, z_delta, xy_drift) = self._depth_and_click(gs, index_pos, z_val)
        # OR'd with the 3-finger-pinch edge above, never assigned over it —
        # enabling the depth push must not silently disable the pinch click.
        click_fired = click_fired or z_click

        # ── 6. Jitter Calculation ──────────────────────────────────────────────
        jitter_stats = self._jitter_slots[slot].update(
            (raw_ix, raw_iy), (gs.smooth_ix, gs.smooth_iy))

        # MediaPipe scales Z like X, so the width converts all three the same
        # way and a consumer gets one coherent space to draw in.
        landmarks = tuple(
            (p.x * W, p.y * H, p.z * W) for p in lm
        ) if self.emit_landmarks else ()

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
            "landmarks":           landmarks,
            "is_pinching":         gs.pinch_active,
            "click_just_fired":    click_fired,
            "is_index_isolated":   is_isolated,
            # Hand signs
            "hand_sign":           hand_sign,
            "fingers_extended":    fingers_extended,
            "finger_extension":    finger_extension,
            "is_fist":             hand_sign == SIGN_FIST,
            "is_open_palm":        hand_sign == SIGN_OPEN_PALM,
            "grip_openness":       grip_openness,
            # Motion
            "index_velocity":      gs.vel,
            "index_velocity_raw":  gs.raw_vel,
            "index_speed":         gs.speed,
            "index_accel":         gs.accel,
            "smoothing_enabled":   self.smoothing_enabled,
            "z_delta":             z_delta,
            "xy_drift":            xy_drift,
            "depth_active":        depth_active,
            "depth_m":             depth_m,
            "depth_m_raw":         depth_raw_m,
            "depth_source":        depth_source,
            # Deprecated aliases -- see the schema note at the top.
            "tof_active":          depth_active,
            "tof_z_m":             depth_m,
            "tof_z_raw":           depth_raw_m,
            "is_3_finger_pinching":is_3_pinching,
            "jitter":              jitter_stats,
            # Stabilizer telemetry
            "stabilizer_state":     self.tof_stabilizer.state,
            "stabilizer_progress":  self.tof_stabilizer.progress,
            "stabilizer_noise_amp": self.tof_stabilizer.z_noise_amplitude,
            "stabilizer_gate":      self.tof_stabilizer.noise_gate,
        }

    # ── Hand sign debounce ────────────────────────────────────────────────────
    def _debounce_sign(self, raw_sign: str, gs: "_GestureState | None" = None) -> str:
        """
        Only report a sign once it has held for `_SIGN_DEBOUNCE` frames.

        Returns the currently confirmed sign, which may be the previous one
        while a new candidate is still being confirmed.
        """
        gs = self._gs if gs is None else gs

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
    def sample_depth(self, px: int, py: int, lm_z: float = 0.0) -> tuple[bool, float, str]:
        """
        Samples physical depth (in meters) at 2D (X, Y) pixel coordinates from ToF camera frame.
        Falls back to relative estimation if ToF is inactive.
        """
        if not (self.depth_active or self.depth_simulated):
            return False, 0.0, "RGB estimate (no depth sensor)"

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

        measured = z_m is not None
        if z_m is None:
            # Simulated hardware ToF depth reading centered around 0.45m calibrated baseline
            z_m = 0.45 + (lm_z * 0.6)

        z_m = round(max(0.15, z_m), 4)
        # Say what it is. The old labels said "ToF Hardware" for a number
        # computed from a MediaPipe landmark, which is how everyone came to
        # believe this engine had a depth sensor in it. That honesty has to
        # hold per frame too: a sensor that is open but stale (run() drops
        # maps older than DEPTH_STALE_S) or all holes at the fingertip
        # contributed nothing to this number, so neither the flag nor the
        # label may report it as a sensor reading.
        if measured:
            return True, z_m, f"Depth sensor ({self.depth_device_name})"
        if self.depth_active:
            return False, z_m, f"Depth estimate ({self.depth_device_name}: no reading)"
        # Simulated mode is an explicit opt-in to fabricated depth, so it
        # keeps reporting active=True — games asked to be lied to.
        return True, z_m, "Simulated depth (no sensor)"

    # ── Deprecated names ──────────────────────────────────────────────────────
    # "ToF" described a time-of-flight sensor this engine never had: the
    # numbers under these names were derived from a MediaPipe landmark. The
    # canonical names are depth_*, but six games and their HUDs read the old
    # ones, so they stay as aliases rather than breaking every consumer at
    # once. They are settable because games assign tof_simulated directly.
    #
    # Each now warns. The payload KEYS deliberately do not: they are the
    # schema every game indexes blind, and a warning there would fire once per
    # frame with nothing a player could do about it. Attributes and methods
    # are written by hand at a handful of call sites, which is exactly where a
    # warning can be acted on.

    @property
    def tof_active(self) -> bool:
        _warn_deprecated("tof_active", "depth_active")
        return self.depth_active

    @tof_active.setter
    def tof_active(self, value: bool) -> None:
        _warn_deprecated("tof_active", "depth_active")
        self.depth_active = bool(value)

    @property
    def tof_simulated(self) -> bool:
        _warn_deprecated("tof_simulated", "depth_simulated")
        return self.depth_simulated

    @tof_simulated.setter
    def tof_simulated(self, value: bool) -> None:
        _warn_deprecated("tof_simulated", "depth_simulated")
        self.depth_simulated = bool(value)

    @property
    def tof_device_name(self) -> str:
        _warn_deprecated("tof_device_name", "depth_device_name")
        return self.depth_device_name

    @tof_device_name.setter
    def tof_device_name(self, value: str) -> None:
        _warn_deprecated("tof_device_name", "depth_device_name")
        self.depth_device_name = str(value)

    def sample_tof_depth(self, px: int, py: int, lm_z: float = 0.0):
        """Deprecated alias of `sample_depth`."""
        _warn_deprecated("sample_tof_depth()", "sample_depth()")
        return self.sample_depth(px, py, lm_z)

    # ── Z-push click algorithm ────────────────────────────────────────────────
    def _detect_z_click(self, z_now: float, xy_now: tuple,
                        threshold: float | None = None,
                        gs: "_GestureState | None" = None,
                        source: str = "mp") -> tuple:
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
        # `gs` selects which hand's window to use — defaulting to slot 0 kept
        # two tracked hands sharing one Z history, so hand A's depth samples
        # could fire hand B's click.
        gs = self._gs if gs is None else gs
        thr = _Z_CLICK_THRESHOLD if threshold is None else float(threshold)

        # ToF metres (~0.45) and MediaPipe relative Z (~±0.05) are incompatible
        # units. If the depth source flipped since the last sample (sensor
        # drop-out, `depth_simulated` toggled mid-session), a window max in one
        # unit against z_now in the other reads as a huge push and fires a
        # phantom click — so the window restarts on every source change.
        if source != gs.z_source:
            gs.z_history.clear()
            gs.z_cooldown = 0
            gs.z_source = source

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
    #: Static text of the calibration overlay: (text, y_ratio, scale_k, BGR, thickness).
    _CALIB_STATIC_TEXT = (
        ("!! LID-SHAKE STABILIZATION !!",         0.32, 1.05, (0, 140, 255),   3),
        ("Please do not move or change position", 0.46, 0.75, (255, 255, 255), 2),
        ("Press  X  to cancel",                   0.90, 0.50, (140, 140, 140), 1),
    )

    def _calibration_chrome(self, h: int, w: int) -> dict:
        """
        The parts of the calibration overlay that depend only on frame size.

        Built once per resolution and reused. This whole overlay runs in the
        capture thread on every frame of a 3-second calibration, and it used to
        allocate a full-frame tint buffer and re-measure four fixed strings each
        time in order to draw the same pixels in the same places.
        """
        cached = self._calib_chrome
        if cached is not None and cached["size"] == (h, w):
            return cached

        scale_ref = w / 800.0
        tint = np.empty((h, w, 3), dtype=np.uint8)
        tint[:] = (15, 10, 30)                       # deep navy

        placed = []
        for text, y_ratio, scale_k, color, thick in self._CALIB_STATIC_TEXT:
            scale = scale_k * scale_ref
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
            placed.append((text, (w - tw) // 2, int(h * y_ratio), scale, color, thick))

        bar_w = int(w * 0.60)
        chrome = {
            "size":       (h, w),
            "tint":       tint,
            "scale_ref":  scale_ref,
            "secs_scale": 0.65 * scale_ref,
            "secs_y":     int(h * 0.58),
            "static":     placed,
            "bar":        ((w - bar_w) // 2, int(h * 0.68), bar_w, int(14 * scale_ref)),
            "ring":       (w // 2, int(h * 0.80), int(28 * scale_ref)),
        }
        self._calib_chrome = chrome
        return chrome

    @staticmethod
    def _calib_text(frame, text, tx, ty, scale, color, thick):
        """Centred DUPLEX text with the overlay's drop-shadow."""
        cv2.putText(frame, text, (tx + 2, ty + 2),
                    cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 2)
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, thick)

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
        chrome = self._calibration_chrome(h, w)

        # ── 1. Dark semi-transparent wash ────────────────────────────────────
        frame = cv2.addWeighted(frame, 0.22, chrome["tint"], 0.78, 0)

        # ── 2. Text blocks ───────────────────────────────────────────────────
        for text, tx, ty, scale, color, thick in chrome["static"]:
            self._calib_text(frame, text, tx, ty, scale, color, thick)

        # The only line whose width changes frame to frame, so the only one
        # still worth measuring here.
        secs_text = f"Calibrating...  {self.tof_stabilizer.time_remaining:.1f}s remaining"
        scale = chrome["secs_scale"]
        (tw, _), _ = cv2.getTextSize(secs_text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        self._calib_text(frame, secs_text, (w - tw) // 2, chrome["secs_y"],
                         scale, (0, 220, 200), 2)

        # ── 3. Progress bar ──────────────────────────────────────────────────
        bar_x, bar_y, bar_w, bar_h = chrome["bar"]
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 60), -1)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), (0, 180, 255), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (100, 100, 110), 1)

        # ── 4. Countdown ring ────────────────────────────────────────────────
        cx, cy, ring_r = chrome["ring"]
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
            "landmarks":         (),
            "is_pinching":       False,
            "click_just_fired":  False,
            "is_index_isolated": False,
            # Hand signs
            "hand_sign":         SIGN_UNKNOWN,
            "fingers_extended":  (False, False, False, False, False),
            "finger_extension":  (0.0, 0.0, 0.0, 0.0, 0.0),
            "is_fist":           False,
            "is_open_palm":      False,
            "grip_openness":     0.0,
            # Motion
            "index_velocity":     (0.0, 0.0),
            "index_velocity_raw": (0.0, 0.0),
            "index_speed":        0.0,
            "index_accel":        0.0,
            # Slot / handedness. Present here too so a consumer can read
            # payload["handedness"] on a no-hand frame without guarding.
            "slot":              0,
            "handedness":        "unknown",
            "handedness_score":  0.0,
            "smoothing_enabled": self.smoothing_enabled,
            "z_delta":           0.0,
            "xy_drift":          0.0,
            "depth_active":      False,
            "depth_m":           0.0,
            "depth_m_raw":       0.0,
            "depth_source":      "RGB estimate (no depth sensor)",
            "tof_active":        False,
            "tof_z_m":           0.0,
            "tof_z_raw":         0.0,
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
        payload = {
            "target_x": tx,
            "target_y": ty,
            "face_visible": False,
            "face_box":     (0, 0, 0, 0),
            "face_count":   0,
            "face_hands_rejected":       0,
            "face_hands_rejected_total": self.face_hands_rejected_total,
            "frame":    frame,
            "scene_luma":     self.scene_luma,
            "low_light_gain": self.low_light_gain,
            "depth_device":   self.depth_device_name if self.depth_stream else "",
            "depth_fps":      self.depth_fps,
            "hand_device":    self.hand_device_name,
            "face_device":    self.face_device_name,
            **self._empty_gesture(),
            "hands":      (),
            "hand_count": 0,
            "hand_left":  None,
            "hand_right": None,
        }
        if self.emit_depth_grid:
            payload["depth_grid"] = self._build_depth_grid([], None)
        if self.emit_person_mask:
            payload["person_mask"] = self._get_person_mask(frame)
        return payload
