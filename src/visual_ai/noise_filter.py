"""
noise_filter.py — Timing & Signal Stream Noise Filter wrapper for Visual AI Game Engine.

Suppresses transient false-positive gesture events, clicks, pinches, and OCR results
during camera warm-up, initial state loading, or scene transitions.
Provides generic input stream smoothing (EMA & One-Euro filter).
"""

import math
import time
from typing import Any

Numeric = float | tuple[float, ...]


class NoiseFilter:
    """General-purpose time-based noise window filter."""

    def __init__(self, noise_duration: float = 2.0):
        """
        :param noise_duration: Time window in seconds during which detections/events are suppressed.
        """
        self.noise_duration: float = float(noise_duration)
        self.start_time: float | None = None

    def start(self) -> None:
        """Explicitly start or restart the noise filter timer."""
        self.start_time = time.time()

    def reset(self) -> None:
        """Reset the noise filter timer to current time."""
        self.start_time = time.time()

    def elapsed(self) -> float:
        """Return elapsed time since noise filter started, or 0.0 if not started."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def is_active(self) -> bool:
        """
        Check if current time is within the active noise window.
        Automatically starts timer on first evaluation if not started yet.
        """
        if self.noise_duration <= 0.0:
            return False
        if self.start_time is None:
            self.start()
        return self.elapsed() < self.noise_duration


class GenericStreamFilter:
    """
    Exponential Moving Average (EMA) smoother for scalar numbers or 2D/3D tuples.
    Can be attached to any input stream (mouse, joystick, hand landmarks).
    """

    def __init__(self, alpha: float = 0.25):
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self.prev_val: Numeric | None = None

    def filter(self, val: Numeric) -> Numeric:
        if val is None:
            return val

        is_seq = isinstance(val, (tuple, list))
        prev_is_seq = isinstance(self.prev_val, (tuple, list))

        # A stream that switches shape (scalar -> tuple, or 2D -> 3D) used to
        # raise TypeError or silently drop components via a short zip(). Restart
        # the filter instead: history from a different shape is meaningless.
        if self.prev_val is None or is_seq != prev_is_seq or (
            is_seq and len(val) != len(self.prev_val)
        ):
            self.prev_val = tuple(float(v) for v in val) if is_seq else val
            return val

        if is_seq:
            res_tuple = tuple(
                self.alpha * float(v) + (1.0 - self.alpha) * float(p)
                for v, p in zip(val, self.prev_val)
            )
            self.prev_val = res_tuple
            return res_tuple

        if isinstance(val, (int, float)) and not isinstance(val, bool):
            res = self.alpha * float(val) + (1.0 - self.alpha) * float(self.prev_val)
            self.prev_val = res
            return res

        return val

    def reset(self):
        self.prev_val = None


class OneEuroFilter:
    """
    One-Euro filter (Casiez, Roussel & Vogel, 2012) for scalar or vector input.

    A plain EMA forces one trade-off for the whole stream: enough smoothing to
    kill resting jitter also adds visible lag to fast motion. One-Euro adapts
    the cutoff to the signal's own speed — heavy smoothing when the hand is
    nearly still, near-passthrough when it moves fast — so jitter and lag are
    reduced at the same time.

    For vector input the cutoff is derived from the *magnitude* of the velocity
    across all components, so a diagonal sweep is filtered identically to a
    horizontal one (per-axis cutoffs would warp the path).

    Parameters
    ----------
    freq : float
        Nominal sample rate (Hz), used until real timestamps arrive.
    min_cutoff : float
        Cutoff frequency (Hz) at zero speed. Lower = steadier at rest, laggier.
    beta : float
        Speed coupling. Higher = more responsive to fast motion. Units are
        Hz per (input-unit / second), so pixel streams want small values
        (~0.005) and normalized 0–1 streams want large ones (~5).
    d_cutoff : float
        Cutoff (Hz) for the internal velocity estimate.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.freq = max(1e-3, float(freq))
        self.min_cutoff = max(1e-3, float(min_cutoff))
        self.beta = max(0.0, float(beta))
        self.d_cutoff = max(1e-3, float(d_cutoff))

        self._x_prev: tuple[float, ...] | None = None
        self._dx_prev: tuple[float, ...] | None = None
        self._t_prev: float | None = None
        self._scalar: bool = False

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        """Forget all history; the next sample re-seeds the filter."""
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    @property
    def initialized(self) -> bool:
        return self._x_prev is not None

    @property
    def value(self) -> Numeric | None:
        """Most recent filtered output, or None before the first sample."""
        if self._x_prev is None:
            return None
        return self._x_prev[0] if self._scalar else self._x_prev

    def filter(self, val: Numeric, timestamp: float | None = None) -> Numeric:
        """
        Filter one sample.

        Parameters
        ----------
        val : float or sequence of float
        timestamp : float, optional
            Monotonic time in seconds. When omitted the nominal ``freq`` is used
            as the timestep, which is fine for fixed-rate capture loops.
        """
        scalar = not isinstance(val, (tuple, list))
        vec: tuple[float, ...] = (float(val),) if scalar else tuple(float(v) for v in val)

        # A single NaN/inf sample would otherwise seed _x_prev/_dx_prev and
        # every later output stays NaN until reset() — in the pipeline that is
        # 12 hand-lost frames away, so one bad landmark poisoned the stream
        # for as long as the hand stayed visible. Hold the last good value
        # instead (or pass the bad sample through unfiltered if there is none).
        if not all(math.isfinite(v) for v in vec):
            if self._x_prev is not None:
                return self._x_prev[0] if self._scalar else self._x_prev
            return val

        # Shape change invalidates the history (same reasoning as GenericStreamFilter).
        if self._x_prev is not None and (scalar != self._scalar or len(vec) != len(self._x_prev)):
            self.reset()

        self._scalar = scalar

        if self._x_prev is None:
            self._x_prev = vec
            self._dx_prev = tuple(0.0 for _ in vec)
            self._t_prev = timestamp
            return val

        if timestamp is not None and self._t_prev is not None:
            dt = timestamp - self._t_prev
            # Duplicate/rewound timestamps would divide by ~0 and blow the alpha up.
            if not math.isfinite(dt) or dt <= 1e-6:
                dt = 1.0 / self.freq
            else:
                self.freq = 1.0 / dt
        else:
            dt = 1.0 / self.freq
        self._t_prev = timestamp

        # Low-pass the derivative, then set the cutoff from its magnitude.
        a_d = self._alpha(self.d_cutoff, dt)
        dx = tuple((v - p) / dt for v, p in zip(vec, self._x_prev))
        dx_hat = tuple(a_d * d + (1.0 - a_d) * dp for d, dp in zip(dx, self._dx_prev))
        speed = math.sqrt(sum(d * d for d in dx_hat))

        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)

        x_hat = tuple(a * v + (1.0 - a) * p for v, p in zip(vec, self._x_prev))

        self._x_prev = x_hat
        self._dx_prev = dx_hat

        return x_hat[0] if scalar else x_hat


def ema_alpha_to_cutoff(alpha: float, freq: float = 30.0) -> float:
    """
    Convert a first-order EMA smoothing factor to the equivalent One-Euro
    ``min_cutoff`` in Hz, so existing ``smooth_alpha`` tuning carries over as
    the *resting* smoothness of a One-Euro filter.
    """
    a = max(1e-4, min(0.999, float(alpha)))
    return (a * float(freq)) / (2.0 * math.pi * (1.0 - a))


class FilteredGestureDetector:
    """
    Wrapper class for gesture & OCR detection engines.
    Intercepts and discards gesture inputs/events during the noise filter duration.
    """

    def __init__(self, detector_or_bus: Any = None, noise_duration: float = 2.0):
        self.noise_filter = NoiseFilter(noise_duration=noise_duration)
        self.detector = detector_or_bus

    def detect_gesture(self, frame: Any) -> Any | None:
        """
        Detects gesture on input frame. If within noise duration, returns None without processing.
        """
        if self.noise_filter.is_active():
            return None

        if self.detector and hasattr(self.detector, "detect_gesture"):
            return self.detector.detect_gesture(frame)
        return None


class PipelineNoiseFilter:
    """
    Applies noise filtering directly to VisionPipeline payload outputs.
    Suppresses transient gesture flags while passing coordinates and camera frames through.
    """

    #: Boolean trigger flags zeroed while the noise window is open.
    #: ``is_3_finger_pinching`` is the key VisionPipeline actually emits — the
    #: previous list named only ``is_3_pinching``/``pinch_active``, which exist
    #: in no payload, so the 3-finger pinch was never suppressed and two bogus
    #: keys were injected into every frame instead.
    #:
    #: Only transient triggers belong here. The filter can only rewrite its
    #: payload copy, not the pipeline's internal counters, so masking any
    #: persistent state would make the HUD snap the instant the window closed.
    SUPPRESSED_FLAGS = (
        "is_pinching",
        "click_just_fired",
        "is_3_finger_pinching",
        # Legacy aliases — suppressed when a caller supplies them.
        "is_3_pinching",
        "pinch_active",
    )

    #: Numeric fields zeroed while the noise window is open.
    SUPPRESSED_SCALARS = (
        "z_delta",
        "xy_drift",
    )

    def __init__(self, noise_duration: float = 2.0):
        self.filter = NoiseFilter(noise_duration=noise_duration)

    def process_payload(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Zero out gesture trigger flags while inside the noise window.

        Only keys already present are modified — the filter never invents keys,
        so a payload shape stays exactly as the pipeline defined it. Inside the
        window the result is a shallow copy and the caller's dict is untouched;
        outside it the caller's dict is handed straight back.
        """
        if payload is None:
            return None

        # The window is open for the first couple of seconds of a session and
        # shut for the rest of it — and shut permanently when noise_duration is
        # <= 0, which is how the filter ships. Copying a ~55-key payload every
        # frame in order to change nothing in it was the entire per-frame cost
        # of a feature that is off by default. Nothing is modified on this path,
        # so there is no raw data left for a copy to preserve.
        #
        # is_active() is still what decides, so it keeps starting the timer on
        # first evaluation exactly as before.
        if not self.filter.is_active():
            return payload

        def _suppress(d: dict) -> None:
            for key in self.SUPPRESSED_FLAGS:
                if key in d:
                    d[key] = False
            for key in self.SUPPRESSED_SCALARS:
                if key in d:
                    d[key] = 0.0

        filtered_payload = payload.copy()
        _suppress(filtered_payload)

        # A shallow copy still shares the per-hand gesture dicts, so the flags
        # would survive the window inside "hands"/"hand_left"/"hand_right" and
        # any multi-hand consumer would bypass the gate. Suppress copies, and
        # keep the handedness shortcuts aliasing the "hands" entries exactly as
        # the pipeline built them.
        if filtered_payload.get("hands"):
            copies = {id(hand): hand.copy() for hand in filtered_payload["hands"]}
            for hand_copy in copies.values():
                _suppress(hand_copy)
            filtered_payload["hands"] = tuple(
                copies[id(hand)] for hand in filtered_payload["hands"])
            for key in ("hand_left", "hand_right"):
                original = filtered_payload.get(key)
                if original is not None and id(original) in copies:
                    filtered_payload[key] = copies[id(original)]

        return filtered_payload

    def reset(self) -> None:
        """Reset noise filter timer."""
        self.filter.reset()
