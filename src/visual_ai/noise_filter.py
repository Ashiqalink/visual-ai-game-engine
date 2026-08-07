"""
noise_filter.py — Timing & Signal Stream Noise Filter wrapper for Visual AI Game Engine.

Suppresses transient false-positive gesture events, clicks, pinches, and OCR results
during camera warm-up, initial state loading, or scene transitions.
Provides generic input stream smoothing (EMA & One-Euro filter).
"""

import time
import math
from typing import Any, Dict, Optional, Tuple, Union


class NoiseFilter:
    """General-purpose time-based noise window filter."""

    def __init__(self, noise_duration: float = 2.0):
        """
        :param noise_duration: Time window in seconds during which detections/events are suppressed.
        """
        self.noise_duration: float = float(noise_duration)
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """Explicitly start or restart the noise filter timer."""
        self.start_time = time.time()

    def reset(self) -> None:
        """Reset the noise filter timer to current time."""
        self.start_time = time.time()

    def set_duration(self, duration: float) -> None:
        """Update the noise window duration in seconds."""
        self.noise_duration = max(0.0, float(duration))

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
        self.prev_val: Optional[Union[float, Tuple[float, ...]]] = None

    def filter(self, val: Union[float, Tuple[float, ...]]) -> Union[float, Tuple[float, ...]]:
        if self.prev_val is None:
            self.prev_val = val
            return val

        if isinstance(val, (int, float)):
            res = self.alpha * float(val) + (1.0 - self.alpha) * float(self.prev_val)
            self.prev_val = res
            return res

        if isinstance(val, (tuple, list)):
            prev_tuple = tuple(self.prev_val)
            res_list = [
                self.alpha * float(v) + (1.0 - self.alpha) * float(p)
                for v, p in zip(val, prev_tuple)
            ]
            res_tuple = tuple(res_list)
            self.prev_val = res_tuple
            return res_tuple

        return val

    def reset(self):
        self.prev_val = None


class FilteredGestureDetector:
    """
    Wrapper class for gesture & OCR detection engines.
    Intercepts and discards gesture inputs/events during the noise filter duration.
    """

    def __init__(self, detector_or_bus: Any = None, noise_duration: float = 2.0):
        self.noise_filter = NoiseFilter(noise_duration=noise_duration)
        self.detector = detector_or_bus
        self.bus = getattr(detector_or_bus, "bus", getattr(detector_or_bus, "event_bus", None))

    def is_noise_window_active(self) -> bool:
        """Check if currently within the noise suppression window."""
        return self.noise_filter.is_active()

    def reset_noise_filter(self) -> None:
        """Reset timing window when restarting game state or transitioning levels."""
        self.noise_filter.reset()

    def filter_event(self, event_name: str, payload: Any = None) -> Optional[Dict[str, Any]]:
        """
        Filters an incoming gesture event.
        Returns None if inside noise duration window, otherwise returns dict payload.
        """
        if self.noise_filter.is_active():
            return None
        return {"event": event_name, "payload": payload}

    def detect_gesture(self, frame: Any) -> Optional[Any]:
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

    def __init__(self, noise_duration: float = 2.0):
        self.filter = NoiseFilter(noise_duration=noise_duration)

    def process_payload(self, payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Processes and mutates pipeline payload dict.
        Zeroes out gesture trigger flags when inside noise window.
        """
        if payload is None:
            return None

        # Create shallow copy of payload to preserve raw data if needed
        filtered_payload = payload.copy()

        if self.filter.is_active():
            filtered_payload["is_pinching"] = False
            filtered_payload["click_just_fired"] = False
            filtered_payload["is_3_pinching"] = False
            filtered_payload["pinch_active"] = False
            filtered_payload["z_delta"] = 0.0

        return filtered_payload

    def reset(self) -> None:
        """Reset noise filter timer."""
        self.filter.reset()
