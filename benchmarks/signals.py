"""
signals.py — Deterministic synthetic input streams for the test benches.

Every generator returns ``(truth, raw)`` arrays of equal length:

    truth : the movement that actually happened (what a perfect filter would
            output — noise-free)
    raw   : what the sensor reported — truth plus the noise being defended
            against

Because ``truth`` is known exactly, the benches can score real accuracy
(RMSE against truth, lag against truth) rather than the usual
"looks smoother, therefore better" eyeball test. Every stream is seeded, so
two runs of the same bench produce identical numbers and a tuning change is
the only thing that can move a score.

Units: landmark streams are in **pixels** on an 800x600 frame; ToF streams are
in **metres**. Sample rate is 30 Hz throughout, matching ``capture_fps``.
"""

from __future__ import annotations

import numpy as np

FPS = 30.0
DT = 1.0 / FPS


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _t(n: int) -> np.ndarray:
    return np.arange(n) * DT


# ── 1-D landmark streams (pixels) ─────────────────────────────────────────────

def rest(n: int = 300, value: float = 400.0, jitter_px: float = 2.0,
         seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Hand held still. Pure sensor jitter — the case smoothing exists for."""
    truth = np.full(n, value)
    raw = truth + _rng(seed).normal(0.0, jitter_px, n)
    return truth, raw


def tremor(n: int = 300, value: float = 400.0, amp_px: float = 3.0,
           hz: float = 8.0, jitter_px: float = 1.0,
           seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Held still, but with a physiological 8 Hz hand tremor on top of jitter.

    Near the Nyquist limit of a 30 Hz stream, so a filter that only averages
    neighbouring frames cannot remove it without eating real motion too.
    """
    t = _t(n)
    truth = np.full(n, value)
    raw = truth + amp_px * np.sin(2 * np.pi * hz * t) + _rng(seed).normal(0, jitter_px, n)
    return truth, raw


def slow_drift(n: int = 300, value: float = 400.0, span_px: float = 120.0,
               jitter_px: float = 2.0, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Deliberate slow aim — a filter must track this without visible lag."""
    t = _t(n)
    truth = value + span_px * np.sin(2 * np.pi * 0.25 * t)
    raw = truth + _rng(seed).normal(0.0, jitter_px, n)
    return truth, raw


def fast_sweep(n: int = 300, start: float = 100.0, speed_px_s: float = 900.0,
               jitter_px: float = 2.5, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    A fast swipe across the frame and back. Punishes lag: an EMA tuned to kill
    resting jitter trails the true fingertip by several frames here.
    """
    t = _t(n)
    half = n // 2
    ramp = np.concatenate([t[:half], t[half - 1] - (t[half:] - t[half - 1])])
    truth = start + speed_px_s * ramp
    raw = truth + _rng(seed).normal(0.0, jitter_px, n)
    return truth, raw


def flick(n: int = 240, low: float = 250.0, high: float = 550.0,
          jitter_px: float = 2.0, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    A step: the hand snaps to a new position mid-stream and holds.

    Yields the classic step-response numbers — rise time, overshoot, settling.
    """
    truth = np.full(n, low)
    truth[n // 3:] = high
    raw = truth + _rng(seed).normal(0.0, jitter_px, n)
    return truth, raw


def dropouts(n: int = 300, value: float = 400.0, jitter_px: float = 2.0,
             spike_px: float = 120.0, every: int = 37,
             seed: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """
    Resting hand with occasional single-frame tracker glitches — MediaPipe
    briefly latching onto the wrong hand. Measures spike rejection.
    """
    truth = np.full(n, value)
    raw = truth + _rng(seed).normal(0.0, jitter_px, n)
    raw[every::every] += spike_px
    return truth, raw


# ── 2-D landmark streams (pixels) ─────────────────────────────────────────────

def circle_2d(n: int = 300, cx: float = 400.0, cy: float = 300.0,
              radius: float = 150.0, hz: float = 0.4, jitter_px: float = 2.0,
              seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """
    A circular sweep — the case where per-axis cutoffs would warp the path.
    Returns ``(N, 2)`` arrays.
    """
    t = _t(n)
    angle = 2 * np.pi * hz * t
    truth = np.stack([cx + radius * np.cos(angle), cy + radius * np.sin(angle)], axis=1)
    raw = truth + _rng(seed).normal(0.0, jitter_px, truth.shape)
    return truth, raw


# ── ToF depth streams (metres) ────────────────────────────────────────────────

def tof_rest(n: int = 300, depth: float = 0.45, shake_m: float = 0.004,
             seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Lid shake only: the hand is still, the laptop screen is vibrating."""
    truth = np.full(n, depth)
    raw = truth + _rng(seed).normal(0.0, shake_m, n)
    return truth, raw


def tof_punch(n: int = 300, depth: float = 0.45, reach: float = 0.22,
              shake_m: float = 0.004, seed: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """
    Rest, then a fast punch toward the camera and back, three times over.

    The signal the stabilizer must **not** destroy — the earlier
    mean-subtracting implementation flattened exactly this.
    """
    t = _t(n)
    truth = np.full(n, depth)
    for k in range(3):
        centre = (0.2 + 0.28 * k) * t[-1]
        truth -= reach * np.exp(-((t - centre) ** 2) / (2 * 0.09 ** 2))
    raw = truth + _rng(seed).normal(0.0, shake_m, n)
    return truth, raw


def tof_drift(n: int = 600, depth: float = 0.45, drift_m: float = 0.06,
              shake_m: float = 0.004, seed: int = 13) -> tuple[np.ndarray, np.ndarray]:
    """
    The player slowly leans back over 20 s while the lid still shakes. Tests
    that the baseline tracks genuine slow drift instead of gating it away.
    """
    truth = depth + drift_m * np.linspace(0.0, 1.0, n)
    raw = truth + _rng(seed).normal(0.0, shake_m, n)
    return truth, raw


def tof_blind(n: int = 300, shake_m: float = 0.0,
              seed: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """A disabled / blind ToF sensor: every reading is 0.0."""
    truth = np.zeros(n)
    return truth, truth.copy()


def tof_intermittent(n: int = 300, depth: float = 0.45, shake_m: float = 0.004,
                     valid_every: int = 25,
                     seed: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """
    A sensor that only returns a reading occasionally — everything else is 0.0.
    Too few valid samples to calibrate honestly.
    """
    truth = np.full(n, depth)
    raw = np.zeros(n)
    noise = _rng(seed).normal(0.0, shake_m, n)
    raw[::valid_every] = truth[::valid_every] + noise[::valid_every]
    return truth, raw


def tof_quiet(n: int = 300, depth: float = 0.45, shake_m: float = 0.0001,
              seed: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """A rock-steady rig: calibration noise is near zero, so the gate floor rules."""
    truth = np.full(n, depth)
    raw = truth + _rng(seed).normal(0.0, shake_m, n)
    return truth, raw
