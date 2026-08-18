"""
jitter_analyzer.py — Real-time motion spatial jitter calculation tool for Visual AI Game Engine.

Calculates, over a rolling window, for both the raw and the smoothed stream:

- Detrended jitter RMS   — high-frequency wobble in pixels, with steady motion
                           removed. This is the number that actually answers
                           "how shaky is the cursor?"
- High-frequency energy  — mean magnitude of the second difference; sensitive to
                           frame-to-frame direction flips (the "spectrum" metric)
- Step statistics        — mean / max per-frame travel, i.e. speed and spikes
- Jitter reduction %     — how much of the raw jitter the smoothing removed

Why detrending matters
----------------------
An earlier version reported the standard deviation of per-frame step *distance*
as "jitter". Step distance is a speed, so that metric rises whenever the hand
moves quickly and smoothly — a perfectly clean sweep that accelerates reads as
heavy jitter, while a hand shaking in place at a constant rate reads as none.

Detrending fixes this. For each interior sample the local linear trend is
removed::

    residual[i] = p[i] - (p[i-1] + p[i+1]) / 2

Constant-velocity motion has zero residual regardless of speed, so only
high-frequency deviation survives.

Units of the reported jitter
---------------------------
``raw_jitter_std`` / ``smoothed_jitter_std`` are the **RMS magnitude of the
position's noise vector, in pixels** — "the fingertip wanders about this far
from where a smooth path would put it".

The detrending stencil inflates variance by 1.5x per axis, so the RMS residual
magnitude is divided by sqrt(1.5). That normalisation is independent of how many
axes are tracked: for isotropic per-axis noise sigma over ``d`` axes the metric
reads ``sigma * sqrt(d)``, i.e. 2.83 px for 2 px of shake on each of x and y.
Divide by ``sqrt(d)`` if you want the per-axis figure instead.
"""

import math
from collections import deque

import numpy as np

#: Residual-variance inflation from the (p[i-1] + p[i+1]) / 2 detrending stencil.
_DETREND_GAIN = math.sqrt(1.5)

#: Keys every stats dict carries, so consumers can index without .get() guards.
_STAT_KEYS = (
    "raw_jitter_std",
    "raw_jitter_mean_px",
    "raw_max_jump_px",
    "raw_hf_energy_px",
    "raw_speed_px",
    "smoothed_jitter_std",
    "smoothed_jitter_mean_px",
    "smoothed_max_jump_px",
    "smoothed_hf_energy_px",
    "smoothed_speed_px",
    "jitter_reduction_pct",
    "sample_count",
)


class JitterAnalyzer:
    """
    Computes position jitter metrics for 2D or 3D input landmark positions over
    a rolling frame window.

    Parameters
    ----------
    window_size : int
        Number of frames to keep in rolling history for metric calculation
        (default 30 ~ 1 second at 30 fps). Clamped to a minimum of 3, which is
        the smallest window the detrending stencil can use.
    """

    #: Minimum window the detrend stencil needs (one interior sample).
    MIN_WINDOW = 3

    def __init__(self, window_size: int = 30):
        # A window of 0 raised ValueError from deque(maxlen=-1); a window of 1
        # built a zero-length delta deque that silently discarded every sample.
        self.window_size = max(self.MIN_WINDOW, int(window_size))

        self.raw_positions = deque(maxlen=self.window_size)
        self.smoothed_positions = deque(maxlen=self.window_size)
        self.deltas_raw = deque(maxlen=self.window_size - 1)
        self.deltas_smoothed = deque(maxlen=self.window_size - 1)

        # Cumulative stats
        self.total_frames = 0

    @staticmethod
    def empty_stats() -> dict:
        """
        Zeroed stats dict with the full key set.

        Use this wherever a frame produces no measurement (no hand visible,
        simulated mode) so downstream consumers always see the same shape.
        """
        stats = {key: 0.0 for key in _STAT_KEYS}
        stats["sample_count"] = 0
        return stats

    def reset(self):
        """Clear history buffers."""
        self.raw_positions.clear()
        self.smoothed_positions.clear()
        self.deltas_raw.clear()
        self.deltas_smoothed.clear()
        self.total_frames = 0

    def update(self, raw_pos, smoothed_pos=None) -> dict:
        """
        Record a new coordinate sample and update jitter metrics.

        Parameters
        ----------
        raw_pos : (x, y) or (x, y, z)
            Unfiltered input landmark coordinate.
        smoothed_pos : same arity as raw_pos, optional
            Filtered / EMA smoothed coordinate for side-by-side jitter reduction
            assessment. Defaults to ``raw_pos`` (reduction then reads 0%).

        Returns
        -------
        dict containing current frame jitter statistics.
        """
        self.total_frames += 1

        raw = tuple(float(v) for v in raw_pos)
        smooth = raw if smoothed_pos is None else tuple(float(v) for v in smoothed_pos)

        # Mismatched arity would corrupt the paired comparison — fall back to
        # the shared prefix rather than raising inside the capture thread.
        if len(smooth) != len(raw):
            n = min(len(raw), len(smooth))
            raw, smooth = raw[:n], smooth[:n]

        # A dimensionality change mid-stream ((x, y) frames followed by an
        # (x, y, z) one) would put ragged tuples in the window: np.asarray in
        # _detrended_metrics raises on the ragged list, and _sub's zip would
        # silently truncate the delta. Restart the window in the new arity.
        if self.raw_positions and len(raw) != len(self.raw_positions[-1]):
            self.raw_positions.clear()
            self.smoothed_positions.clear()
            self.deltas_raw.clear()
            self.deltas_smoothed.clear()

        if self.raw_positions:
            prev_raw = self.raw_positions[-1]
            prev_smooth = self.smoothed_positions[-1]
            self.deltas_raw.append(_norm(_sub(raw, prev_raw)))
            self.deltas_smoothed.append(_norm(_sub(smooth, prev_smooth)))

        self.raw_positions.append(raw)
        self.smoothed_positions.append(smooth)

        return self.get_stats()

    def get_stats(self) -> dict:
        """Returns statistical jitter calculations over the current rolling window."""
        # Detrending needs at least one interior sample plus its neighbours.
        if len(self.raw_positions) < self.MIN_WINDOW or len(self.deltas_raw) < 2:
            stats = self.empty_stats()
            stats["sample_count"] = len(self.raw_positions)
            return stats

        raw_jitter, raw_hf = _detrended_metrics(self.raw_positions)
        smooth_jitter, smooth_hf = _detrended_metrics(self.smoothed_positions)

        raw_steps = np.asarray(self.deltas_raw, dtype=float)
        smooth_steps = np.asarray(self.deltas_smoothed, dtype=float)

        # Reduction is measured on detrended jitter, not on step distance, so
        # moving fast does not masquerade as poor smoothing. Clamped so a
        # smoother that adds noise reports a bounded negative rather than a
        # nonsense magnitude.
        if raw_jitter > 1e-5:
            reduction_pct = (1.0 - (smooth_jitter / raw_jitter)) * 100.0
            reduction_pct = max(-100.0, min(100.0, reduction_pct))
        else:
            reduction_pct = 0.0

        return {
            "raw_jitter_std":          round(raw_jitter, 2),
            "raw_jitter_mean_px":      round(float(raw_steps.mean()), 2),
            "raw_max_jump_px":         round(float(raw_steps.max()), 2),
            "raw_hf_energy_px":        round(raw_hf, 2),
            "raw_speed_px":            round(float(raw_steps.mean()), 2),
            "smoothed_jitter_std":     round(smooth_jitter, 2),
            "smoothed_jitter_mean_px": round(float(smooth_steps.mean()), 2),
            "smoothed_max_jump_px":    round(float(smooth_steps.max()), 2),
            "smoothed_hf_energy_px":   round(smooth_hf, 2),
            "smoothed_speed_px":       round(float(smooth_steps.mean()), 2),
            "jitter_reduction_pct":    round(reduction_pct, 1),
            "sample_count":            len(self.raw_positions),
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sub(a: tuple, b: tuple) -> tuple:
    return tuple(x - y for x, y in zip(a, b))


def _norm(v: tuple) -> float:
    return math.sqrt(sum(c * c for c in v))


def _detrended_metrics(positions) -> tuple[float, float]:
    """
    Returns ``(jitter_rms_px, hf_energy_px)`` for a position history.

    jitter_rms_px : RMS magnitude of the detrended residual, normalised back to
                    the RMS magnitude of the underlying position noise (see the
                    module docstring). Zero for any constant-velocity path.
    hf_energy_px  : mean magnitude of the second difference — the raw
                    high-frequency content, larger for rapid direction flips.
    """
    pts = np.asarray(positions, dtype=float)          # (N, dims)
    if pts.shape[0] < 3:
        return 0.0, 0.0

    # Second difference: p[i-1] - 2*p[i] + p[i+1]. The detrended residual is
    # exactly -0.5 * that, so both metrics come from one computation.
    second_diff = pts[:-2] - 2.0 * pts[1:-1] + pts[2:]
    second_diff_mag = np.linalg.norm(second_diff, axis=1)

    residual_mag = 0.5 * second_diff_mag
    jitter_rms = float(np.sqrt(np.mean(residual_mag ** 2))) / _DETREND_GAIN
    hf_energy = float(np.mean(second_diff_mag))

    return jitter_rms, hf_energy
