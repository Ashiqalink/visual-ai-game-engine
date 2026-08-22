"""
metrics.py — Scoring functions shared by the benches.

The three numbers that decide whether a smoother is good are in tension, so
all three are always reported together:

    jitter_rms   how much frame-to-frame wobble survives  (lower is better)
    rmse         how far the output sits from the truth   (lower is better)
    lag_ms       how late the output tracks the truth     (lower is better)

Any filter can win one of them alone — a hard freeze has zero jitter, a
passthrough has zero lag. Only a filter that wins all three is actually better.
"""

from __future__ import annotations

import math

import numpy as np

#: Variance inflation of the (p[i-1] + p[i+1]) / 2 detrending stencil.
#: Matches ``visual_ai.jitter_analyzer`` so numbers are comparable to the
#: engine's own live HUD readout.
_DETREND_GAIN = math.sqrt(1.5)


def _as2d(a) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def jitter_rms(series) -> float:
    """
    RMS magnitude of the detrended residual — high-frequency wobble with any
    steady motion removed, so a fast clean sweep scores 0.

    Same definition as ``JitterAnalyzer.raw_jitter_std``.
    """
    pts = _as2d(series)
    if pts.shape[0] < 3:
        return 0.0
    residual = 0.5 * np.linalg.norm(pts[:-2] - 2.0 * pts[1:-1] + pts[2:], axis=1)
    return float(np.sqrt(np.mean(residual ** 2))) / _DETREND_GAIN


def hf_energy(series) -> float:
    """Mean second-difference magnitude — sensitive to direction flips."""
    pts = _as2d(series)
    if pts.shape[0] < 3:
        return 0.0
    return float(np.mean(np.linalg.norm(pts[:-2] - 2.0 * pts[1:-1] + pts[2:], axis=1)))


def rmse(series, truth) -> float:
    """Root-mean-square distance from the noise-free truth."""
    a, b = _as2d(series), _as2d(truth)
    n = min(a.shape[0], b.shape[0])
    return float(np.sqrt(np.mean(np.sum((a[:n] - b[:n]) ** 2, axis=1))))


def max_error(series, truth) -> float:
    """Worst single-frame distance from the truth."""
    a, b = _as2d(series), _as2d(truth)
    n = min(a.shape[0], b.shape[0])
    return float(np.max(np.linalg.norm(a[:n] - b[:n], axis=1)))


def reduction_pct(after: float, before: float) -> float:
    """
    How much of ``before`` was removed, as a percentage, clamped to [-100, 100]
    so a filter that adds noise reports a bounded negative.
    """
    if before <= 1e-9:
        return 0.0
    return float(max(-100.0, min(100.0, (1.0 - after / before) * 100.0)))


def lag_ms(series, truth, dt: float, max_shift: int = 30) -> float:
    """
    Estimated tracking delay in milliseconds.

    Finds the whole-frame shift that best aligns ``series`` with ``truth``
    (lowest RMSE), then refines to sub-frame precision with a parabola through
    the three surrounding error values. A negative result means the output
    leads the truth, which only happens on overshoot.

    Only meaningful when the truth actually moves — a constant truth aligns
    equally well at every shift, so 0.0 is returned.
    """
    a, b = _as2d(series), _as2d(truth)
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    if n < 8 or float(np.max(np.abs(np.diff(b, axis=0)))) < 1e-9:
        return 0.0

    # A shift at or beyond the series length leaves an empty overlap, and
    # np.mean of an empty slice is NaN — one NaN in the error table makes the
    # min() below order-dependent garbage. Keep at least 4 overlapping frames
    # (n >= 8 is guaranteed above, so this never goes below 4).
    max_shift = min(max_shift, n - 4)

    def err(shift: int) -> float:
        # Positive shift = output is late: compare output[k+shift] to truth[k].
        if shift >= 0:
            x, y = a[shift:], b[:n - shift]
        else:
            x, y = a[:n + shift], b[-shift:]
        return float(np.mean(np.sum((x - y) ** 2, axis=1)))

    shifts = range(-max_shift, max_shift + 1)
    errors = {s: err(s) for s in shifts}
    best = min(errors, key=errors.get)

    # Parabolic refinement between the neighbouring error samples.
    frac = 0.0
    if -max_shift < best < max_shift:
        e0, e1, e2 = errors[best - 1], errors[best], errors[best + 1]
        denom = e0 - 2.0 * e1 + e2
        if abs(denom) > 1e-12:
            frac = 0.5 * (e0 - e2) / denom
            frac = max(-1.0, min(1.0, frac))

    return (best + frac) * dt * 1000.0


def overshoot_pct(series, truth) -> float:
    """
    Peak overshoot past a step, as a percentage of the step height. 0 for a
    filter that never exceeds its target.
    """
    a = np.asarray(series, dtype=float).ravel()
    b = np.asarray(truth, dtype=float).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]

    idx = np.flatnonzero(np.abs(np.diff(b)) > 1e-9)
    if idx.size == 0:
        return 0.0
    step_at = int(idx[0]) + 1
    height = b[step_at] - b[step_at - 1]
    if abs(height) < 1e-9:
        return 0.0
    beyond = (a[step_at:] - b[step_at:]) * np.sign(height)
    return float(max(0.0, beyond.max() / abs(height) * 100.0))


def settle_frames(series, truth, tol_frac: float = 0.05) -> float:
    """
    Frames after a step until the output stays within ``tol_frac`` of the step
    height, forever after. ``inf`` if it never settles.
    """
    a = np.asarray(series, dtype=float).ravel()
    b = np.asarray(truth, dtype=float).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]

    idx = np.flatnonzero(np.abs(np.diff(b)) > 1e-9)
    if idx.size == 0:
        return 0.0
    step_at = int(idx[0]) + 1
    height = abs(b[step_at] - b[step_at - 1])
    if height < 1e-9:
        return 0.0

    tol = tol_frac * height
    outside = np.flatnonzero(np.abs(a[step_at:] - b[step_at:]) > tol)
    if outside.size == 0:
        return 0.0
    last = int(outside[-1])
    return float("inf") if last >= n - step_at - 1 else float(last + 1)


def throughput(fn, samples, repeats: int = 3) -> float:
    """Median per-sample cost of ``fn`` in microseconds, over ``repeats`` passes."""
    import time

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for s in samples:
            fn(s)
        timings.append((time.perf_counter() - start) / len(samples) * 1e6)
    return float(np.median(timings))
