"""
bench_filters.py — Test bench for the landmark smoothing filters.

Runs every filter in ``visual_ai.noise_filter`` over the same synthetic hand
streams and scores each one on the three numbers that trade off against each
other: surviving jitter, error against the truth, and tracking lag.

Run it::

    python benchmarks/bench_filters.py
    python benchmarks/bench_filters.py --html report.html

Filters under test
------------------
``raw``               no smoothing — the control
``ema-0.20``          ``GenericStreamFilter``, the pipeline's legacy default
``ema-0.35``          a looser EMA, for the lag/jitter trade-off comparison
``one-euro (shipped)````OneEuroFilter`` with exactly the pipeline's defaults
``one-euro (snappy)`` a higher-beta tuning, for comparison

The pass/fail checks are applied to **the shipped configuration only** — the
other rows are context that shows whether the shipped tuning sits in a sensible
place.
"""

from __future__ import annotations

import numpy as np

from harness import BenchResult, Scenario, bootstrap
import metrics
import signals

_MODULES = bootstrap()

from visual_ai.noise_filter import (  # noqa: E402  (import must follow bootstrap)
    GenericStreamFilter,
    OneEuroFilter,
    ema_alpha_to_cutoff,
)

FPS = signals.FPS
DT = signals.DT

#: The tuning VisionPipeline actually constructs (see ``_make_position_filter``):
#: ``smooth_alpha=0.20`` reinterpreted as a resting cutoff, ``filter_beta=0.006``.
SHIPPED_ALPHA = 0.20
SHIPPED_BETA = 0.006
SHIPPED_LABEL = "one-euro (shipped)"


# ── Filter adapters ───────────────────────────────────────────────────────────

def _filters() -> dict:
    """Fresh instances of every filter under test, keyed by display label."""
    shipped_cutoff = ema_alpha_to_cutoff(SHIPPED_ALPHA, FPS)
    return {
        "raw": None,
        "ema-0.20": GenericStreamFilter(alpha=0.20),
        "ema-0.35": GenericStreamFilter(alpha=0.35),
        SHIPPED_LABEL: OneEuroFilter(freq=FPS, min_cutoff=shipped_cutoff,
                                     beta=SHIPPED_BETA, d_cutoff=1.0),
        "one-euro (snappy)": OneEuroFilter(freq=FPS, min_cutoff=shipped_cutoff,
                                           beta=0.05, d_cutoff=1.0),
    }


def _run_filter(filt, raw: np.ndarray) -> np.ndarray:
    """Push a stream through one filter, feeding real timestamps as the pipeline does."""
    if filt is None:
        return raw.copy()

    vector = raw.ndim == 2
    out = []
    for i, sample in enumerate(raw):
        value = tuple(float(v) for v in sample) if vector else float(sample)
        if isinstance(filt, OneEuroFilter):
            result = filt.filter(value, timestamp=i * DT)
        else:
            result = filt.filter(value)
        out.append(result)
    return np.asarray(out, dtype=float)


def _score(out: np.ndarray, raw: np.ndarray, truth: np.ndarray,
           step: bool = False) -> dict:
    """
    The full metric set for one filter on one scenario.

    ``step`` gates the step-response metrics. Overshoot and settling are
    defined against a single discontinuity, so on a continuously moving signal
    they would anchor to frame 1 of the ramp and report nonsense — they are
    left at zero there and shown as "—" instead.
    """
    settle = metrics.settle_frames(out, truth) if (step and out.ndim == 1) else 0.0
    return {
        "step": step,
        "jitter": metrics.jitter_rms(out),
        "jitter_reduction_pct": metrics.reduction_pct(metrics.jitter_rms(out),
                                                      metrics.jitter_rms(raw)),
        "rmse": metrics.rmse(out, truth),
        "max_error": metrics.max_error(out, truth),
        "lag_ms": metrics.lag_ms(out, truth, DT),
        "overshoot_pct": (metrics.overshoot_pct(out, truth)
                          if (step and out.ndim == 1) else 0.0),
        "settle_frames": settle,
        "hf_energy": metrics.hf_energy(out),
    }


#: Metrics where a smaller number is the better result.
_LOWER_IS_BETTER = frozenset({
    "jitter", "rmse", "max_error", "lag_ms", "overshoot_pct", "settle_frames", "hf_energy",
})


def _fmt(value: float, digits: int = 2) -> str:
    if value == float("inf"):
        return "never"
    return f"{value:,.{digits}f}"


# ── Scenario definitions ──────────────────────────────────────────────────────

def _scenarios() -> list[dict]:
    """
    Each entry pairs a signal with the requirements the *shipped* filter must
    meet on it. Thresholds are deliberately loose enough to pass today and
    tight enough that a real regression trips them.
    """
    return [
        {
            "name": "rest",
            "description": "Hand held still, 2 px of sensor jitter. The case smoothing exists for.",
            "signal": signals.rest(),
            "unit": "px",
            "requirements": [
                ("jitter reduction", "jitter_reduction_pct", {"min_value": 55.0}, "%"),
                ("error vs truth", "rmse", {"max_value": 1.5}, "px"),
            ],
        },
        {
            "name": "tremor 8 Hz",
            "description": "Held still with a 3 px physiological tremor near the Nyquist limit.",
            "signal": signals.tremor(),
            "unit": "px",
            "requirements": [
                ("jitter reduction", "jitter_reduction_pct", {"min_value": 40.0}, "%"),
                ("error vs truth", "rmse", {"max_value": 3.0}, "px"),
            ],
        },
        {
            "name": "slow aim",
            "description": "Deliberate slow tracking motion, 0.25 Hz over 120 px.",
            "signal": signals.slow_drift(),
            "unit": "px",
            "requirements": [
                ("error vs truth", "rmse", {"max_value": 8.0}, "px"),
                ("lag", "lag_ms", {"max_value": 60.0}, "ms"),
            ],
        },
        {
            "name": "fast sweep",
            "description": "900 px/s swipe and return — the case that exposes EMA lag.",
            "signal": signals.fast_sweep(),
            "unit": "px",
            "requirements": [
                ("lag", "lag_ms", {"max_value": 70.0}, "ms"),
                ("error vs truth", "rmse", {"max_value": 40.0}, "px"),
            ],
        },
        {
            "name": "flick (step)",
            "description": "300 px snap to a new position, then hold. Step response.",
            "signal": signals.flick(),
            "unit": "px",
            "step": True,
            "requirements": [
                ("overshoot", "overshoot_pct", {"max_value": 10.0}, "%"),
                ("settling", "settle_frames", {"max_value": 20.0}, "frames"),
            ],
        },
        {
            "name": "tracker glitches",
            "description": "Resting hand with a 120 px single-frame mis-track every 37 frames.",
            "signal": signals.dropouts(),
            "unit": "px",
            # One-Euro reads a single-frame glitch as fast motion and opens its
            # cutoff, so it leaks more of the spike than a fixed EMA would.
            # The excursion cap is the real guard — rmse is graded loosely
            # because eight injected 120 px glitches dominate it by design.
            "requirements": [
                ("worst excursion", "max_error", {"max_value": 90.0}, "px"),
                ("error vs truth", "rmse", {"max_value": 18.0}, "px"),
            ],
        },
        {
            "name": "2-D circle",
            "description": "Circular sweep — checks the path is not warped per axis.",
            "signal": signals.circle_2d(),
            "unit": "px",
            "requirements": [
                ("error vs truth", "rmse", {"max_value": 12.0}, "px"),
                ("lag", "lag_ms", {"max_value": 70.0}, "ms"),
            ],
        },
    ]


# ── Bench ─────────────────────────────────────────────────────────────────────

def run() -> BenchResult:
    result = BenchResult(
        name="Landmark smoothing filters",
        subtitle=f"visual_ai.noise_filter — {_MODULES['visual_ai.noise_filter']}",
    )

    comparison_rows: list[list[str]] = []

    for spec in _scenarios():
        truth, raw = spec["signal"]
        scen = Scenario(name=spec["name"], description=spec["description"],
                        y_label=spec["unit"])

        is_step = spec.get("step", False)
        scored: dict[str, dict] = {}
        outputs: dict[str, np.ndarray] = {}

        for label, filt in _filters().items():
            out = _run_filter(filt, raw)
            outputs[label] = out
            scored[label] = _score(out, raw, truth, step=is_step)

        # Per-scenario comparison block, all filters side by side.
        for label, sc in scored.items():
            comparison_rows.append([
                spec["name"] if label == "raw" else "",
                label,
                _fmt(sc["jitter"]),
                f"{sc['jitter_reduction_pct']:+.0f}%" if label != "raw" else "—",
                _fmt(sc["rmse"]),
                _fmt(sc["lag_ms"], 1),
                _fmt(sc["overshoot_pct"], 1) if is_step else "—",
                _fmt(sc["settle_frames"], 0) if is_step else "—",
            ])

        shipped = scored[SHIPPED_LABEL]
        scen.metrics = {k: v for k, v in shipped.items()}
        scen.metrics["filter"] = SHIPPED_LABEL

        # Which alternative tuning, if any, clearly beats the shipped one here.
        # Reported as context, never as a failure: a tuning that wins one
        # scenario usually loses another, which is the whole point of the table.
        rivals = {l: s for l, s in scored.items() if l not in ("raw", SHIPPED_LABEL)}

        for name, key, bounds, unit in spec["requirements"]:
            detail = ""
            if rivals and key in _LOWER_IS_BETTER:
                winner = min(rivals, key=lambda l: rivals[l][key])
                if rivals[winner][key] < shipped[key] * 0.9:
                    detail = f"{winner} scores {rivals[winner][key]:,.4g} here"
            scen.check(name, shipped[key], unit, detail=detail, **bounds)

        # Traces for the charts: truth, raw, the shipped filter, and one EMA
        # for contrast. 2-D scenarios chart the x axis only.
        def flat(a: np.ndarray) -> list[float]:
            return list(a[:, 0]) if a.ndim == 2 else list(a)

        scen.traces = {
            "raw": flat(raw),
            "truth": flat(truth),
            SHIPPED_LABEL: flat(outputs[SHIPPED_LABEL]),
            "ema-0.20": flat(outputs["ema-0.20"]),
        }
        result.scenarios.append(scen)

    result.tables.append({
        "title": "All filters, all scenarios",
        "note": ("jitter = detrended wobble that survived (lower better) · "
                 "rmse = distance from the true motion · lag = tracking delay"),
        "headers": ["scenario", "filter", "jitter", "reduction", "rmse",
                    "lag ms", "overshoot", "settle"],
        "rows": comparison_rows,
        "aligns": "llrrrrrr",
    })
    result.tables.append(_throughput_table())
    result.scenarios.append(_tradeoff_verdict())

    return result


def _throughput_table() -> dict:
    """Per-sample cost of each filter — smoothing must not cost frame time."""
    _, raw = signals.slow_drift(n=2000)
    rows = []
    for label, filt in _filters().items():
        if filt is None:
            continue
        us = metrics.throughput(lambda v: filt.filter(float(v)), raw)
        rows.append([label, f"{us:,.2f}", f"{us * 2 * 30 / 1000:,.3f}"])
    return {
        "title": "Cost per sample",
        "note": "budget column = two filtered landmarks at 30 fps, as % of a 33 ms frame",
        "headers": ["filter", "µs / sample", "% of frame budget"],
        "rows": rows,
        "aligns": "lrr",
    }


def _tradeoff_verdict() -> Scenario:
    """
    The design claim One-Euro exists to make: at matched resting smoothness it
    should track fast motion with less lag than the EMA it replaced.

    Both filters are measured on the same two streams — the resting stream sets
    smoothness, the sweep sets lag.
    """
    scen = Scenario(
        name="one-euro vs ema trade-off",
        description=("The reason the pipeline moved off a plain EMA: same resting "
                     "steadiness, less lag when the hand actually moves."),
        y_label="px",
    )

    cutoff = ema_alpha_to_cutoff(SHIPPED_ALPHA, FPS)

    _, rest_raw = signals.rest()
    sweep_truth, sweep_raw = signals.fast_sweep()

    ema_rest = _run_filter(GenericStreamFilter(alpha=SHIPPED_ALPHA), rest_raw)
    euro_rest = _run_filter(
        OneEuroFilter(freq=FPS, min_cutoff=cutoff, beta=SHIPPED_BETA), rest_raw)

    ema_sweep = _run_filter(GenericStreamFilter(alpha=SHIPPED_ALPHA), sweep_raw)
    euro_sweep = _run_filter(
        OneEuroFilter(freq=FPS, min_cutoff=cutoff, beta=SHIPPED_BETA), sweep_raw)

    ema_jitter, euro_jitter = metrics.jitter_rms(ema_rest), metrics.jitter_rms(euro_rest)
    ema_lag = metrics.lag_ms(ema_sweep, sweep_truth, DT)
    euro_lag = metrics.lag_ms(euro_sweep, sweep_truth, DT)

    scen.metrics = {
        "ema_rest_jitter_px": ema_jitter,
        "one_euro_rest_jitter_px": euro_jitter,
        "ema_sweep_lag_ms": ema_lag,
        "one_euro_sweep_lag_ms": euro_lag,
    }

    scen.check("resting jitter vs ema", euro_jitter - ema_jitter, "px",
               max_value=0.25,
               detail=f"ema {ema_jitter:.2f} px · one-euro {euro_jitter:.2f} px")
    scen.check("lag saved on fast motion", ema_lag - euro_lag, "ms",
               min_value=5.0,
               detail=f"ema {ema_lag:.1f} ms · one-euro {euro_lag:.1f} ms")

    scen.traces = {
        "sweep truth": list(sweep_truth),
        "ema-0.20": list(ema_sweep),
        "one-euro": list(euro_sweep),
    }
    scen.notes = ("A negative 'resting jitter vs ema' value means One-Euro is also "
                  "steadier at rest, not merely equal.")
    return scen


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from run_all import main

    raise SystemExit(main(["--only", "filters", *sys.argv[1:]]))
