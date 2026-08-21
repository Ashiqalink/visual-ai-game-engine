"""
bench_stabilizer.py — Test bench for ``visual_ai.tof_stabilizer.ToFStabilizer``.

Drives the stabilizer through synthetic ToF depth streams where the true depth
is known exactly, so each run answers the two questions that matter and are in
direct tension:

    * how much lid-shake vibration was removed?
    * how much of the real punch/push signal survived?

A stabilizer can trivially win either one alone — reporting a constant kills
all vibration, doing nothing preserves all signal — so both are always scored
together, alongside the failure paths (blind sensor, too few samples) where
the correct behaviour is to refuse to activate.

Run it::

    python benchmarks/bench_stabilizer.py
    python benchmarks/bench_stabilizer.py --html report.html

All timing runs on a fake clock (see ``harness.fake_clock``), so a 3-second
calibration window completes in microseconds and results are reproducible.
"""

from __future__ import annotations

import metrics
import numpy as np
import signals
from harness import BenchResult, Scenario, bootstrap, fake_clock, quiet

_MODULES = bootstrap()

import visual_ai.tof_stabilizer as tof_mod  # noqa: E402
from visual_ai.tof_stabilizer import ToFStabilizer  # noqa: E402

FPS = signals.FPS
DT = signals.DT
CALIB_SECONDS = 3.0
CALIB_FRAMES = int(CALIB_SECONDS * FPS)

MM = 1000.0   # metres -> millimetres, the unit these numbers are readable in


# ── Driving helpers ───────────────────────────────────────────────────────────

def calibrate(stab: ToFStabilizer, samples, duration: float = CALIB_SECONDS) -> ToFStabilizer:
    """
    Run one full calibration window on the fake clock.

    Feeds ``samples`` one per frame, then advances past the end of the window
    and ticks once, exactly as ``VisionPipeline`` does from its capture loop.
    """
    with fake_clock(tof_mod) as clock, quiet():
        stab.begin(duration)
        for z in samples:
            if stab.state != ToFStabilizer.STATE_SAMPLING:
                break
            stab.feed(float(z))
            clock.advance(DT)
        clock.advance(duration)
        stab.tick()
    return stab


def run_live(stab: ToFStabilizer, samples) -> np.ndarray:
    """Push a live depth stream through ``correct()`` and collect the output."""
    with quiet():
        return np.asarray([stab.correct(float(z)) for z in samples], dtype=float)


def split(truth: np.ndarray, raw: np.ndarray):
    """Cut a stream into its calibration window and the live remainder."""
    return (raw[:CALIB_FRAMES],
            truth[CALIB_FRAMES:], raw[CALIB_FRAMES:])


def _calibrated(truth, raw, **kwargs):
    """Calibrate on the head of a stream; return ``(stab, truth_tail, raw_tail)``."""
    calib, truth_tail, raw_tail = split(truth, raw)
    stab = calibrate(ToFStabilizer(**kwargs), calib)
    return stab, truth_tail, raw_tail


# ── Scenarios ─────────────────────────────────────────────────────────────────

def scenario_calibration() -> Scenario:
    """Does calibration measure the resting depth and the vibration amplitude?"""
    scen = Scenario(
        name="calibration accuracy",
        description="3 s window at 0.45 m with 4 mm RMS lid shake. Measures baseline and noise.",
        y_label="mm",
    )
    truth, raw = signals.tof_rest(n=400, depth=0.45, shake_m=0.004)
    calib = raw[:CALIB_FRAMES]
    stab = calibrate(ToFStabilizer(), calib)

    true_sigma = float(np.std(calib))
    scen.metrics = {
        "state": stab.state,
        "samples": CALIB_FRAMES,
        "baseline_m": stab.z_baseline,
        "baseline_error_mm": abs(stab.z_baseline - 0.45) * MM,
        "measured_noise_mm": stab.z_noise_amplitude * MM,
        "true_noise_mm": true_sigma * MM,
        "gate_mm": stab.noise_gate * MM,
    }

    scen.check("activated", 1.0 if stab.is_calibrated else 0.0, "",
               min_value=1.0, detail=f"state={stab.state}")
    scen.check("baseline error", abs(stab.z_baseline - 0.45) * MM, "mm",
               max_value=2.0, detail="true resting depth 450.0 mm")
    scen.check("noise estimate error", abs(stab.z_noise_amplitude - true_sigma) * MM,
               "mm", max_value=1.0,
                detail=f"measured {stab.z_noise_amplitude * MM:.2f} "
                       f"vs true {true_sigma * MM:.2f} mm")
    scen.check("gate width", stab.noise_gate * MM, "mm", min_value=2.0, max_value=20.0,
               detail=f"gate_k={stab.gate_k} x noise")
    scen.check("progress complete", stab.progress, "", min_value=1.0)

    scen.traces = {"calibration samples": list(calib * MM)}
    return scen


def scenario_rest_suppression() -> Scenario:
    """The core job: hold the reported depth steady while the lid vibrates."""
    scen = Scenario(
        name="vibration suppression",
        description="Hand still at 0.45 m, lid shaking at 4 mm RMS. Output should barely move.",
        y_label="mm",
    )
    stab, truth, raw = _calibrated(*signals.tof_rest(n=500, depth=0.45, shake_m=0.004))
    out = run_live(stab, raw)

    raw_std, out_std = float(np.std(raw)) * MM, float(np.std(out)) * MM
    raw_pp = float(np.ptp(raw)) * MM
    out_pp = float(np.ptp(out)) * MM
    # Gaussian shake has long tails: a handful of samples always land outside
    # the gate and leak their excess. The 1st-99th percentile band is what the
    # player actually perceives as steadiness, so that is what is graded;
    # the absolute worst-case swing is graded separately and loosely.
    out_band = float(np.percentile(out, 99) - np.percentile(out, 1)) * MM
    raw_band = float(np.percentile(raw, 99) - np.percentile(raw, 1)) * MM

    scen.metrics = {
        "raw_std_mm": raw_std,
        "stabilized_std_mm": out_std,
        "raw_peak_to_peak_mm": raw_pp,
        "stabilized_peak_to_peak_mm": out_pp,
        "raw_p1_p99_band_mm": raw_band,
        "stabilized_p1_p99_band_mm": out_band,
        "wobble_removed_pct": metrics.reduction_pct(out_std, raw_std),
        "mean_depth_error_mm": abs(float(np.mean(out)) - 0.45) * MM,
    }

    scen.check("wobble removed", metrics.reduction_pct(out_std, raw_std), "%",
               min_value=85.0, detail=f"{raw_std:.2f} mm -> {out_std:.2f} mm RMS")
    scen.check("residual band (p1-p99)", out_band, "mm", max_value=3.0,
               detail=f"raw band {raw_band:.1f} mm")
    scen.check("worst-case swing", out_pp, "mm", max_value=0.45 * raw_pp,
               detail=f"raw swings {raw_pp:.1f} mm peak-to-peak")
    scen.check("depth not shifted", abs(float(np.mean(out)) - 0.45) * MM, "mm",
               max_value=2.0, detail="absolute depth must be preserved")

    scen.traces = {"raw depth": list(raw * MM), "stabilized": list(out * MM)}
    return scen


def scenario_punch_fidelity() -> Scenario:
    """Real movement must survive — this is what the old implementation destroyed."""
    scen = Scenario(
        name="punch fidelity",
        description="Three 220 mm punches toward the camera through 4 mm of lid shake.",
        y_label="mm",
    )
    stab, truth, raw = _calibrated(*signals.tof_punch(n=500, depth=0.45, reach=0.22))
    out = run_live(stab, raw)

    true_reach = (0.45 - float(np.min(truth))) * MM
    got_reach = (0.45 - float(np.min(out))) * MM
    preserved = got_reach / true_reach * 100.0 if true_reach > 1e-9 else 0.0
    corr = float(np.corrcoef(out, truth)[0, 1])

    scen.metrics = {
        "true_reach_mm": true_reach,
        "measured_reach_mm": got_reach,
        "reach_preserved_pct": preserved,
        "rmse_mm": metrics.rmse(out, truth) * MM,
        "correlation": corr,
        "lag_ms": metrics.lag_ms(out, truth, DT),
        "min_depth_mm": float(np.min(out)) * MM,
    }

    scen.check("reach preserved", preserved, "%", min_value=85.0,
               detail=f"{got_reach:.0f} of {true_reach:.0f} mm")
    scen.check("shape correlation", corr, "", min_value=0.98)
    scen.check("error vs true depth", metrics.rmse(out, truth) * MM, "mm", max_value=15.0)
    scen.check("lag", metrics.lag_ms(out, truth, DT), "ms", max_value=50.0)
    scen.check("no collapse to clamp", float(np.min(out)) * MM,
               min_value=(ToFStabilizer.MIN_DEPTH_M * MM) + 50.0, unit="mm",
               detail="a reading pinned at the 50 mm floor means the signal was destroyed")

    scen.traces = {
        "raw depth": list(raw * MM),
        "true depth": list(truth * MM),
        "stabilized": list(out * MM),
    }
    return scen


def scenario_drift_tracking() -> Scenario:
    """Slow genuine movement must be tracked, not gated away as vibration."""
    scen = Scenario(
        name="slow lean tracking",
        description="Player leans back 60 mm over 20 s while the lid still shakes.",
        y_label="mm",
    )
    stab, truth, raw = _calibrated(*signals.tof_drift(n=700, depth=0.45, drift_m=0.06))
    out = run_live(stab, raw)

    tail = slice(-30, None)
    final_error = abs(float(np.mean(out[tail])) - float(np.mean(truth[tail]))) * MM
    tracked = ((float(np.mean(out[tail])) - float(np.mean(out[:30])))
               / (float(np.mean(truth[tail])) - float(np.mean(truth[:30]))) * 100.0)

    scen.metrics = {
        "final_error_mm": final_error,
        "drift_tracked_pct": tracked,
        "rmse_mm": metrics.rmse(out, truth) * MM,
        "output_jitter_mm": metrics.jitter_rms(out) * MM,
        "raw_jitter_mm": metrics.jitter_rms(raw) * MM,
    }

    scen.check("drift tracked", tracked, "%", min_value=90.0,
               detail="gating slow drift away would freeze the player's depth")
    scen.check("final depth error", final_error, "mm", max_value=8.0)
    scen.check("still smoothing", metrics.reduction_pct(metrics.jitter_rms(out),
                                                        metrics.jitter_rms(raw)),
               "%", min_value=60.0, detail="drift tracking must not disable the gate")

    scen.traces = {
        "raw depth": list(raw * MM),
        "true depth": list(truth * MM),
        "stabilized": list(out * MM),
    }
    return scen


def scenario_blind_sensor() -> Scenario:
    """A disabled ToF reports 0.0 forever — calibrating on that is meaningless."""
    scen = Scenario(
        name="blind sensor refused",
        description="Every reading is 0.0 (ToF off). Calibration must fail loudly, not activate.",
    )
    _, raw = signals.tof_blind(n=400)
    stab = calibrate(ToFStabilizer(), raw[:CALIB_FRAMES])

    passthrough = run_live(stab, [0.45, 0.30, 0.0, float("nan")])
    clean = [v for v in passthrough[:2]]

    scen.metrics = {
        "state": stab.state,
        "last_error": stab.last_error or "",
        "valid_samples": stab.sample_count,
    }

    scen.check("stayed inactive", 0.0 if stab.is_calibrated else 1.0, "", min_value=1.0,
               detail=f"state={stab.state}")
    scen.check("reported a reason", 1.0 if stab.last_error else 0.0, "", min_value=1.0,
               detail=stab.last_error or "no last_error set")
    scen.check("passthrough unchanged", max(abs(clean[0] - 0.45), abs(clean[1] - 0.30)),
               "m", max_value=1e-9, detail="an inactive stabilizer must not alter readings")
    return scen


def scenario_too_few_samples() -> Scenario:
    """Intermittent sensor: not enough valid readings to justify a baseline."""
    scen = Scenario(
        name="sparse sensor refused",
        description=f"One valid reading every 25 frames — under the "
                    f"{ToFStabilizer.MIN_SAMPLES}-sample minimum.",
    )
    _, raw = signals.tof_intermittent(n=400, valid_every=25)
    stab = calibrate(ToFStabilizer(), raw[:CALIB_FRAMES])

    scen.metrics = {"state": stab.state, "last_error": stab.last_error or ""}
    scen.check("stayed inactive", 0.0 if stab.is_calibrated else 1.0, "", min_value=1.0,
               detail=f"state={stab.state}")
    scen.check("baseline cleared", abs(stab.z_baseline), "m", max_value=1e-9)
    return scen


def scenario_gate_floor() -> Scenario:
    """On a rock-steady rig the gate must not shrink to nothing."""
    scen = Scenario(
        name="gate floor on a steady rig",
        description="0.1 mm of noise. The gate should clamp to min_gate_m, not collapse.",
        y_label="mm",
    )
    stab, truth, raw = _calibrated(*signals.tof_quiet(n=400))
    out = run_live(stab, raw)

    scen.metrics = {
        "measured_noise_mm": stab.z_noise_amplitude * MM,
        "gate_mm": stab.noise_gate * MM,
        "floor_mm": stab.min_gate_m * MM,
        "output_std_mm": float(np.std(out)) * MM,
    }
    scen.check("gate at floor", stab.noise_gate * MM, "mm",
               min_value=stab.min_gate_m * MM, max_value=stab.min_gate_m * MM + 0.01,
               detail=f"min_gate_m = {stab.min_gate_m * MM:.1f} mm")
    scen.check("output steady", float(np.std(out)) * MM, "mm", max_value=0.5)
    scen.traces = {"raw depth": list(raw * MM), "stabilized": list(out * MM)}
    return scen


def scenario_invalid_readings() -> Scenario:
    """Zeros and NaNs mid-stream must pass through without poisoning the baseline."""
    scen = Scenario(
        name="invalid readings survive",
        description="An active stabilizer fed 0.0 / NaN must pass them "
                    "through and keep its baseline.",
    )
    stab, truth, raw = _calibrated(*signals.tof_rest(n=300, depth=0.45))
    baseline_before = stab.z_baseline

    with quiet():
        zero_out = stab.correct(0.0)
        nan_out = stab.correct(float("nan"))
        after = stab.correct(0.45)

    scen.metrics = {
        "zero_in_zero_out": zero_out,
        "nan_passed_through": float(np.isnan(nan_out)),
        "baseline_drift_mm": abs(stab.z_baseline - baseline_before) * MM,
        "recovery_error_mm": abs(after - 0.45) * MM,
    }
    scen.check("zero passed through", abs(zero_out), "m", max_value=1e-9)
    scen.check("nan passed through", 1.0 if np.isnan(nan_out) else 0.0, "", min_value=1.0)
    scen.check("baseline untouched", abs(stab.z_baseline - baseline_before) * MM, "mm",
               max_value=1.0, detail="invalid frames must not move the resting depth")
    scen.check("recovers immediately", abs(after - 0.45) * MM, "mm", max_value=3.0)
    return scen


def scenario_lifecycle() -> Scenario:
    """begin / cancel / disable must leave the stabilizer in the documented state."""
    scen = Scenario(
        name="lifecycle controls",
        description="cancel() aborts a run in progress; disable() clears an active calibration.",
    )
    stab, truth, raw = _calibrated(*signals.tof_rest(n=300))
    active_ok = stab.is_calibrated

    with quiet():
        stab.disable()
    disabled_ok = (stab.state == ToFStabilizer.STATE_INACTIVE
                   and stab.z_baseline == 0.0
                   and abs(stab.correct(0.37) - 0.37) < 1e-9)

    cancel_stab = ToFStabilizer()
    with fake_clock(tof_mod) as clock, quiet():
        cancel_stab.begin(3.0)
        for z in raw[:20]:
            cancel_stab.feed(float(z))
            clock.advance(DT)
        mid_state = cancel_stab.state
        progress = cancel_stab.progress
        cancel_stab.cancel()
        clock.advance(10.0)
        cancel_stab.tick()
    cancel_ok = (mid_state == ToFStabilizer.STATE_SAMPLING
                 and cancel_stab.state == ToFStabilizer.STATE_INACTIVE)

    scen.metrics = {
        "progress_at_cancel": progress,
        "state_after_cancel": cancel_stab.state,
    }
    scen.check("calibration activates", 1.0 if active_ok else 0.0, "", min_value=1.0)
    scen.check("disable() clears state", 1.0 if disabled_ok else 0.0, "", min_value=1.0,
               detail="state inactive, baseline 0, readings untouched")
    scen.check("cancel() aborts", 1.0 if cancel_ok else 0.0, "", min_value=1.0,
               detail=f"progress was {progress:.0%} when cancelled")
    scen.check("progress mid-run sane", progress, "", min_value=0.05, max_value=0.95)
    return scen


# ── Bench ─────────────────────────────────────────────────────────────────────

def run() -> BenchResult:
    result = BenchResult(
        name="ToF lid-shake stabilizer",
        subtitle=f"visual_ai.tof_stabilizer — {_MODULES['visual_ai.tof_stabilizer']}",
    )
    result.scenarios = [
        scenario_calibration(),
        scenario_rest_suppression(),
        scenario_punch_fidelity(),
        scenario_drift_tracking(),
        scenario_gate_floor(),
        scenario_invalid_readings(),
        scenario_blind_sensor(),
        scenario_too_few_samples(),
        scenario_lifecycle(),
    ]
    result.tables.append(_tuning_sweep())
    result.tables.append(_cost_table())
    return result


def _tuning_sweep() -> dict:
    """
    ``gate_k`` sets the whole trade-off, so sweep it: a wider gate kills more
    vibration and eats more of the punch. This table is how you pick a value.
    """
    rows = []
    for gate_k in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        stab, _, rest_raw = _calibrated(
            *signals.tof_rest(n=500, depth=0.45, shake_m=0.004), gate_k=gate_k)
        rest_out = run_live(stab, rest_raw)
        removed = metrics.reduction_pct(float(np.std(rest_out)),
                                        float(np.std(rest_raw)))

        punch_stab, punch_truth, punch_raw = _calibrated(
            *signals.tof_punch(n=500, depth=0.45, reach=0.22), gate_k=gate_k)
        punch_out = run_live(punch_stab, punch_raw)
        true_reach = 0.45 - float(np.min(punch_truth))
        preserved = (0.45 - float(np.min(punch_out))) / true_reach * 100.0

        rows.append([
            f"{gate_k:.1f}" + ("  ← shipped" if abs(gate_k - 2.5) < 1e-9 else ""),
            f"{stab.noise_gate * MM:.1f}",
            f"{removed:.1f}%",
            f"{preserved:.1f}%",
            f"{metrics.rmse(punch_out, punch_truth) * MM:.2f}",
        ])
    return {
        "title": "gate_k tuning sweep",
        "note": "wider gate = steadier at rest, more of the punch shaved off",
        "headers": ["gate_k", "gate mm", "wobble removed", "reach preserved", "punch rmse mm"],
        "rows": rows,
        "aligns": "lrrrr",
    }


def _cost_table() -> dict:
    """correct() runs on every frame of the capture loop — it must be free."""
    stab, _, raw = _calibrated(*signals.tof_rest(n=2200, depth=0.45))
    with quiet():
        us = metrics.throughput(lambda z: stab.correct(float(z)), raw)
    return {
        "title": "Cost per frame",
        "note": "correct() runs once per captured frame",
        "headers": ["call", "µs / frame", "% of a 33 ms frame"],
        "rows": [["ToFStabilizer.correct()", f"{us:,.2f}", f"{us / 33_333 * 100:,.4f}"]],
        "aligns": "lrr",
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from run_all import main

    raise SystemExit(main(["--only", "stabilizer", *sys.argv[1:]]))
