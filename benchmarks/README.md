# Test benches — ToF stabilizer & landmark filters

Runnable benches that score the stabilizer and the smoothing filters on
synthetic streams **where the true motion is known exactly**, so the output is
measured accuracy rather than "looks smoother to me".

```bash
python benchmarks/run_all.py                     # both benches, tables in the terminal
python benchmarks/run_all.py --html              # also write + open an HTML report
python benchmarks/run_all.py --only stabilizer   # one bench
python benchmarks/run_all.py --only filters --quiet
python benchmarks/run_all.py --json results.json # machine-readable, for CI
```

Exit code is `0` when every check passes, `1` otherwise — safe to wire into CI
or a pre-commit hook as-is. Each bench also runs standalone:

```bash
python benchmarks/bench_stabilizer.py
python benchmarks/bench_filters.py --html
```

## What you get

Per scenario: a sparkline of every trace, then a table of measured value /
requirement / PASS-FAIL. Then two comparison tables per bench — all filters on
all scenarios, a `gate_k` tuning sweep for the stabilizer, and per-sample cost.
`--html` renders the same content with real line charts (raw vs true vs
stabilized), theme-aware and dependency-free.

## The three numbers, and why all three are always shown

Any smoother wins one metric alone — a hard freeze has zero jitter, a
passthrough has zero lag — so a single number can always be gamed:

| metric | means | better |
| --- | --- | --- |
| `jitter` | detrended high-frequency wobble that survived, same definition as `JitterAnalyzer` | lower |
| `rmse` | distance from the true, noise-free motion | lower |
| `lag_ms` | tracking delay, from the best sub-frame alignment against the truth | lower |

The stabilizer's equivalent pairing is **wobble removed %** against **reach
preserved %**: it must kill lid shake without flattening a punch.

## Layout

| file | role |
| --- | --- |
| `run_all.py` | CLI entry point, summary, HTML/JSON output |
| `bench_stabilizer.py` | 9 scenarios for `ToFStabilizer`, plus a `gate_k` sweep |
| `bench_filters.py` | 7 scenarios × 5 filter tunings, plus the One-Euro-vs-EMA verdict |
| `signals.py` | seeded synthetic streams, each returning `(truth, raw)` |
| `metrics.py` | jitter / rmse / lag / overshoot / settling / spike-leak / throughput |
| `harness.py` | import bootstrap, fake clock, tables, sparklines, check records |
| `report.py` | standalone HTML report with inline-SVG charts |

## Two things worth knowing

**Which copy of the code is measured.** The repo carries two copies of the
package — `src/visual_ai` (current) and a stale top-level `visual_ai` that has
no `OneEuroFilter` and no `jitter_analyzer`. A plain `import visual_ai` from
the project root picks the stale one. `harness.bootstrap()` forces `src` to the
front of `sys.path` and prints the resolved file path of every module at the
top of each run, so you can always see what was actually measured.

**Calibration runs on a fake clock.** `ToFStabilizer` ends its window against
`time.time()`, so feeding 3 seconds of samples in 2 ms would never leave the
sampling state. `harness.fake_clock()` swaps the module's `time` reference for
a hand-cranked one — calibration completes in microseconds and results are
reproducible frame-for-frame.

## Thresholds

Every check is calibrated to pass on today's code with headroom, so a failure
means something moved. Verified by injecting regressions: dropping One-Euro's
`beta` (its speed adaptation) fails 7 filter checks; widening `gate_k` 12× fails
5 stabilizer checks.

Two thresholds are deliberately loose, with the reasoning in the source:

* `vibration suppression / worst-case swing` — Gaussian shake has long tails, so
  a few samples always land outside the gate. The graded number is the p1–p99
  band; the absolute worst swing is capped only at 45% of the raw swing.
* `tracker glitches / error vs truth` — One-Euro reads a single-frame glitch as
  fast motion and opens its cutoff, so it leaks more of a spike than a fixed
  EMA. The real guard there is the worst-excursion cap.

## Adding a scenario

Add a generator to `signals.py` returning `(truth, raw)`, then one entry in
`bench_filters._scenarios()` (signal + requirements) or one
`scenario_*()` function in `bench_stabilizer.py`. Charts, tables, HTML, JSON,
and the exit code all follow automatically.
