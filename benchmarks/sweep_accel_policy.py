"""
sweep_accel_policy.py — run ``bench_accel_policy.py`` across scenarios and
repeats, and report whether its verdict is stable or just one lucky machine
state.

One run of that bench answers "does the auto preset hold *right now, on this
clip, in this configuration*". That is a weaker statement than it looks:

  * the tier and hand count are arguments with real effects — ``lite`` is a
    different network and two hands is a second landmark inference per frame;
  * the process is compiled fresh each run, and a device's first compile is not
    its steady state;
  * the machine drifts. Earlier measurements here moved by a factor of two
    between runs with nothing changed.

So this sweeps the matrix, repeats each cell in a *separate process* (the one
source of variance the bench's own interleaved rounds cannot reach, since they
share a compile), and prints per-cell verdicts next to the spread that produced
them. A policy that holds in every cell is a policy. A policy that holds in six
cells out of eight is a coin toss with a docstring.

Each cell also runs both device orders, because the order backends are measured
in is itself a variable — see the ``--order`` note in the bench.

**A cell's failure is a claim about the sweep until it is re-run alone.** The
first full run of this matrix failed the latency check on ``full/2h`` in three
of four repeats — 122/126/126/108% of the iGPU's tracking time, both device
orders — which reads as replication and is not. Four later measurements of that
same cell, each in a fresh session, gave 104/106/114/106%, and a follow-up
varying sustained load moved the gap the wrong way (14% at one pass, 6% at
four). Those four sweep cells shared a thermal state and a position twenty
minutes into a thirty-nine minute run; their agreement was correlated error,
not evidence.

The ``repeat noise`` column does not catch this and is not meant to: it is the
spread *within* one identical cell, and a device sitting at 8% there can still
move 20% between sessions. So read a failing cell as a question, re-run that one
scenario cold, and only then call it a property of the workload::

    python benchmarks/bench_accel_policy.py --tier full --hands 2 --rounds 3

Run it::

    "D:/visual/.venv/Scripts/python.exe" benchmarks/sweep_accel_policy.py
    "D:/visual/.venv/Scripts/python.exe" benchmarks/sweep_accel_policy.py --repeats 3

Exit code 1 means at least one cell disagreed with ``accel.DEFAULT_PRESET``,
or the cells disagreed with each other — both are reasons not to trust a single
run of the bench.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import bold, dim, green, header, red, table  # noqa: E402

BENCH = Path(__file__).resolve().parent / "bench_accel_policy.py"

#: tier x hands. ``full``/1 is what ``VisionPipeline`` defaults to; ``lite``/2
#: is what a game asking for two hands on a slower machine gets. Both matter,
#: and they are different enough networks that a policy true for one is not
#: automatically true for the other.
SCENARIOS = [("full", 1), ("full", 2), ("lite", 1), ("lite", 2)]
ORDERS = ("forward", "reverse")


def run_cell(tier: str, hands: int, order: str, rounds: int,
             passes: int) -> dict | None:
    """One bench process. Returns its parsed JSON, or None if it would not run."""
    proc = subprocess.run(
        [sys.executable, str(BENCH), "--json", "--tier", tier,
         "--hands", str(hands), "--order", order,
         "--rounds", str(rounds), "--passes", str(passes)],
        capture_output=True, text=True)
    # The bench prints exactly one JSON object on stdout under --json, but
    # MediaPipe and OpenVINO both write banners to stderr and, on some builds,
    # to stdout as well. Take the last line that parses rather than the whole
    # stream, and say so loudly if none does — a silently skipped cell would
    # make the sweep look unanimous by being empty.
    payload = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
    if payload is None:
        print(red(f"  cell tier={tier} hands={hands} order={order} produced no "
                  f"JSON (exit {proc.returncode})"))
        if proc.stderr.strip():
            print(dim("    " + proc.stderr.strip().splitlines()[-1]))
        return None
    payload["exit"] = proc.returncode
    return payload


def pct_of(cell: dict, device: str, against: str, key: str) -> float:
    """`device`'s `key` as a percentage of `against`'s, inside one cell.

    Both devices come from the same process and the same interleaved rounds, so
    this ratio cancels whatever the machine was doing at the time — which is the
    whole reason to compare in percent rather than in milliseconds.
    """
    idle = cell.get("idle", {})
    top = idle.get(device, {}).get(key, float("nan"))
    bottom = idle.get(against, {}).get(key, float("nan"))
    if not bottom or bottom != bottom or top != top:
        return float("nan")
    return 100.0 * top / bottom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--repeats", type=int, default=2,
                        help="separate processes per scenario and order")
    parser.add_argument("--rounds", type=int, default=3,
                        help="interleaved rounds inside each process")
    parser.add_argument("--passes", type=int, default=2,
                        help="clip passes per measurement")
    parser.add_argument("--scenarios", default="",
                        help="comma-separated tier:hands, e.g. full:1,lite:2")
    # Half an hour of machine time produces these cells. Keeping the raw ones
    # means a question about how they were summarised is answered by re-reading
    # them rather than by re-measuring a machine that has since moved.
    parser.add_argument("--out", type=Path, default=None, metavar="PATH",
                        help="write every raw cell to this JSON file")
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = []
        for token in args.scenarios.split(","):
            tier, _, hands = token.partition(":")
            scenarios.append((tier.strip(), int(hands or 1)))

    cells = len(scenarios) * len(ORDERS) * args.repeats
    print(header("accel auto-preset policy — sweep",
                 f"{len(scenarios)} scenarios x {len(ORDERS)} orders x "
                 f"{args.repeats} repeats = {cells} processes, "
                 f"{args.rounds} rounds x {args.passes} passes each"))

    started = time.perf_counter()
    results: list[dict] = []
    for tier, hands in scenarios:
        for order in ORDERS:
            for repeat in range(args.repeats):
                print(dim(f"  running tier={tier} hands={hands} order={order} "
                          f"repeat={repeat + 1}/{args.repeats} ..."))
                cell = run_cell(tier, hands, order, args.rounds, args.passes)
                if cell is not None:
                    results.append(cell)
    elapsed = time.perf_counter() - started

    if not results:
        print(red("\nno cell produced a result — nothing was measured"))
        return 1

    # ── Per-cell verdicts ─────────────────────────────────────────────────────
    rows = []
    for cell in results:
        idle = cell.get("idle", {})
        prefers = cell.get("prefers") or []
        chosen = prefers[0] if prefers else "-"
        second = prefers[1] if len(prefers) > 1 else "-"
        marks = "".join(
            (green("P") if chk["passed"] else red("F")) for chk in cell["checks"])
        rows.append([
            f"{cell['tier']}/{cell['hands']}h", cell["order"],
            f"{idle.get(chosen, {}).get('tracking', float('nan')):.2f}",
            f"{idle.get(second, {}).get('tracking', float('nan')):.2f}",
            f"{pct_of(cell, chosen, second, 'tracking'):.0f}%",
            f"{idle.get(chosen, {}).get('cores', float('nan')):.2f}",
            f"{pct_of(cell, chosen, second, 'cores'):.0f}%",
            marks or dim("skipped"),
        ])
    print("\n" + bold("per cell") + dim("  (marks are latency/cpu/headroom; "
                                        "% columns are the chosen device against "
                                        "the runner-up)"))
    print(table(["scenario", "order", "NPU ms", "GPU ms", "ms %", "NPU cores",
                 "cores %", "checks"], rows, aligns="llrrrrrl"))

    # ── Stability ─────────────────────────────────────────────────────────────
    def spread_of(device: str, key: str) -> tuple[float, float, float]:
        values = [c["idle"][device][key] for c in results
                  if device in c.get("idle", {})]
        if not values:
            return (float("nan"),) * 3
        return min(values), statistics.median(values), max(values)

    # Range across every cell pools three different things: repeat-to-repeat
    # noise, the effect of the scenario, and the effect of device order. Pooled,
    # it reads as one huge error bar and says a tight measurement is sloppy —
    # `lite/1h` and `full/2h` are not the same work, and their difference is a
    # result, not a spread. So each is reported on its own.
    def cell_key(cell: dict) -> tuple:
        return (cell["tier"], cell["hands"], cell["order"])

    def repeat_noise(device: str) -> float:
        """Median relative spread between repeats of one identical cell."""
        groups: dict[tuple, list[float]] = {}
        for cell in results:
            if device in cell.get("idle", {}):
                groups.setdefault(cell_key(cell), []).append(
                    cell["idle"][device]["tracking"])
        rel = [(max(v) - min(v)) / statistics.median(v)
               for v in groups.values() if len(v) > 1 and statistics.median(v)]
        return statistics.median(rel) if rel else float("nan")

    def order_effect(device: str) -> float:
        """Median ms the device moved when the measurement order was flipped."""
        pairs: dict[tuple, dict[str, list[float]]] = {}
        for cell in results:
            if device in cell.get("idle", {}):
                slot = pairs.setdefault((cell["tier"], cell["hands"]), {})
                slot.setdefault(cell["order"], []).append(
                    cell["idle"][device]["tracking"])
        deltas = [statistics.median(v["reverse"]) - statistics.median(v["forward"])
                  for v in pairs.values() if "forward" in v and "reverse" in v]
        return statistics.median(deltas) if deltas else float("nan")

    rows = []
    devices = sorted({d for c in results for d in c.get("idle", {})})
    for device in devices:
        lo, mid, hi = spread_of(device, "tracking")
        clo, cmid, chi = spread_of(device, "cores")
        rows.append([device, f"{mid:.2f}", f"{lo:.2f}–{hi:.2f}",
                     f"{repeat_noise(device):.0%}", f"{order_effect(device):+.2f}",
                     f"{cmid:.2f}", f"{clo:.2f}–{chi:.2f}"])
    print("\n" + bold("across every cell")
          + dim("  (repeat noise is within one identical cell; order effect is "
                "reverse minus forward — both are the bench, not the device)"))
    print(table(["backend", "median ms", "range (all cells)", "repeat noise",
                 "order effect", "median cores", "cores range"],
                rows, aligns="lrrrrrr"))

    # ── The cross-machine table ───────────────────────────────────────────────
    # Milliseconds above are about this machine. These ratios are the part that
    # should reappear on another one: each backend against the MediaPipe CPU
    # graph measured in the same process, so the comparison cancels the clock.
    # A range spanning 100% means the ordering flipped somewhere in the matrix.
    def rel_spread(device: str, key: str) -> tuple[float, float, float]:
        values = [c["relative"][device][key] for c in results
                  if device in c.get("relative", {})
                  and c["relative"][device][key] == c["relative"][device][key]]
        if not values:
            return (float("nan"),) * 3
        return min(values), statistics.median(values), max(values)

    if any("relative" in c for c in results):
        base = next(c.get("relative_base", "?") for c in results
                    if "relative" in c)
        rows = []
        for device in devices:
            lo, mid, hi = rel_spread(device, "tracking_pct")
            clo, cmid, chi = rel_spread(device, "cores_pct")
            rows.append([device, f"{mid:.0f}%", f"{lo:.0f}–{hi:.0f}%",
                         f"{cmid:.0f}%", f"{clo:.0f}–{chi:.0f}%"])
        print("\n" + bold(f"relative to {base}")
              + dim("  (100% = the baseline. Compare these across machines; the "
                    "millisecond tables above will not match)"))
        print(table(["backend", "tracking", "range", "cores", "range"],
                    rows, aligns="lrrrr"))

        spec = next((c.get("machine") for c in results if c.get("machine")), None)
        if spec:
            print("\n" + bold("measured on")
                  + dim("  (a percentage is only comparable if you know what it "
                        "was measured on)"))
            print(f"  cpu        {spec.get('cpu', '?')}")
            print(f"  cores      {spec.get('cores_physical', '?')} physical / "
                  f"{spec.get('cores_logical', '?')} logical")
            print("  devices    " + ", ".join(
                f"{k} ({v})" for k, v in (spec.get("devices") or {}).items()))
            print(f"  platform   {spec.get('platform', '?')}, "
                  f"python {spec.get('python', '?')}")

    # ── Did the answer ever change? ───────────────────────────────────────────
    failed = [c for c in results if c.get("failures")]
    # Which device actually won on tracking latency, per cell. The policy check
    # allows the preferred device to be a little behind; this is the stricter
    # question of whether it was ever behind at all, and it is the one that
    # would justify reopening the preset.
    winners = {}
    for cell in results:
        idle = cell.get("idle", {})
        candidates = {d: s["tracking"] for d, s in idle.items()
                      if d in (cell.get("devices") or [])}
        if candidates:
            winners.setdefault(min(candidates, key=candidates.get), 0)
            winners[min(candidates, key=candidates.get)] += 1

    print("\n" + bold("verdict"))
    print(f"  cells run          {len(results)} in {elapsed:.0f}s")
    print(f"  cells that failed  {len(failed)}")
    print("  fastest device     " + ", ".join(
        f"{device} in {count}/{len(results)} cells"
        for device, count in sorted(winners.items(), key=lambda kv: -kv[1])))

    unstable = len(winners) > 1
    if failed:
        print("\n" + red("FAIL") + " — the preset did not hold in every cell")
        for cell in failed:
            for failure in cell["failures"]:
                print(f"  tier={cell['tier']} hands={cell['hands']} "
                      f"order={cell['order']}: {failure}")
        return 1
    if unstable:
        print("\n" + dim("the preset held in every cell, but the fastest device "
                         "was not the same one every time — the margin is inside "
                         "the noise, and a single run of the bench should not be "
                         "quoted as if it settled the ordering."))
        return 0
    print("\n" + green("the preset held in every cell, and the same device won "
                       "every time"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
