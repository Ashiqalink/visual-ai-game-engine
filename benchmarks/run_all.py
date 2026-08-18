"""
run_all.py — Entry point for the stabilizer / filter test benches.

    python benchmarks/run_all.py                    # both benches, terminal output
    python benchmarks/run_all.py --only stabilizer  # one bench
    python benchmarks/run_all.py --html             # write + open an HTML report
    python benchmarks/run_all.py --json out.json    # machine-readable results

Exit code is 0 when every check passed and 1 otherwise, so this can be wired
into CI or a pre-commit hook unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    bold, bootstrap, cyan, dim, green, header, print_scenarios, print_tables, red,
)

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_HTML = BENCH_DIR / "bench_report.html"


def _json_safe(obj):
    """Unwrap numpy scalars and drop non-finite floats so the JSON stays valid."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()          # numpy float64 / bool_ -> Python scalar
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ToF stabilizer and landmark filter test benches.")
    parser.add_argument("--only", choices=["stabilizer", "filters", "resolution"],
                        help="run just one bench (default: all)")
    parser.add_argument("--html", nargs="?", const=str(DEFAULT_HTML), metavar="PATH",
                        help="write an HTML report (default benchmarks/bench_report.html)")
    parser.add_argument("--json", metavar="PATH", help="write results as JSON")
    parser.add_argument("--no-open", action="store_true",
                        help="do not open the HTML report in a browser")
    parser.add_argument("--quiet", action="store_true",
                        help="summary lines only — no per-scenario detail")
    args = parser.parse_args(argv)

    modules = bootstrap()

    benches = []
    if args.only in (None, "stabilizer"):
        import bench_stabilizer
        benches.append(bench_stabilizer)
    if args.only in (None, "filters"):
        import bench_filters
        benches.append(bench_filters)
    if args.only == "resolution" or (
        args.only is None and (BENCH_DIR / "fixtures" / "hand_motion.mp4").exists()
    ):
        import bench_resolution
        benches.append(bench_resolution)

    print(header("Visual AI — stabilizer & filter test bench",
                 "synthetic streams with known ground truth · seeded, so runs are comparable"))
    for name, path in modules.items():
        print(f"  {dim(name.ljust(26))} {dim(path)}")

    results = []
    for module in benches:
        result = module.run()
        results.append(result)

        ok, total = result.counts
        status = green("PASS") if result.passed else red(f"FAIL ({total - ok})")
        print(header(f"{result.name}  [{status}]", result.subtitle))
        if not args.quiet:
            print_scenarios(result)
            print_tables(result)

    total_ok = sum(r.counts[0] for r in results)
    total_all = sum(r.counts[1] for r in results)
    every_pass = total_ok == total_all

    print(header("Summary"))
    for result in results:
        ok, total = result.counts
        mark = green("PASS") if result.passed else red("FAIL")
        failed = [s.name for s in result.scenarios if not s.passed]
        line = f"  {mark}  {result.name.ljust(30)} {ok}/{total} checks"
        if failed:
            line += dim("   failing: " + ", ".join(failed))
        print(line)
    verdict = green(f"{total_ok}/{total_all} checks passed") if every_pass else \
        red(f"{total_all - total_ok} of {total_all} checks failed")
    print(f"\n  {bold(verdict)}")

    payload = [r.as_dict() for r in results]

    if args.html:
        import report

        path = Path(args.html).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.build_html(payload, modules), encoding="utf-8")
        print(f"  {cyan('HTML report')} {path}")
        if not args.no_open:
            webbrowser.open(path.as_uri())

    if args.json:
        path = Path(args.json).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        print(f"  {cyan('JSON results')} {path}")

    print()
    return 0 if every_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
