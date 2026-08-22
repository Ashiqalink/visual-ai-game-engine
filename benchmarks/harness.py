"""
harness.py — Shared plumbing for the stabilizer / filter test benches.

Handles the three things every bench needs and nothing else:

* **Import bootstrap** — the repo once carried a stale top-level ``visual_ai``
  duplicate that shadowed ``src/visual_ai`` (it is gone now — see CLAUDE.md).
  :func:`bootstrap` still forces ``src`` to the front of ``sys.path``, purges
  any wrong copy, and reports which file actually loaded, so a bench run can
  never quietly measure the wrong code if the hazard ever returns.

* **A fake clock** — ``DepthStabilizer`` ends its calibration window against
  ``time.time()``. A bench that feeds 3 seconds of samples in 2 ms would never
  leave the sampling state, so :func:`fake_clock` swaps the module's ``time``
  reference for a hand-cranked one.

* **Display** — ANSI colour (enabled on Windows consoles), box-drawn tables,
  sparklines, and PASS/FAIL check rows.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Import bootstrap ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

_BOOTSTRAPPED = False


def bootstrap() -> dict[str, str]:
    """
    Put ``src/`` at the head of ``sys.path`` and import the modules under test.

    Returns
    -------
    dict
        ``{module_name: file_path}`` for every module the benches exercise, so
        the report can state exactly which copy was measured.
    """
    global _BOOTSTRAPPED

    if not _BOOTSTRAPPED:
        src = str(SRC_DIR)
        while src in sys.path:
            sys.path.remove(src)
        sys.path.insert(0, src)

        # Drop anything already imported from the stale top-level copy. Done
        # once only: purging on a second call would hand the two benches
        # different module objects for the same code, so patching one (the
        # fake clock) would not be visible to the other.
        for name in [n for n in sys.modules
                     if n == "visual_ai" or n.startswith("visual_ai.")]:
            del sys.modules[name]
        _BOOTSTRAPPED = True

    import visual_ai.depth_stabilizer as depth_stabilizer
    import visual_ai.jitter_analyzer as jitter_analyzer
    import visual_ai.noise_filter as noise_filter

    return {
        "visual_ai.noise_filter": noise_filter.__file__,
        "visual_ai.depth_stabilizer": depth_stabilizer.__file__,
        "visual_ai.jitter_analyzer": jitter_analyzer.__file__,
    }


# ── Fake clock ────────────────────────────────────────────────────────────────

class FakeClock:
    """
    Hand-cranked stand-in for the ``time`` module.

    Only ``time()``, ``monotonic()`` and ``perf_counter()`` are provided —
    that is everything the modules under test call.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = float(start)

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now

    def time(self) -> float:
        return self.now

    monotonic = time
    perf_counter = time


@contextlib.contextmanager
def fake_clock(*modules, start: float = 1_000.0):
    """
    Replace ``<module>.time`` with a :class:`FakeClock` for the duration of the
    block, restoring the real module afterwards.

    Usage::

        with fake_clock(depth_stabilizer) as clock:
            stab.begin(3.0)
            for z in samples:
                stab.feed(z)
                clock.advance(1 / 30)
    """
    clock = FakeClock(start)
    originals = [(m, getattr(m, "time")) for m in modules]
    for module, _ in originals:
        module.time = clock
    try:
        yield clock
    finally:
        for module, original in originals:
            module.time = original


@contextlib.contextmanager
def quiet():
    """Swallow the modules' ``print()`` chatter so bench tables stay readable."""
    import io

    real = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = real


# ── Terminal display ──────────────────────────────────────────────────────────

def _enable_ansi() -> bool:
    """Turn on virtual-terminal processing on Windows; report colour support."""
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


_COLOR = _enable_ansi()

# Make block-drawing characters safe on a cp1252 console.
with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def c(text: str, code: str) -> str:
    """Wrap ``text`` in an ANSI colour when the terminal supports it."""
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(t: str) -> str:   return c(t, "1")
def dim(t: str) -> str:    return c(t, "2")
def green(t: str) -> str:  return c(t, "32")
def red(t: str) -> str:    return c(t, "31")
def cyan(t: str) -> str:   return c(t, "36")


def _visible_len(text: str) -> int:
    """Length of ``text`` ignoring ANSI escape sequences."""
    out, i = 0, 0
    while i < len(text):
        if text[i] == "\033":
            i = text.find("m", i) + 1 or len(text)
            continue
        out += 1
        i += 1
    return out


def _pad(text: str, width: int, align: str) -> str:
    gap = max(0, width - _visible_len(text))
    if align == "r":
        return " " * gap + text
    if align == "c":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def table(headers: list[str], rows: list[list[str]], aligns: str | None = None) -> str:
    """Render a box-drawn table. ``aligns`` is one of ``l``/``r``/``c`` per column."""
    aligns = aligns or "l" * len(headers)
    widths = [_visible_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _visible_len(str(cell)))

    def line(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right

    out = [line("┌", "┬", "┐")]
    out.append("│ " + " │ ".join(
        bold(_pad(h, w, aligns[i])) for i, (h, w) in enumerate(zip(headers, widths))
    ) + " │")
    out.append(line("├", "┼", "┤"))
    for row in rows:
        out.append("│ " + " │ ".join(
            _pad(str(cell), w, aligns[i]) for i, (cell, w) in enumerate(zip(row, widths))
        ) + " │")
    out.append(line("└", "┴", "┘"))
    return "\n".join(out)


_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values, width: int = 60) -> str:
    """One-line trace of ``values``, resampled to ``width`` columns."""
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return ""
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[int(i * step)] for i in range(width)]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span < 1e-12:
        return _BLOCKS[0] * len(vals)
    return "".join(_BLOCKS[min(7, int((v - lo) / span * 7.999))] for v in vals)


def header(title: str, subtitle: str = "") -> str:
    bar = "═" * max(len(title) + 4, 62)
    out = f"\n{cyan(bar)}\n  {bold(title)}"
    if subtitle:
        out += f"\n  {dim(subtitle)}"
    return out + f"\n{cyan(bar)}"


# ── Result records ────────────────────────────────────────────────────────────

@dataclass
class Check:
    """One pass/fail assertion about a measured value."""

    name: str
    value: float
    unit: str
    target: str          # human-readable requirement, e.g. "<= 1.0 mm"
    passed: bool
    detail: str = ""

    @property
    def mark(self) -> str:
        return green("PASS") if self.passed else red("FAIL")

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "unit": self.unit,
            "target": self.target,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class Scenario:
    """A named test case: its traces, its measured metrics, and its checks."""

    name: str
    description: str
    checks: list[Check] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    traces: dict = field(default_factory=dict)   # label -> list[float], for charts
    x_label: str = "frame"
    y_label: str = ""
    notes: str = ""

    @property
    def passed(self) -> bool:
        return all(chk.passed for chk in self.checks)

    def check(self, name: str, value: float, unit: str, *,
              max_value: float | None = None, min_value: float | None = None,
              detail: str = "") -> Check:
        """Record a check; supply ``max_value``, ``min_value``, or both."""
        # Cast out of numpy scalars here: a numpy comparison yields np.bool_,
        # which json.dumps refuses, and the failure would only surface at the
        # very end of a long run.
        value = float(value)
        ok = True
        parts = []
        if max_value is not None:
            ok = ok and value <= max_value
            parts.append(f"<= {max_value:g}")
        if min_value is not None:
            ok = ok and value >= min_value
            parts.append(f">= {min_value:g}")
        chk = Check(name, value, unit, " and ".join(parts) + f" {unit}".rstrip(), ok, detail)
        self.checks.append(chk)
        return chk

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "passed": self.passed,
            "checks": [chk.as_dict() for chk in self.checks],
            "metrics": {k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in self.metrics.items()},
            "traces": {k: [round(float(x), 6) for x in v] for k, v in self.traces.items()},
            "x_label": self.x_label,
            "y_label": self.y_label,
            "notes": self.notes,
        }


@dataclass
class BenchResult:
    """Everything one bench produced."""

    name: str
    subtitle: str
    scenarios: list[Scenario] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)   # {title, headers, rows, note}

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scenarios)

    @property
    def counts(self) -> tuple[int, int]:
        checks = [chk for s in self.scenarios for chk in s.checks]
        return sum(1 for chk in checks if chk.passed), len(checks)

    def as_dict(self) -> dict:
        ok, total = self.counts
        return {
            "name": self.name,
            "subtitle": self.subtitle,
            "passed": self.passed,
            "checks_passed": ok,
            "checks_total": total,
            "scenarios": [s.as_dict() for s in self.scenarios],
            "tables": self.tables,
        }


def print_scenarios(result: BenchResult) -> None:
    """Print each scenario's trace, notes, and check rows.

    The per-scenario ``metrics`` dicts are deliberately not rendered here or
    in the HTML report — they reach output only via ``--json``.
    """
    for scen in result.scenarios:
        status = green("PASS") if scen.passed else red("FAIL")
        print(f"\n  {bold(scen.name)}  [{status}]")
        print(f"  {dim(scen.description)}")

        for label, series in scen.traces.items():
            print(f"    {dim(label.ljust(18))} {sparkline(series)}")

        if scen.checks:
            rows = [
                [chk.name, f"{chk.value:,.4g}", chk.unit, chk.target, chk.mark,
                 dim(chk.detail)]
                for chk in scen.checks
            ]
            body = table(["check", "measured", "unit", "requirement", "", "note"],
                         rows, aligns="lrlllr")
            print("\n".join("    " + ln for ln in body.splitlines()))

        if scen.notes:
            print(f"    {dim(scen.notes)}")


def print_tables(result: BenchResult) -> None:
    for spec in result.tables:
        print(f"\n  {bold(spec['title'])}")
        if spec.get("note"):
            print(f"  {dim(spec['note'])}")
        body = table(spec["headers"], spec["rows"], spec.get("aligns"))
        print("\n".join("  " + ln for ln in body.splitlines()))
