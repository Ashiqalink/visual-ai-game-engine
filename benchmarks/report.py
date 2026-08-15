"""
report.py — Renders bench results as a standalone HTML page.

No dependencies and no network: every chart is inline SVG drawn from the trace
arrays the benches collected, so the file opens anywhere and survives being
emailed around. Light and dark themes both follow the viewer's system setting.
"""

from __future__ import annotations

import html
import math
from datetime import datetime

_MAX_POINTS = 400          # traces are resampled to this before drawing
_SERIES_COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"]

_CSS = """
:root {
  --bg:#f7f7f8; --panel:#fff; --ink:#1b1b1f; --muted:#6b6b75; --line:#e3e3e8;
  --pass:#1a7f4b; --pass-bg:#e6f4ec; --fail:#b3261e; --fail-bg:#fbeae9;
  --s1:#9aa0a6; --s2:#1b1b1f; --s3:#2563eb; --s4:#d97706; --grid:#eeeef2;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme=light]) {
    --bg:#131316; --panel:#1a1a1f; --ink:#ececf1; --muted:#9a9aa5; --line:#2c2c34;
    --pass:#54d18b; --pass-bg:#12291e; --fail:#ff8a80; --fail-bg:#2c1614;
    --s1:#5c5c66; --s2:#ececf1; --s3:#7aa2ff; --s4:#f0b429; --grid:#232329;
  }
}
* { box-sizing:border-box; }
body { margin:0; padding:32px 20px 72px; background:var(--bg); color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1080px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:19px; margin:38px 0 10px; letter-spacing:-0.01em; }
h3 { font-size:15px; margin:0; }
.sub { color:var(--muted); font-size:13px; margin:0 0 6px; }
.mono { font-family:ui-monospace,"Cascadia Code",Consolas,monospace; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:16px 18px; margin:12px 0; }
.row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.spacer { flex:1; }
.badge { font-size:11px; font-weight:700; letter-spacing:.06em; padding:3px 9px;
  border-radius:999px; text-transform:uppercase; }
.pass { color:var(--pass); background:var(--pass-bg); }
.fail { color:var(--fail); background:var(--fail-bg); }
.summary { display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 8px; }
.tile { flex:1 1 190px; background:var(--panel); border:1px solid var(--line);
  border-radius:12px; padding:14px 16px; }
.tile .n { font-size:24px; font-weight:650; letter-spacing:-0.02em; }
.tile .l { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
.tbl-scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line);
  white-space:nowrap; }
th { color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase;
  letter-spacing:.05em; }
td.num, th.num { text-align:right; font-family:ui-monospace,Consolas,monospace; }
tr:last-child td { border-bottom:none; }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--muted);
  margin:8px 0 0; }
.swatch { display:inline-block; width:16px; height:3px; border-radius:2px;
  vertical-align:middle; margin-right:6px; }
svg { width:100%; height:auto; display:block; }
.note { color:var(--muted); font-size:12px; margin:6px 0 0; }
footer { color:var(--muted); font-size:12px; margin-top:40px; text-align:center; }
"""


def _e(text) -> str:
    return html.escape(str(text))


def _resample(values: list[float], limit: int = _MAX_POINTS) -> list[float]:
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(vals) <= limit:
        return vals
    step = len(vals) / limit
    return [vals[int(i * step)] for i in range(limit)]


def _chart(traces: dict, y_label: str, x_label: str) -> str:
    """Multi-series line chart as inline SVG."""
    series = {k: _resample(v) for k, v in traces.items() if v}
    series = {k: v for k, v in series.items() if len(v) >= 2}
    if not series:
        return ""

    w, h = 1000.0, 260.0
    pad_l, pad_r, pad_t, pad_b = 58.0, 12.0, 12.0, 26.0
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b

    lo = min(min(v) for v in series.values())
    hi = max(max(v) for v in series.values())
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad_v = (hi - lo) * 0.08
    lo, hi = lo - pad_v, hi + pad_v
    n_max = max(len(v) for v in series.values())

    def x_of(i: int, n: int) -> float:
        return pad_l + (i / max(1, n - 1)) * plot_w

    def y_of(v: float) -> float:
        return pad_t + (1.0 - (v - lo) / (hi - lo)) * plot_h

    parts = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" role="img" '
             f'aria-label="{_e(", ".join(series))}">']

    # Horizontal gridlines with value labels.
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = hi - frac * (hi - lo)
        y = pad_t + frac * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="var(--muted)" '
                     f'font-family="ui-monospace,Consolas,monospace">{value:,.4g}</text>')

    for idx, (label, values) in enumerate(series.items()):
        color = _SERIES_COLORS[idx % len(_SERIES_COLORS)]
        width = 1.2 if idx == 0 else 1.8
        pts = " ".join(f"{x_of(i, len(values)):.1f},{y_of(v):.1f}"
                       for i, v in enumerate(values))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="{width}" stroke-linejoin="round" '
                     f'stroke-linecap="round" opacity="{0.75 if idx == 0 else 1}"/>')

    parts.append(f'<text x="{pad_l}" y="{h - 6:.0f}" font-size="11" '
                 f'fill="var(--muted)">0</text>')
    parts.append(f'<text x="{w - pad_r:.0f}" y="{h - 6:.0f}" font-size="11" '
                 f'text-anchor="end" fill="var(--muted)">{n_max} {_e(x_label)}s</text>')
    if y_label:
        parts.append(f'<text x="{pad_l}" y="{pad_t - 1:.0f}" font-size="11" '
                     f'fill="var(--muted)">{_e(y_label)}</text>')
    parts.append("</svg>")

    legend = " ".join(
        f'<span><span class="swatch" style="background:'
        f'{_SERIES_COLORS[i % len(_SERIES_COLORS)]}"></span>{_e(label)}</span>'
        for i, label in enumerate(series)
    )
    return "".join(parts) + f'<div class="legend">{legend}</div>'


def _checks_table(checks: list[dict]) -> str:
    if not checks:
        return ""
    rows = []
    for chk in checks:
        badge = ('<span class="badge pass">pass</span>' if chk["passed"]
                 else '<span class="badge fail">fail</span>')
        value = chk["value"]
        shown = "never" if value == float("inf") else f"{value:,.4g}"
        rows.append(
            f'<tr><td>{_e(chk["name"])}</td>'
            f'<td class="num">{shown}</td><td>{_e(chk["unit"])}</td>'
            f'<td class="mono">{_e(chk["target"])}</td><td>{badge}</td>'
            f'<td style="color:var(--muted);white-space:normal">{_e(chk["detail"])}</td></tr>'
        )
    return ('<div class="tbl-scroll"><table><thead><tr><th>Check</th>'
            '<th class="num">Measured</th><th>Unit</th><th>Requirement</th>'
            '<th>Result</th><th>Note</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def _generic_table(spec: dict) -> str:
    aligns = spec.get("aligns") or "l" * len(spec["headers"])
    head = "".join(
        f'<th class="{"num" if aligns[i] == "r" else ""}">{_e(hdr)}</th>'
        for i, hdr in enumerate(spec["headers"])
    )
    body = "".join(
        "<tr>" + "".join(
            f'<td class="{"num" if aligns[i] == "r" else ""}">{_e(cell)}</td>'
            for i, cell in enumerate(row)
        ) + "</tr>"
        for row in spec["rows"]
    )
    note = f'<p class="note">{_e(spec["note"])}</p>' if spec.get("note") else ""
    return (f'<div class="panel"><h3>{_e(spec["title"])}</h3>{note}'
            f'<div class="tbl-scroll"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div></div>')


def build_html(results: list[dict], modules: dict[str, str]) -> str:
    total_checks = sum(r["checks_total"] for r in results)
    passed_checks = sum(r["checks_passed"] for r in results)
    scenarios = sum(len(r["scenarios"]) for r in results)
    all_pass = passed_checks == total_checks
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = [
        '<div class="wrap">',
        "<h1>Stabilizer &amp; filter test bench</h1>",
        f'<p class="sub">Run {_e(stamp)} · {scenarios} scenarios · '
        f'{total_checks} checks</p>',
        '<div class="summary">',
        f'<div class="tile"><div class="n">'
        f'{"ALL PASS" if all_pass else f"{total_checks - passed_checks} FAILING"}'
        f'</div><div class="l">Overall</div></div>',
        f'<div class="tile"><div class="n">{passed_checks}/{total_checks}</div>'
        f'<div class="l">Checks passed</div></div>',
        f'<div class="tile"><div class="n">{scenarios}</div>'
        f'<div class="l">Scenarios</div></div>',
        "</div>",
        '<div class="panel"><h3>Code under test</h3>'
        + "".join(f'<p class="note mono">{_e(name)} — {_e(path)}</p>'
                  for name, path in modules.items())
        + "</div>",
    ]

    for res in results:
        badge = ('<span class="badge pass">pass</span>' if res["passed"]
                 else '<span class="badge fail">fail</span>')
        body.append(f'<h2>{_e(res["name"])} {badge}</h2>')
        body.append(f'<p class="sub mono">{_e(res["subtitle"])}</p>')

        for scen in res["scenarios"]:
            scen_badge = ('<span class="badge pass">pass</span>' if scen["passed"]
                          else '<span class="badge fail">fail</span>')
            body.append('<div class="panel">')
            body.append(f'<div class="row"><h3>{_e(scen["name"])}</h3>'
                        f'<div class="spacer"></div>{scen_badge}</div>')
            body.append(f'<p class="note">{_e(scen["description"])}</p>')
            chart = _chart(scen["traces"], scen["y_label"], scen["x_label"])
            if chart:
                body.append(chart)
            body.append(_checks_table(scen["checks"]))
            if scen["notes"]:
                body.append(f'<p class="note">{_e(scen["notes"])}</p>')
            body.append("</div>")

        for spec in res["tables"]:
            body.append(_generic_table(spec))

    body.append('<footer>Generated by benchmarks/run_all.py — '
                're-run after any tuning change to compare.</footer>')
    body.append("</div>")

    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Stabilizer &amp; filter bench</title><style>{_CSS}</style></head>'
            f'<body>{"".join(body)}</body></html>')
