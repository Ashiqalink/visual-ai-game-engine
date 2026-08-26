"""
devicelog.py — what the accelerators actually did during real play.

Everything in `accel.py` was measured on recorded clips. Real play is the one
thing a clip cannot stand in for: re-acquisition after a hand leaves frame,
motion blur, the game's own draw competing with the iGPU, thermal drift over a
ten-minute session. The rule in this workspace is that live hand tracking
cannot be verified offline — so this module exists to bring evidence *back*
from a live session instead of asking someone to remember how it felt.

A session writes JSON Lines to `data/device-logs/device-YYYY-MM-DD.jsonl`:

    {"kind": "session", ... hand_device, matte_device, devices, resolution ...}
    {"kind": "sample",  ... one per `interval_s` of play ...}
    {"kind": "close",   ... totals for the whole session ...}

A `sample` is a window, not a frame: per-frame lines would be tens of thousands
of records for a five-minute game and would themselves cost frame time. Each
window carries the frame count, the wall clock it covered, the resulting fps,
latency p50/p95, and — when hands run on OpenVINO — the *delta* in that
backend's own `timings` counters, which is how a run answers "did the NPU stay
at 3.4 ms once the game was drawing too, and how often did the detector have to
re-run because tracking dropped?"

Off by default, like the per-game run logs, and for the same reason: this is an
instrument for the machine doing the tuning, not something that should follow a
copy of the game to whoever else runs it. `VISUAL_AI_DEVICE_LOG=1` turns it on
for one run, the marker file `data/device-logs/logging-enabled` turns it on for
this machine, and an explicit `VISUAL_AI_DEVICE_LOG=0` beats the marker so one
run can always be kept out.

Standard library only, local files only — no socket, no network, and nothing
about the player: device names, timings and frame counts, never a frame.

Nothing here may take a game down. Every write is guarded; the first failure
disables the log for the session and is reported through :meth:`status`, which
belongs on the HUD — a logger that has silently stopped writing looks exactly
like a quiet session otherwise.
"""
from __future__ import annotations

import json
import os
import time

#: Turns logging on (1/true/yes/on) or off (0/false/no/off) for one run.
ENV_ENABLED = "VISUAL_AI_DEVICE_LOG"
#: Overrides where the logs are written.
ENV_DIR = "VISUAL_AI_DEVICE_LOG_DIR"

#: Written by `enable_here`; presence turns logging on for this machine.
MARKER_NAME = "logging-enabled"

_DEFAULT_DIR = os.path.join("data", "device-logs")

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def log_dir() -> str:
    """The directory session logs go to."""
    return os.environ.get(ENV_DIR, "").strip() or _DEFAULT_DIR


def marker_path(directory: str | None = None) -> str:
    """Where the opt-in marker sits."""
    return os.path.join(directory or log_dir(), MARKER_NAME)


def enabled(directory: str | None = None) -> bool:
    """Is device logging switched on, by environment or by marker file?"""
    value = os.environ.get(ENV_ENABLED, "").strip().lower()
    if value in _FALSE:
        return False
    if value in _TRUE:
        return True
    return os.path.exists(marker_path(directory))


def enable_here(directory: str | None = None, note: str = "") -> str:
    """Switch logging on for this machine by writing the marker."""
    directory = directory or log_dir()
    os.makedirs(directory, exist_ok=True)
    path = marker_path(directory)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Device logging is on for this machine.\n"
                     "Delete this file to turn it off.\n" + (note and note + "\n"))
    return path


def disable_here(directory: str | None = None) -> bool:
    """Switch logging off for this machine. Existing logs are left alone."""
    path = marker_path(directory)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. `values` need not be sorted; it is not mutated."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = int(round(fraction * (len(ordered) - 1)))
    return float(ordered[rank])


class DeviceLog:
    """
    One play session's device log.

    Construct it once next to the pipeline and call :meth:`frame` per frame;
    a frame that does not close a window costs a list append and one float
    comparison, which is why this can sit in the capture loop.

    `clock` and `now` are injected rather than read from :mod:`time` directly
    so a test can advance time by hand and assert on the numbers a window
    produces — a window whose fps cannot be predicted cannot be checked.
    """

    def __init__(self, path: str | None = None, directory: str | None = None,
                 interval_s: float = 5.0, is_enabled: bool | None = None,
                 clock=time.perf_counter, now=time.time):
        self.directory = directory or log_dir()
        self.interval_s = max(0.0, float(interval_s))
        self._clock = clock
        self._now = now
        self.enabled = enabled(self.directory) if is_enabled is None else bool(is_enabled)
        self.path = path or os.path.join(
            self.directory,
            "device-{}.jsonl".format(
                time.strftime("%Y-%m-%d", time.localtime(self._now()))))
        self.error: str | None = None
        self.records_written = 0

        self._session_started = self._clock()
        self._window_started = self._session_started
        self._latencies: list[float] = []
        self._hands: list[int] = []
        self._session_frames = 0
        self._samples = 0
        #: Latest value of every counter the caller has handed over.
        self._counters: dict[str, float] = {}
        #: Their values when the current window opened, for the delta.
        self._window_counters: dict[str, float] = {}
        self._closed = False

    # ── reporting ─────────────────────────────────────────────────────────────

    def status(self) -> str:
        """One short HUD string: what the log is doing, or why it is not."""
        if self.error:
            return f"device log off ({self.error})"
        if not self.enabled:
            return "device log off"
        return f"device log {os.path.basename(self.path)} ({self.records_written})"

    # ── writing ───────────────────────────────────────────────────────────────

    def _write(self, record: dict) -> None:
        """Append one record. A failure disables the log rather than raising."""
        if not self.enabled:
            return
        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as exc:
            # A full disk or a read-only directory must cost the player
            # nothing but the log itself — and must not be silent.
            self.enabled = False
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[devicelog] logging stopped: {self.error}")
            return
        self.records_written += 1

    def session(self, **fields) -> None:
        """
        Write the header: which device each network landed on, and on what.

        The fallback strings from `accel.describe` belong here in full — a log
        that says "CPU (MediaPipe)" without "(npu unavailable)" cannot tell a
        machine that has no NPU from one whose driver failed that morning.
        """
        record = {"kind": "session", "t": round(self._now(), 3)}
        record.update(fields)
        self._write(record)

    def frame(self, latency_ms: float | None = None, hands: int = 0,
              counters: dict | None = None) -> None:
        """
        Record one processed frame, closing the window if it is due.

        `counters` is a backend's own cumulative dictionary — pass
        `OpenVINOHands.timings` straight in. It is read, never held: the window
        stores the running values and reports the difference, so a caller
        handing over the same mutable dict every frame is correct.
        """
        if not self.enabled:
            return
        self._session_frames += 1
        if latency_ms is not None:
            self._latencies.append(float(latency_ms))
        self._hands.append(int(hands))
        if counters:
            for key, value in counters.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._counters[key] = float(value)
        if self._clock() - self._window_started >= self.interval_s:
            self.flush()

    def _window(self, kind: str, started: float, frames: int,
                latencies: list[float], hands: list[int]) -> dict:
        elapsed = max(0.0, self._clock() - started)
        record = {
            "kind": kind,
            "t": round(self._now(), 3),
            "seconds": round(elapsed, 3),
            "frames": frames,
            "fps": round(frames / elapsed, 2) if elapsed > 0 else 0.0,
        }
        if latencies:
            record["latency_ms"] = {
                "mean": round(sum(latencies) / len(latencies), 3),
                "p50": round(_percentile(latencies, 0.50), 3),
                "p95": round(_percentile(latencies, 0.95), 3),
                "max": round(max(latencies), 3),
            }
        if hands:
            record["hands_mean"] = round(sum(hands) / len(hands), 3)
        return record

    def flush(self) -> None:
        """Close the current window and write it as a `sample`. No frames, no line."""
        if not self.enabled or not self._hands:
            return
        record = self._window("sample", self._window_started, len(self._hands),
                              self._latencies, self._hands)
        delta = {key: round(value - self._window_counters.get(key, 0.0), 3)
                 for key, value in self._counters.items()}
        if delta:
            record["counters"] = delta
        self._samples += 1
        record["sample"] = self._samples
        self._write(record)
        # The next window measures from the same instant this one ended, so
        # windows tile the session rather than losing the write to a gap.
        self._window_started = self._clock()
        self._latencies = []
        self._hands = []
        self._window_counters = dict(self._counters)

    def close(self, **fields) -> None:
        """Flush the open window and write the session totals. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.flush()
        if not self.enabled:
            return
        record = self._window("close", self._session_started,
                              self._session_frames, [], [])
        record["samples"] = self._samples
        if self._counters:
            record["counters"] = {key: round(value, 3)
                                  for key, value in self._counters.items()}
        record.update(fields)
        self._write(record)


def load(path: str) -> list[dict]:
    """Read a session log back. A run killed mid-write leaves a partial line."""
    records: list[dict] = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                print(f"[devicelog] skipping unreadable line "
                      f"{os.path.basename(path)}:{line_no}")
    return records
