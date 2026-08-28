"""
bench_accel_policy.py — re-measure the three claims behind ``accel.DEFAULT_PRESET``.

``accel.py``'s module docstring justifies ``"auto": {"hand": ("NPU", "GPU")}``
with three numbers and one assertion:

    hand tracking, end to end   MediaPipe CPU 19.4 ms   iGPU 2.0 ms   NPU 3.4 ms
    CPU cost of the same        1.0 core                2.3 cores     1.4 cores

    "...hands on the NPU — nearly iGPU speed, a third of the CPU, and it leaves
     the iGPU free for the game's own drawing"

Those were measured once, on a Core Ultra 5 225H, and nothing has re-measured
them since. Every part of the policy is falsifiable, so this bench states each
part as a check that can fail:

    latency   the preferred device is within ``--latency-tol-pct`` of the fastest
    cpu       the preferred device costs no more CPU than the runner-up
    headroom  under a busy iGPU, the preferred device degrades less

Every number is reported **both ways**: the millisecond it was measured in, and
the percentage it is of a baseline measured in the same run. Both are needed and
neither replaces the other. Milliseconds are what a frame budget is spent in —
"2.0 ms" is the thing a 16.7 ms frame can or cannot afford, and a ratio cannot
answer that. Percentages are what survives a change of machine: a faster CPU
makes the MediaPipe baseline quicker and the accelerator's win smaller, so two
machines' millisecond tables cannot be laid side by side, while their ratios can.
Each device therefore carries two denominators:

    vs mediapipe   what the accelerator is worth on *this* machine's CPU — the
                   only backend every machine has, and therefore the only
                   baseline a second machine can reproduce
    vs the runner-up   whether the preset picked the right device, which is a
                   comparison and never needs a unit at all

The checks are stated in percent, because a tolerance in milliseconds means
something different on every machine. A second machine running this should
produce different milliseconds and similar percentages. Where the percentages
differ too, the preset is machine-dependent and ``accel.py``'s single hard-coded
order is the thing to revisit — so ``--json`` also carries the machine's CPU,
core count and device names, since a percentage is only comparable if you know
what it was measured on.

``bench_hand_backend.py`` already answers "does each device track the same hand"
— fingertip rmse against MediaPipe — and this bench deliberately does not repeat
that. It answers the different question of *which device the default should
pick*, which needs the two axes that bench does not measure: CPU cores burned,
and what happens when something else wants the iGPU.

Three things this measures carefully, because each of them moved the answer:

  * **Tracking frames are reported apart from searching frames.** Only a quarter
    of the clip holds a hand, and the palm detector runs on the other three
    quarters. Pooled, the NPU looks 2x slower than the iGPU; split, it is level
    on the frames that are what play consists of and slower only while hunting.
    A pooled median measures a session spent staring at an empty room.
  * **Rounds interleave the devices.** Each round measures every backend once,
    so thermal state and background load land on all of them together. Running
    one device to completion and then the next attributes the machine's drift
    to whichever device went last — and the drift here is comparable in size to
    the difference being measured.
  * **``--order reverse`` flips the sequence within a round.** Two networks run
    back to back cost more than either alone, so position in the sequence is
    itself a variable. If the verdict depends on the order, the verdict is
    about the bench and not about the machine.

The ``±`` columns are round-to-round spread inside **one** process, and they are
smaller than this bench's real error. The same scenario measured in a fresh
session has moved 20% on numbers whose ± said 8% — see the note in
``sweep_accel_policy.py`` for the run that showed it. Treat a single run's
FAIL as a reason to run it again, not as a result.

The headroom load is a proxy: the palm network looped on the iGPU, not a game's
real draw path. Games here draw through pygame/SDL and OpenCV, which is mostly
CPU work, so this measures iGPU contention in isolation rather than reproducing
a frame of Sling. It is labelled that way in the output and the check is scoped
to what it actually shows.

Needs hardware — an Intel iGPU or NPU and the ``openvino`` package. Without at
least two devices to choose between there is no policy to test, and it says so
and exits 0.

Run it::

    "D:/visual/.venv/Scripts/python.exe" benchmarks/bench_accel_policy.py
    "D:/visual/.venv/Scripts/python.exe" benchmarks/bench_accel_policy.py --tier lite --hands 2
    "D:/visual/.venv/Scripts/python.exe" benchmarks/bench_accel_policy.py --rounds 5 --order reverse
    # one JSON object on stdout, for sweep_accel_policy.py:
    "D:/visual/.venv/Scripts/python.exe" benchmarks/bench_accel_policy.py --json

Exit code 1 means the measured machine no longer supports the policy.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness import bold, dim, green, header, red, table  # noqa: E402

from visual_ai import accel, openvino_hands  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hand_motion.mp4"
SIZE = (640, 480)

#: The denominator for every "vs mediapipe" percentage. MediaPipe's CPU graph is
#: the one backend present on every machine, so it is the only baseline a second
#: machine can reproduce.
MP_LABEL = "mediapipe (CPU)"

#: How far behind the fastest device the preferred one may sit and still be
#: "nearly iGPU speed", as a percentage rather than a millisecond count so the
#: same tolerance means the same thing on a slower machine. 25% is set from the
#: bench's own error: flipping ``--order`` moves these devices by up to 21% on
#: this machine, so a tighter tolerance would be measuring the measurement.
LATENCY_TOL_PCT = 25.0
#: What share of the runner-up's CPU cost the preferred device may burn. 100%
#: is "no worse"; the policy claims far better than that, and the gap measured
#: here is ~60%, so this fails long before the claim is merely dented.
CPU_TOL_PCT = 100.0
#: Clip passes inside one measurement.
PASSES = 2
#: Measurement rounds. Every backend is measured once per round, in order, so a
#: machine that warms up or throttles does it to all of them at once. Three is
#: the minimum that gives a spread worth printing; the aggregate is the median
#: of the per-round numbers, so one bad round does not carry the answer.
ROUNDS = 3


def load_frames() -> list[np.ndarray]:
    if not FIXTURE.is_file():
        raise SystemExit(f"missing fixture: {FIXTURE}")
    capture = cv2.VideoCapture(str(FIXTURE))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(cv2.resize(frame, SIZE), cv2.COLOR_BGR2RGB)
        # The pipeline hands MediaPipe a non-writeable view so the wrapper skips
        # its defensive copy; measuring the writeable path would measure a copy
        # the real caller never pays.
        rgb.flags.writeable = False
        frames.append(rgb)
    capture.release()
    return frames


class GPULoad:
    """A busy iGPU, as a stand-in for a game drawing while hands are tracked.

    The palm network in a loop rather than a real draw path: what the headroom
    check needs is *something else wanting the iGPU*, and a network this repo
    already ships is reproducible in a way a windowed render is not.
    """

    def __init__(self, tier: str) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.available = False
        self.error = ""
        try:
            import openvino as ov

            palm_path, _ = openvino_hands._model_paths(tier)
            core = ov.Core()
            compiled = core.compile_model(core.read_model(palm_path), "GPU",
                                          {"PERFORMANCE_HINT": "THROUGHPUT"})
            self._request = compiled.create_infer_request()
            shape = compiled.input(0).shape
            self._tensor = np.zeros([int(d) for d in shape], dtype=np.float32)
            self.available = True
        except Exception as exc:                      # pragma: no cover - hardware
            self.error = f"{type(exc).__name__}: {exc}"

    def _spin(self) -> None:
        while not self._stop.is_set():
            self._request.infer([self._tensor])

    def __enter__(self) -> GPULoad:
        if self.available:
            self._stop.clear()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
            time.sleep(0.3)                           # let the device get busy
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def measure(tracker, frames: list[np.ndarray], passes: int) -> dict:
    """Per-frame latency and the CPU cores it burned, over `passes` clip passes.

    CPU cost is process-wide on purpose: an accelerator that offloads the maths
    and then spin-waits for the result has not saved the CPU anything, and a
    per-thread number would hide exactly that.
    """
    for frame in frames[:15]:
        tracker.process(frame)

    process = psutil.Process()
    times: list[float] = []
    # Split on whether the frame came back with a hand: a frame the detector is
    # searching is a different piece of work from a frame it is tracking, and
    # this clip is three-quarters searching. See the module docstring.
    tracking: list[float] = []
    searching: list[float] = []
    cpu_before = process.cpu_times()
    wall_before = time.perf_counter()
    for _ in range(passes):
        for frame in frames:
            started = time.perf_counter()
            result = tracker.process(frame)
            elapsed = (time.perf_counter() - started) * 1e3
            times.append(elapsed)
            (tracking if result.multi_hand_landmarks else searching).append(elapsed)
    wall = time.perf_counter() - wall_before
    cpu_after = process.cpu_times()
    cpu_s = ((cpu_after.user - cpu_before.user)
             + (cpu_after.system - cpu_before.system))
    ordered = sorted(times)
    return {
        "median": statistics.median(ordered),
        "p90": ordered[int(0.90 * len(ordered))],
        # The number the policy is about: what a frame costs while a player's
        # hand is actually in view.
        "tracking": statistics.median(tracking) if tracking else float("nan"),
        "searching": statistics.median(searching) if searching else float("nan"),
        "cores": cpu_s / wall if wall > 0 else float("nan"),
        "detection": len(tracking) / len(times),
    }


def build(device: str | None, tier: str, hands: int):
    """A tracker for `device`, or None if this machine cannot make one.

    `device=None` means MediaPipe's own CPU graph — the path every fallback
    out of the preset lands on, and the only baseline the policy is arguing
    against.
    """
    complexity = 1 if tier == "full" else 0
    if device is None:
        try:
            from mediapipe.python.solutions import hands as mp_hands
        except Exception:
            try:
                import mediapipe.solutions.hands as mp_hands
            except Exception:
                return None
        return mp_hands.Hands(
            static_image_mode=False, max_num_hands=hands,
            model_complexity=complexity, min_detection_confidence=0.7,
            # 0.65 on the CPU graph, 0.45 on OpenVINO — the gates differ by
            # design (see OpenVINOHands.__init__), and using one number for
            # both would measure a backend nobody ships.
            min_tracking_confidence=0.65)
    try:
        return openvino_hands.OpenVINOHands(
            device=device, max_num_hands=hands, model_complexity=complexity,
            min_detection_confidence=0.7, min_tracking_confidence=0.45)
    except openvino_hands.OpenVINOHandsUnavailable:
        return None


def rounds_of(trackers: dict, frames, passes: int, rounds: int) -> dict[str, list[dict]]:
    """Measure every backend once per round, in `trackers` order.

    Interleaved rather than one backend at a time: the machine's own drift over
    a two-minute bench is the same size as the gaps being measured, and running
    a device to completion charges that drift to whichever device happened to
    be last.
    """
    per_label: dict[str, list[dict]] = {label: [] for label in trackers}
    for _ in range(rounds):
        for label, tracker in trackers.items():
            per_label[label].append(measure(tracker, frames, passes))
    return per_label


def aggregate(samples: list[dict]) -> dict:
    """Median across rounds, plus the round-to-round spread on each number."""
    out: dict[str, float] = {}
    for key in ("tracking", "searching", "median", "p90", "cores", "detection"):
        values = [s[key] for s in samples]
        out[key] = statistics.median(values)
        out[key + "_spread"] = max(values) - min(values)
    out["rounds"] = len(samples)
    return out


def ratio(value: float, base: float) -> float:
    """`value` as a percentage of `base`, or nan when the base is unusable.

    Returned rather than formatted so the JSON carries the same number the table
    prints — a second machine is compared against these, and a rounded string is
    not a measurement.
    """
    if not base or base != base or value != value:      # nan-safe
        return float("nan")
    return 100.0 * value / base


def machine_spec(devices: list[str]) -> dict:
    """What the percentages were measured on.

    A ratio is only comparable across machines if both are labelled. Without
    this, two runs that disagree are indistinguishable from one machine having
    an NPU the other does not.
    """
    # device_name falls back to echoing the id, which is not a CPU model; only
    # take it when it actually resolved to something else.
    cpu = accel.device_name("CPU")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() if cpu == "CPU" else cpu,
        "cores_logical": psutil.cpu_count(logical=True),
        "cores_physical": psutil.cpu_count(logical=False),
        "devices": {d: accel.device_name(d) for d in devices},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--tier", choices=("full", "lite"), default="full",
                        help="model tier; full is VisionPipeline's default "
                             "model_complexity=1")
    parser.add_argument("--hands", type=int, default=1,
                        help="max hands to track (the clip holds one)")
    parser.add_argument("--passes", type=int, default=PASSES,
                        help="clip passes per measurement")
    parser.add_argument("--rounds", type=int, default=ROUNDS,
                        help="interleaved measurement rounds")
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward",
                        help="device order within a round; reverse tests whether "
                             "position in the sequence is what is being measured")
    # Tolerances are arguments, not only constants, so the failure path can be
    # exercised on a machine where the policy currently holds. A check nobody
    # has watched fail is a check nobody has tested.
    parser.add_argument("--latency-tol-pct", type=float, default=LATENCY_TOL_PCT,
                        metavar="PCT", help=f"how much slower than the fastest "
                                            f"device the preferred one may be "
                                            f"(default {LATENCY_TOL_PCT:.0f}%%)")
    parser.add_argument("--cpu-tol-pct", type=float, default=CPU_TOL_PCT,
                        metavar="PCT", help=f"share of the runner-up's cores the "
                                            f"preferred device may burn "
                                            f"(default {CPU_TOL_PCT:.0f}%%)")
    parser.add_argument("--json", action="store_true",
                        help="emit one JSON object on stdout and nothing else")
    args = parser.parse_args()

    quiet = args.json

    def say(*parts) -> None:
        if not quiet:
            print(*parts)

    present = accel.available_devices()
    preferred_order = [d for d in ("NPU", "GPU") if d in present]
    say(header("accel auto-preset policy",
               f"{FIXTURE.name} at {SIZE[0]}x{SIZE[1]}, tier={args.tier}, "
               f"hands={args.hands}, {args.rounds} rounds x {args.passes} passes, "
               f"order={args.order}"))
    if not present:
        say("\nno OpenVINO devices (is `openvino` installed?) — "
            "the preset can only resolve to the MediaPipe CPU path here.")
        if quiet:
            print(json.dumps({"skipped": "no openvino devices"}))
        return 0
    say(f"\ndevices: {', '.join(f'{d} ({accel.device_name(d)})' for d in present)}")
    say(f"auto preset prefers: {' then '.join(preferred_order) or '(nothing)'}")
    if len(preferred_order) < 2:
        say(dim("\nfewer than two candidate devices — there is no choice to "
                "test, and the numbers below would not be a policy check."))

    frames = load_frames()

    # ── Build every backend once, then interleave ─────────────────────────────
    sequence: list[tuple[str, str | None]] = (
        [(MP_LABEL, None)] + [(d, d) for d in present])
    if args.order == "reverse":
        sequence.reverse()
    trackers = {}
    for label, device in sequence:
        tracker = build(device, args.tier, args.hands)
        if tracker is None:
            say(dim(f"  {label}: unavailable"))
            continue
        trackers[label] = tracker

    idle = {label: aggregate(samples) for label, samples
            in rounds_of(trackers, frames, args.passes, args.rounds).items()}

    rows = [[label, f"{s['tracking']:.2f}", f"±{s['tracking_spread'] / 2:.2f}",
             f"{s['searching']:.2f}", f"{s['p90']:.2f}",
             f"{s['cores']:.2f}", f"±{s['cores_spread'] / 2:.2f}",
             f"{s['detection']:.0%}"]
            for label, s in idle.items()]
    say("\n" + bold("idle machine")
        + dim("  (tracking = a hand is in frame, which is what play looks like; "
              "± is half the round-to-round spread)"))
    say(table(["backend", "tracking ms", "±", "searching ms", "p90 ms",
               "cores", "±", "det"], rows, aligns="lrrrrrrr"))

    # ── The same numbers as ratios ────────────────────────────────────────────
    # Printed as a separate table rather than extra columns because it is a
    # different question: the ms table asks what a frame costs here, this one
    # asks what carries to another machine. Denominator is MediaPipe's CPU graph
    # when it built, and the fastest measured backend otherwise — a machine
    # without MediaPipe still has a comparison, it just has a weaker one, and the
    # header says which one was used.
    base_label = MP_LABEL if MP_LABEL in idle else min(
        idle, key=lambda k: idle[k]["tracking"])
    base = idle[base_label]
    relative = {
        label: {
            "tracking_pct": ratio(s["tracking"], base["tracking"]),
            "searching_pct": ratio(s["searching"], base["searching"]),
            "p90_pct": ratio(s["p90"], base["p90"]),
            "cores_pct": ratio(s["cores"], base["cores"]),
            # Round-to-round spread as a share of the number it sits on: this is
            # the bench's own error bar, and any gap smaller than it is noise.
            "tracking_noise_pct": ratio(s["tracking_spread"], s["tracking"]),
        }
        for label, s in idle.items()
    }
    rows = [[label, f"{r['tracking_pct']:.0f}%", f"{r['searching_pct']:.0f}%",
             f"{r['p90_pct']:.0f}%", f"{r['cores_pct']:.0f}%",
             f"{100 - r['tracking_pct']:+.0f}%",
             f"±{r['tracking_noise_pct'] / 2:.0f}%"]
            for label, r in relative.items()]
    say("\n" + bold(f"relative to {base_label}")
        + dim("  (100% = the baseline; lower is faster/cheaper. These are the "
              "numbers that should match on another machine)"))
    say(table(["backend", "tracking", "searching", "p90", "cores", "speedup",
               "noise"], rows, aligns="lrrrrrr"))

    # ── Contended iGPU ────────────────────────────────────────────────────────
    load = GPULoad(args.tier)
    busy: dict[str, dict] = {}
    if not load.available:
        say(dim(f"\niGPU load unavailable ({load.error or 'no GPU'}) "
                "— the headroom claim is not tested."))
    else:
        ov_trackers = {label: t for label, t in trackers.items() if label in present}
        with load:
            busy = {label: aggregate(samples) for label, samples
                    in rounds_of(ov_trackers, frames, args.passes, args.rounds).items()}
        # Degradation carries both units for the same reason as everything else:
        # "+1.4 ms" is what the frame budget loses here, "+70%" is what another
        # machine should also see if the effect is about the device and not
        # about this silicon.
        rows = [[label, f"{s['tracking']:.2f}", f"±{s['tracking_spread'] / 2:.2f}",
                 f"{s['tracking'] - idle[label]['tracking']:+.2f}",
                 f"{ratio(s['tracking'], idle[label]['tracking']) - 100:+.0f}%",
                 f"{s['cores']:.2f}", f"{s['detection']:.0%}"]
                for label, s in busy.items()]
        say("\n" + bold("iGPU busy") + dim("  (palm net looped on the GPU — a "
                                           "proxy load, not a game's draw path)"))
        say(table(["backend", "tracking ms", "±", "vs idle", "vs idle %", "cores",
                   "det"], rows, aligns="lrrrrrr"))

    # ── The policy, as checks that can fail ───────────────────────────────────
    verdict: dict = {
        "tier": args.tier, "hands": args.hands, "order": args.order,
        "rounds": args.rounds, "passes": args.passes,
        "devices": present, "prefers": preferred_order,
        "machine": machine_spec(present),
        "tolerances": {"latency_pct": args.latency_tol_pct,
                       "cpu_pct": args.cpu_tol_pct},
        "idle": idle, "busy": busy,
        "relative": relative, "relative_base": base_label,
        "checks": [], "failures": [],
    }

    def finish(code: int) -> int:
        if quiet:
            print(json.dumps(verdict, default=float))
        return code

    if len(preferred_order) < 2 or not {*preferred_order} <= {*idle}:
        say(dim("\nno policy checks run: fewer than two candidate devices built."))
        verdict["skipped"] = "fewer than two candidate devices"
        return finish(0)

    chosen, runner_up = preferred_order[0], preferred_order[1]
    lines: list[tuple[bool, str]] = []

    # Each check is decided in percent and reported in both units: the percentage
    # is what a second machine can compare against, the millisecond is what tells
    # you whether the gap matters inside a 16.7 ms frame.
    fastest = min(idle, key=lambda k: idle[k]["tracking"])
    gap = idle[chosen]["tracking"] - idle[fastest]["tracking"]
    gap_pct = ratio(gap, idle[fastest]["tracking"])
    ok = gap_pct <= args.latency_tol_pct
    lines.append((ok, f"latency   {chosen} {idle[chosen]['tracking']:.2f} ms is "
                      f"{gap_pct:+.0f}% ({gap:+.2f} ms) against the fastest "
                      f"({fastest} {idle[fastest]['tracking']:.2f} ms), tolerance "
                      f"{args.latency_tol_pct:.0f}%"))
    verdict["checks"].append({"name": "latency", "passed": ok, "pct": gap_pct,
                              "ms": gap, "fastest": fastest})
    if not ok:
        verdict["failures"].append(
            f"{chosen} is {gap_pct:.0f}% ({gap:.2f} ms) slower than {fastest}; "
            f"'nearly iGPU speed' no longer holds on this machine")

    # A share of the runner-up rather than a difference of cores: "burns 60% of
    # the CPU the iGPU does" is the same sentence on any core count, where "0.9
    # cores fewer" is not.
    core_gap = idle[chosen]["cores"] - idle[runner_up]["cores"]
    core_pct = ratio(idle[chosen]["cores"], idle[runner_up]["cores"])
    ok = core_pct <= args.cpu_tol_pct
    lines.append((ok, f"cpu       {chosen} burns {core_pct:.0f}% of {runner_up}'s "
                      f"CPU ({idle[chosen]['cores']:.2f} vs "
                      f"{idle[runner_up]['cores']:.2f} cores, {core_gap:+.2f})"))
    verdict["checks"].append({"name": "cpu", "passed": ok, "pct": core_pct,
                              "ms": None, "cores": core_gap})
    if not ok:
        verdict["failures"].append(
            f"{chosen} costs {core_pct:.0f}% of {runner_up}'s CPU "
            f"({core_gap:+.2f} cores); the CPU half of the policy is inverted")

    if busy.get(chosen) and busy.get(runner_up):
        chosen_delta = busy[chosen]["tracking"] - idle[chosen]["tracking"]
        runner_delta = busy[runner_up]["tracking"] - idle[runner_up]["tracking"]
        # Each device's degradation as a share of its *own* idle time, so a
        # device that starts slower is not credited for its head start.
        chosen_pct = ratio(chosen_delta, idle[chosen]["tracking"])
        runner_pct = ratio(runner_delta, idle[runner_up]["tracking"])
        ok = chosen_pct <= runner_pct
        lines.append((ok, f"headroom  under a busy iGPU {chosen} degrades "
                          f"{chosen_pct:+.0f}% ({chosen_delta:+.2f} ms) against "
                          f"{runner_up}'s {runner_pct:+.0f}% "
                          f"({runner_delta:+.2f} ms)"))
        verdict["checks"].append({"name": "headroom", "passed": ok,
                                  "pct": chosen_pct - runner_pct,
                                  "ms": chosen_delta - runner_delta})
        if not ok:
            verdict["failures"].append(
                f"{chosen} degrades more than {runner_up} under iGPU load "
                f"({chosen_pct:+.0f}% vs {runner_pct:+.0f}%, "
                f"{chosen_delta:+.2f} vs {runner_delta:+.2f} ms)")
    else:
        lines.append((True, dim("headroom  not tested (no iGPU load)")))

    say("\n" + bold("policy checks"))
    for ok, line in lines:
        say(f"  {green('PASS') if ok else red('FAIL')}  {line}")

    if verdict["failures"]:
        say("\n" + red("FAIL") + " — accel.py's auto preset is not what this "
                                 "machine measures")
        for failure in verdict["failures"]:
            say(f"  {failure}")
        return finish(1)
    say("\n" + green("the auto preset matches what this machine measures"))
    return finish(0)


if __name__ == "__main__":
    raise SystemExit(main())
