"""
tof_stabilizer.py — ToF Lid-Shake Stabilization for Visual AI Game Engine.

When a laptop lid vibrates (from a fan, wind, or physical bump) the camera
shifts slightly, causing the entire ToF depth stream to wobble. This module
measures the ambient Z fluctuation over a short calibration window while the
user holds still, then suppresses fluctuations of that magnitude from all
future readings while letting real hand movement through unchanged.

How the correction works
------------------------
Calibration measures two things while the user holds still:

    z_baseline          mean resting depth (m)
    z_noise_amplitude   std-dev of the vibration around it (m)

The correction is a **noise gate around the baseline**, not a subtraction of
the baseline:

    delta = z_raw - z_baseline
    gate  = max(min_gate_m, gate_k * z_noise_amplitude)

    |delta| <= gate  ->  vibration only. Report the baseline; let the baseline
                         creep toward z_raw very slowly so genuine slow drift
                         is still tracked.
    |delta|  > gate  ->  real movement. Report it, minus the gate width so the
                         output is continuous at the gate edge, and re-seat the
                         baseline quickly toward the new resting depth.

Absolute depth is preserved: calibrating at 0.45 m and then reading 0.30 m
still reports ~0.30 m. (The previous implementation subtracted the mean
outright, which collapsed every reading to the 0.05 m clamp and destroyed the
punch/push signal entirely.)

Integration (via VisionPipeline)
---------------------------------
    # Start 3-second calibration (shows warning overlay on frame):
    pipeline.begin_stabilization(duration=3.0)

    # Start 5-second calibration:
    pipeline.begin_stabilization(duration=5.0)

    # Abort a calibration that is in progress (keeps any previous calibration):
    pipeline.cancel_stabilization()

    # Turn off correction:
    pipeline.disable_stabilization()

    # Read corrected depth from queue payload (transparent to consumers):
    data = ai_queue.get_nowait()
    depth = data["tof_z_m"]                # already corrected
    noise = data["stabilizer_noise_amp"]   # measured fluctuation amplitude (m)
    state = data["stabilizer_state"]       # "inactive" | "sampling" | "active"

Queue payload keys added
-------------------------
    stabilizer_state    : str   — "inactive" | "sampling" | "active"
    stabilizer_progress : float — 0.0 → 1.0 during sampling
    stabilizer_noise_amp: float — std-dev of Z during calibration (metres)
    stabilizer_gate     : float — current noise-gate width (metres)
    tof_z_raw           : float — uncorrected tof_z_m before correction
"""

import math
import threading
import time


class ToFStabilizer:
    """
    Calibrates and corrects ambient ToF Z wobble caused by physical device
    vibration (fan, wind, lid shake).

    States
    ------
    inactive  : No correction applied. Default on startup.
    sampling  : Collecting calibration samples for N seconds.
                User should hold device and hand still.
    active    : Baseline computed; gating every tof_z_m reading.

    Attributes
    ----------
    state               : str   — current state string
    z_baseline          : float — resting depth measured during calibration (m),
                                  slowly tracked afterwards
    z_noise_amplitude   : float — std-dev of Z noise measured (m), useful as
                                  a dynamic punch-threshold floor for consumer games
    z_offset            : float — correction applied to the most recent reading
                                  (z_raw - corrected, in metres). Telemetry only.
    last_error          : str | None — why the last calibration failed, if it did
    """

    STATE_INACTIVE: str = "inactive"
    STATE_SAMPLING: str = "sampling"
    STATE_ACTIVE:   str = "active"

    #: Calibration is rejected below this many valid samples — activating on
    #: one or two readings produces a meaningless baseline.
    MIN_SAMPLES: int = 8

    #: Depth floor, metres. Readings never report closer than this.
    MIN_DEPTH_M: float = 0.05

    #: Baseline tracking rates (per frame).
    _TRACK_ALPHA_IN_GATE:  float = 0.02   # slow — follows thermal/positional drift
    _TRACK_ALPHA_OUT_GATE: float = 0.15   # fast — re-seats after real movement

    def __init__(self, gate_k: float = 2.5, min_gate_m: float = 0.002) -> None:
        """
        Parameters
        ----------
        gate_k : float
            Noise gate width as a multiple of the measured std-dev. 2.5 covers
            ~99% of Gaussian vibration.
        min_gate_m : float
            Absolute floor for the gate (metres). Guards against an
            unrealistically small gate when calibration happens to be very
            quiet. 2 mm by default.
        """
        self.state:             str   = self.STATE_INACTIVE
        self.z_baseline:        float = 0.0
        self.z_noise_amplitude: float = 0.0
        self.z_offset:          float = 0.0
        self.last_error:        str | None = None

        self.gate_k:     float = max(0.0, float(gate_k))
        self.min_gate_m: float = max(0.0, float(min_gate_m))

        self._samples:    list[float] = []
        self._start_time: float = 0.0
        self._duration:   float = 3.0

        # begin()/cancel()/disable() arrive from the game thread while
        # feed()/tick()/correct() run on the pipeline thread. The GIL keeps
        # individual attribute writes safe, but not transitions: _finalise()
        # racing begin() could activate on the OLD sample set and silently
        # swallow the calibration the user just started. Reentrant because
        # feed() → tick() → _finalise() nests.
        self._lock = threading.RLock()

        # Snapshot of the last completed calibration, so cancelling a
        # recalibration restores it instead of leaving stabilization off.
        self._prev_calibration: tuple[float, float] | None = None

    # ── Read-only properties ───────────────────────────────────────────────────

    @property
    def progress(self) -> float:
        """
        Calibration progress in [0.0, 1.0].

        Returns
        -------
        float
            0.0 = not started / inactive,
            0–1 = fraction of sampling window elapsed,
            1.0 = sampling complete / active.
        """
        if self.state == self.STATE_ACTIVE:
            return 1.0
        if self.state == self.STATE_INACTIVE:
            return 0.0
        elapsed = time.time() - self._start_time
        return min(1.0, elapsed / self._duration)

    @property
    def time_remaining(self) -> float:
        """Seconds remaining in current calibration window, or 0.0 otherwise."""
        if self.state != self.STATE_SAMPLING:
            return 0.0
        return max(0.0, self._duration - (time.time() - self._start_time))

    @property
    def sample_count(self) -> int:
        """Number of valid Z samples collected so far."""
        return len(self._samples)

    @property
    def is_calibrated(self) -> bool:
        """True when a completed calibration is being applied."""
        return self.state == self.STATE_ACTIVE

    @property
    def noise_gate(self) -> float:
        """
        Current gate half-width in metres. Depth changes smaller than this are
        treated as vibration. Games can use it as a movement-threshold floor.
        """
        if self.state != self.STATE_ACTIVE:
            return 0.0
        return max(self.min_gate_m, self.gate_k * self.z_noise_amplitude)

    # ── Public API ─────────────────────────────────────────────────────────────

    def begin(self, duration: float = 3.0) -> None:
        """
        Start a fresh calibration run.

        Call this from the consumer game (e.g., on a keypress).
        VisionPipeline will show a warning overlay on frames automatically
        and feed raw ToF Z values here until the window elapses.

        Parameters
        ----------
        duration : float
            How many seconds to sample. 3.0 or 5.0 seconds recommended.
            Clamped to a minimum of 1.0 second.
        """
        requested = float(duration)
        with self._lock:
            if self.state == self.STATE_ACTIVE:
                self._prev_calibration = (self.z_baseline, self.z_noise_amplitude)
            self._samples    = []
            self._start_time = time.time()
            self._duration   = max(1.0, requested)
            self.z_baseline         = 0.0
            self.z_noise_amplitude  = 0.0
            self.z_offset           = 0.0
            self.last_error         = None
            self.state       = self.STATE_SAMPLING

        note = ""
        if requested < 1.0:
            note = f" (requested {requested:.2f}s, clamped to minimum)"
        print(
            f"[ToFStabilizer] Calibration started — "
            f"hold still for {self._duration:.0f}s{note}"
        )

    def feed(self, z_value: float) -> bool:
        """
        Feed one raw ToF Z sample during the calibration window.

        Called automatically by VisionPipeline every frame while
        state == STATE_SAMPLING. Do not call manually from consumer games.

        Parameters
        ----------
        z_value : float
            Raw, uncorrected ToF depth reading in metres.

        Returns
        -------
        bool
            True exactly once: the frame on which calibration completes.
            False every other frame.
        """
        with self._lock:
            if self.state != self.STATE_SAMPLING:
                return False

            # Discard zero / negative / NaN readings — an inactive or blind ToF
            # sensor reports 0.0, and those must not become calibration data.
            if math.isfinite(z_value) and z_value > 0.0:
                self._samples.append(float(z_value))

            return self.tick()

    def tick(self) -> bool:
        """
        Advance the calibration clock without supplying a sample.

        VisionPipeline calls this once per frame from the capture loop so that
        a calibration window always ends on time — including when no hand is
        in view and ``feed()`` is therefore never reached. Without it the
        stabilizer stays in ``sampling`` forever and the warning overlay never
        clears.

        Returns
        -------
        bool
            True exactly once: the frame on which calibration finishes
            (successfully or not). False otherwise.
        """
        with self._lock:
            if self.state != self.STATE_SAMPLING:
                return False
            if (time.time() - self._start_time) < self._duration:
                return False
            self._finalise()
            return True

    def correct(self, z_value: float) -> float:
        """
        Gate ambient vibration out of a live ToF Z reading.

        No-op when state is ``inactive`` or ``sampling``, or when the reading is
        invalid (<= 0 / NaN) — the raw value is returned untouched.

        Parameters
        ----------
        z_value : float
            Raw ToF depth reading in metres.

        Returns
        -------
        float
            Corrected depth reading in metres (clamped >= MIN_DEPTH_M).
            Absolute depth is preserved; only the vibration band is removed.
        """
        with self._lock:
            if self.state != self.STATE_ACTIVE:
                self.z_offset = 0.0
                return z_value

            if not math.isfinite(z_value) or z_value <= 0.0:
                self.z_offset = 0.0
                return z_value

            delta = z_value - self.z_baseline
            gate  = self.noise_gate

            if abs(delta) <= gate:
                # Vibration band: hold the reported depth steady at the baseline
                # and let the baseline creep toward the reading so slow, genuine
                # drift (thermal, posture) is still tracked.
                self.z_baseline += self._TRACK_ALPHA_IN_GATE * delta
                corrected = self.z_baseline
            else:
                # Real movement: subtract the gate width so the output is
                # continuous across the gate edge, then re-seat the baseline
                # toward the new resting depth so future vibration is measured
                # around it.
                excess = delta - math.copysign(gate, delta)
                corrected = self.z_baseline + excess
                self.z_baseline += self._TRACK_ALPHA_OUT_GATE * excess

            corrected = max(self.MIN_DEPTH_M, corrected)
            self.z_offset = z_value - corrected
            return corrected

    def cancel(self) -> None:
        """
        Abort a calibration that is in progress.

        Restores the previous completed calibration when there was one
        (the contract the module docstring has always promised), otherwise
        backs out to ``inactive``. Safe to call in any state.
        """
        with self._lock:
            if self.state != self.STATE_SAMPLING:
                return
            self._samples = []
            if self._prev_calibration is not None:
                self.z_baseline, self.z_noise_amplitude = self._prev_calibration
                self.state = self.STATE_ACTIVE
                print("[ToFStabilizer] Calibration cancelled — previous calibration restored.")
            else:
                self.state = self.STATE_INACTIVE
                print("[ToFStabilizer] Calibration cancelled.")

    def disable(self) -> None:
        """Turn off stabilization and clear all calibration data."""
        with self._lock:
            self.state             = self.STATE_INACTIVE
            self.z_baseline        = 0.0
            self.z_noise_amplitude = 0.0
            self.z_offset          = 0.0
            self._samples          = []
            self._prev_calibration = None
        print("[ToFStabilizer] Stabilization disabled.")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _finalise(self) -> None:
        """Compute baseline and noise amplitude from samples, then activate."""
        n = len(self._samples)

        if n < self.MIN_SAMPLES:
            # Not enough usable depth data — activating here would gate against
            # a meaningless baseline, so stay off and say why.
            self.last_error = (
                f"only {n} valid ToF sample(s) collected "
                f"(need {self.MIN_SAMPLES})"
            )
            self.state             = self.STATE_INACTIVE
            self.z_baseline        = 0.0
            self.z_noise_amplitude = 0.0
            self._samples          = []
            print(
                f"[ToFStabilizer] Calibration FAILED — {self.last_error}. "
                f"Is the ToF sensor enabled and a hand in view? "
                f"Stabilization left off."
            )
            return

        mu  = sum(self._samples) / n
        var = sum((s - mu) ** 2 for s in self._samples) / n

        self.z_baseline        = mu
        self.z_noise_amplitude = math.sqrt(var)
        self.z_offset          = 0.0
        self.last_error        = None
        self.state             = self.STATE_ACTIVE
        self._samples          = []

        print(
            f"[ToFStabilizer] Calibration complete — "
            f"samples={n}, "
            f"baseline={self.z_baseline:.4f} m, "
            f"noise_amp={self.z_noise_amplitude * 1000:.1f} mm, "
            f"gate=±{self.noise_gate * 1000:.1f} mm"
        )

    def __repr__(self) -> str:
        return (
            f"ToFStabilizer("
            f"state={self.state!r}, "
            f"baseline={self.z_baseline:.4f} m, "
            f"noise={self.z_noise_amplitude:.4f} m, "
            f"gate={self.noise_gate:.4f} m, "
            f"progress={self.progress:.0%})"
        )
