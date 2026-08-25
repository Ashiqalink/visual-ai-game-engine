"""Depth producers.

The pipeline has always been able to *consume* depth: `sample_depth`
medians a patch around the fingertip, rescales sensor pixels into RGB frame
space, and rejects no-return zeros; `_build_depth_grid` resizes a whole depth
map into the payload. What never existed was anything that produced one.
`depth_active` and `depth_map` were assigned in `__init__` and nowhere else, so
the "ToF IR Hardware" branch was unreachable and every depth number in every
payload came from `0.45 + lm_z * 0.6` -- a MediaPipe landmark wearing a depth
sensor's label.

This module is the producer. A `DepthSource` yields frames of uint16
millimetres, 0 meaning no return, which is the convention the consumer side
already expects and what essentially every ToF sensor reports natively.

    source = open_depth_source("auto")      # first backend that opens
    stream = DepthStream(source); stream.start()
    depth  = stream.latest()                # newest frame, or None

Backends, and what each is worth:

    SyntheticDepthSource   a moving hand-shaped blob over a back wall. No
                           hardware, fully deterministic, and the only way to
                           exercise the ToF path in tests -- which is exactly
                           what the old landmark hack pretended to be, except
                           this one is honest about it and never reaches a game
                           unless asked for by name.
    ReplayDepthSource      plays back frames recorded with DepthRecorder. This
                           is how a real sensor's data gets into a test suite
                           on a machine that does not have the sensor.
    OpenNI2DepthSource     OpenCV's built-in OpenNI2 capture -- Orbbec Astra,
                           Structure, Xtion, and most OpenNI-class ToF units.
                           Needs an OpenCV built with OpenNI2 support.
    UVCDepthSource         a depth camera exposed as a plain UVC video device
                           streaming 16-bit Y16, which is how many low-cost ToF
                           modules present themselves.
    RealSenseDepthSource   Intel RealSense via pyrealsense2, if installed.

Only the first two have been run against real data here: this machine has one
RGB webcam and no depth device, so the three hardware backends are written
from their documented APIs and are unverified. They are structured so that a
failure to open is reported and skipped rather than raised -- `open_depth_source`
falls through to the next candidate, and the pipeline stays on its RGB path.
`DepthSource.last_error` carries the reason, and `probe_depth_sources()`
prints the whole table for `play doctor`.
"""

import os
import threading
import time

import numpy as np

from visual_ai.capture import default_backend

try:
    import cv2
except ImportError:                                   # pragma: no cover
    cv2 = None


# Frames are uint16 millimetres, 0 = no return. Anything outside this is
# almost certainly a decode error rather than a real reading.
MIN_VALID_MM = 100        # 10 cm; closer than any ToF sensor resolves
MAX_VALID_MM = 10000      # 10 m


class DepthSourceError(RuntimeError):
    pass


class DepthSource:
    """One depth producer. Subclasses implement `_open`, `_read`, `_close`."""

    #: Shown in payloads and diagnostics.
    name = "depth"

    #: True for backends that synthesise rather than measure. The pipeline
    #: refuses to label these "ToF IR Hardware", because a payload that claims
    #: a sensor it does not have is how the old code misled everyone.
    synthetic = False

    def __init__(self) -> None:
        self.is_open = False
        self.last_error = ""
        self.frames_read = 0
        self.resolution = (0, 0)     # (w, h), filled on first frame
        self._opened_at = 0.0

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> bool:
        """Try to open. Returns True on success; never raises."""
        if self.is_open:
            return True
        try:
            self._open()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.is_open = False
            return False
        self.is_open = True
        self._opened_at = time.perf_counter()
        return True

    def read(self) -> np.ndarray | None:
        """Newest depth frame as uint16 mm, or None if unavailable."""
        if not self.is_open:
            return None
        try:
            frame = self._read()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        if frame is None:
            return None
        frame = sanitize(frame)
        if frame is None:
            self.last_error = "frame failed validation"
            return None
        self.frames_read += 1
        self.resolution = (frame.shape[1], frame.shape[0])
        return frame

    def close(self) -> None:
        if not self.is_open:
            return
        try:
            self._close()
        except Exception as exc:                       # pragma: no cover
            self.last_error = f"{type(exc).__name__}: {exc}"
        self.is_open = False

    @property
    def fps(self) -> float:
        if not self.frames_read or not self._opened_at:
            return 0.0
        return self.frames_read / max(1e-6, time.perf_counter() - self._opened_at)

    # -- subclass hooks ----------------------------------------------------

    def _open(self) -> None:
        raise NotImplementedError

    def _read(self) -> np.ndarray | None:
        raise NotImplementedError

    def _close(self) -> None:
        pass


def sanitize(frame: np.ndarray | None) -> np.ndarray | None:
    """Coerce a raw sensor frame to uint16 mm, or None if it is not depth.

    Sensors disagree about dtype and units far more than their datasheets
    suggest: float metres, int32 mm, and 16-bit mm all occur, and a
    misconfigured UVC stream hands back an 8-bit BGR image that would sail
    through a shape check. Everything funnels through here so the rest of the
    engine only ever sees one representation.
    """
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 3:
        # A three-channel "depth" frame means the capture handed back colour.
        if arr.shape[2] != 1:
            return None
        arr = arr[:, :, 0]
    if arr.ndim != 2 or arr.size == 0:
        return None

    if arr.dtype == np.uint16:
        out = arr
    elif np.issubdtype(arr.dtype, np.floating):
        # Float depth is metres by convention; anything above 100 is already
        # millimetres from a sensor that just used the wrong dtype.
        finite = arr[np.isfinite(arr)]
        peak = float(finite.max()) if finite.size else 0.0
        scale = 1000.0 if peak <= 100.0 else 1.0
        out = np.nan_to_num(arr * scale, nan=0.0, posinf=0.0, neginf=0.0)
        out = out.astype(np.uint16)
    elif np.issubdtype(arr.dtype, np.integer):
        out = np.clip(arr, 0, 65535).astype(np.uint16)
    else:
        return None

    # Out-of-range readings become 0 -- "no return" -- which every consumer
    # here already handles, rather than a plausible-looking wrong distance.
    out = out.copy()
    bad = (out < MIN_VALID_MM) | (out > MAX_VALID_MM)
    out[bad] = 0
    return out


# ── Synthetic ────────────────────────────────────────────────────────────────

class SyntheticDepthSource(DepthSource):
    """A hand-shaped blob orbiting in front of a wall. No hardware needed.

    Deterministic: frame N is a pure function of N, so a test can assert on
    exact distances. Motion is expressed per second and driven by the frame
    counter at the declared fps, so it does not change with how fast the
    consumer happens to pull.
    """

    name = "synthetic"
    synthetic = True

    def __init__(self, width: int = 320, height: int = 240, fps: float = 30.0,
                 wall_m: float = 2.2, hand_near_m: float = 0.30,
                 hand_far_m: float = 0.80, radius_px: int = 45) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.declared_fps = float(fps)
        self.wall_mm = int(wall_m * 1000)
        self.near_mm = int(hand_near_m * 1000)
        self.far_mm = int(hand_far_m * 1000)
        self.radius_px = int(radius_px)
        self._n = 0
        self._next_at = 0.0

    def _open(self) -> None:
        self._n = 0
        self._next_at = time.perf_counter()

    def _read(self) -> np.ndarray:
        # Pace to the declared rate. A source with nothing to wait for will
        # otherwise free-run at thousands of frames a second and burn a core
        # producing frames no consumer asked for -- real sensors block in
        # their own read, and this one has to imitate that too.
        now = time.perf_counter()
        if self._next_at > now:
            time.sleep(self._next_at - now)
        self._next_at = max(time.perf_counter(), self._next_at + 1.0 / self.declared_fps)

        t = self._n / self.declared_fps
        self._n += 1

        frame = np.full((self.height, self.width), self.wall_mm, dtype=np.uint16)

        # Orbit in X/Y, breathe in Z, all at different rates so no two frames
        # in a short window are alike.
        cx = int(self.width / 2 + np.cos(t * 1.7) * self.width * 0.22)
        cy = int(self.height / 2 + np.sin(t * 1.1) * self.height * 0.18)
        span = self.far_mm - self.near_mm
        hand_mm = int(self.near_mm + span * (0.5 + 0.5 * np.sin(t * 0.9)))

        ys, xs = np.ogrid[:self.height, :self.width]
        blob = (xs - cx) ** 2 + (ys - cy) ** 2 <= self.radius_px ** 2
        frame[blob] = hand_mm

        # A ToF sensor drops returns at object edges and on dark surfaces;
        # consumers that cannot cope with holes will break in the field, so
        # they should break here first.
        edge = ((xs - cx) ** 2 + (ys - cy) ** 2 <= (self.radius_px + 2) ** 2) & ~blob
        frame[edge] = 0
        return frame


# ── Replay ───────────────────────────────────────────────────────────────────

class ReplayDepthSource(DepthSource):
    """Play back frames saved by `DepthRecorder`.

    This is what makes hardware testable without hardware: record a minute on
    the machine that has the sensor, commit the file, and every other machine
    can run the ToF path against real sensor noise, real dropouts and real
    resolution.
    """

    name = "replay"
    synthetic = True     # real data, but not a live sensor

    def __init__(self, path: str | os.PathLike, loop: bool = True,
                 fps: float = 30.0) -> None:
        super().__init__()
        self.path = str(path)
        self.loop = bool(loop)
        self.declared_fps = float(fps)
        self._frames = None
        self._i = 0
        self._next_at = 0.0

    def _open(self) -> None:
        if not os.path.exists(self.path):
            raise DepthSourceError(f"no such recording: {self.path}")
        data = np.load(self.path)
        frames = data["frames"] if hasattr(data, "keys") else data
        if frames.ndim != 3 or frames.shape[0] == 0:
            raise DepthSourceError(
                f"expected a (frames, h, w) stack, got {frames.shape}")
        self._frames = frames
        self._i = 0
        self._next_at = time.perf_counter()
        self.name = f"replay:{os.path.basename(self.path)}"

    def _read(self) -> np.ndarray | None:
        # Same pacing as the synthetic source: a recording played back as fast
        # as memory allows is not a rehearsal of the sensor it came from.
        now = time.perf_counter()
        if self._next_at > now:
            time.sleep(self._next_at - now)
        self._next_at = max(time.perf_counter(), self._next_at + 1.0 / self.declared_fps)

        if self._frames is None or self._i >= len(self._frames):
            if not self.loop or self._frames is None:
                return None
            self._i = 0
        frame = self._frames[self._i]
        self._i += 1
        return frame

    def _close(self) -> None:
        self._frames = None


class DepthRecorder:
    """Collect depth frames and write them where ReplayDepthSource can read.

    Bounded on purpose: depth is 2 bytes per pixel per frame, so an unbounded
    recorder on a 640x480 sensor eats 18 MB a second.
    """

    def __init__(self, max_frames: int = 900) -> None:
        self.max_frames = int(max_frames)
        self.frames = []

    def feed(self, frame: np.ndarray | None) -> bool:
        if frame is None or len(self.frames) >= self.max_frames:
            return False
        self.frames.append(np.asarray(frame, dtype=np.uint16))
        return True

    @property
    def full(self) -> bool:
        return len(self.frames) >= self.max_frames

    def save(self, path: str | os.PathLike) -> str | os.PathLike:
        if not self.frames:
            raise DepthSourceError("nothing recorded")
        stack = np.stack(self.frames)
        np.savez_compressed(path, frames=stack)
        return path


# ── Hardware backends ────────────────────────────────────────────────────────

class OpenNI2DepthSource(DepthSource):
    """OpenNI2 through OpenCV's own capture backend.

    Covers Orbbec Astra, Occipital Structure, Asus Xtion and most OpenNI-class
    units. Requires an OpenCV built with OpenNI2 support -- the pip
    `opencv-python` wheels are not, so this will usually report that it cannot
    open, which is why it is a candidate rather than the default.
    """

    name = "openni2"

    def __init__(self, index: int = 0) -> None:
        super().__init__()
        self.index = int(index)
        self._cap = None

    def _open(self) -> None:
        if cv2 is None:
            raise DepthSourceError("OpenCV is not available")
        cap = cv2.VideoCapture(self.index + cv2.CAP_OPENNI2)
        if not cap.isOpened():
            cap.release()
            raise DepthSourceError(
                "OpenNI2 capture would not open (OpenCV is probably built "
                "without OpenNI2 support)")
        self._cap = cap

    def _read(self) -> np.ndarray | None:
        if self._cap is None or not self._cap.grab():
            return None
        # CAP_OPENNI_DEPTH_MAP is 16-bit millimetres, which is already the
        # representation the rest of this module uses.
        ok, depth = self._cap.retrieve(flag=cv2.CAP_OPENNI_DEPTH_MAP)
        return depth if ok else None

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class UVCDepthSource(DepthSource):
    """A depth camera presenting as a UVC device streaming 16-bit Y16.

    Many inexpensive ToF modules do exactly this. Two settings decide whether
    it works: CONVERT_RGB off, or OpenCV helpfully converts the 16-bit depth
    into an 8-bit colour image and the depth is gone before you see it; and
    the Y16 fourcc, without which the driver may default to a colour format.
    """

    name = "uvc-y16"

    def __init__(self, index: int = 1, width: int | None = None,
                 height: int | None = None, fourcc: str = "Y16 ") -> None:
        super().__init__()
        self.index = int(index)
        self.req_size = (width, height)
        self.fourcc = fourcc
        self._cap = None

    def _open(self) -> None:
        if cv2 is None:
            raise DepthSourceError("OpenCV is not available")
        # Platform backend, not DirectShow: this has to open on macOS and
        # Linux too, where CAP_DSHOW names an API that does not exist.
        cap = cv2.VideoCapture(self.index, default_backend())
        if not cap.isOpened():
            cap.release()
            raise DepthSourceError(f"video index {self.index} would not open")
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.req_size[0]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_size[1])

        ok, probe = cap.read()
        if not ok or probe is None:
            cap.release()
            raise DepthSourceError(f"video index {self.index} gave no frame")
        if sanitize(probe) is None:
            cap.release()
            raise DepthSourceError(
                f"video index {self.index} is not a 16-bit depth stream "
                f"(got {probe.dtype} {probe.shape}) -- this is what an ordinary "
                f"webcam looks like here")
        self._cap = cap
        self.name = f"uvc-y16:{self.index}"

    def _read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RealSenseDepthSource(DepthSource):
    """Intel RealSense via pyrealsense2, if it is installed.

    The depth scale is read from the device rather than assumed: D400-series
    parts default to 0.001 m per unit but it is configurable, and guessing it
    puts every distance out by a constant factor.
    """

    name = "realsense"

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        super().__init__()
        self.req = (int(width), int(height), int(fps))
        self._pipe = None
        self._scale_to_mm = 1.0

    def _open(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise DepthSourceError("pyrealsense2 is not installed") from exc

        w, h, fps = self.req
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        profile = pipe.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        # Device units are metres per integer step; convert to mm.
        self._scale_to_mm = float(depth_sensor.get_depth_scale()) * 1000.0
        self._pipe = pipe

    def _read(self) -> np.ndarray | None:
        if self._pipe is None:
            return None
        frames = self._pipe.wait_for_frames(timeout_ms=1000)
        depth = frames.get_depth_frame()
        if not depth:
            return None
        raw = np.asanyarray(depth.get_data())
        if abs(self._scale_to_mm - 1.0) < 1e-6:
            return raw
        return (raw.astype(np.float32) * self._scale_to_mm)

    def _close(self) -> None:
        if self._pipe is not None:
            self._pipe.stop()
            self._pipe = None


# ── Streaming ────────────────────────────────────────────────────────────────

class DepthStream(threading.Thread):
    """Pull a source on its own thread and keep only the newest frame.

    Depth and colour run at their own rates and neither should wait for the
    other: a blocking `wait_for_frames` inside the capture loop would pace the
    RGB path off the depth sensor. Only the latest frame is kept, for the same
    reason the payload queue has maxsize 1 -- a stale depth frame is worse than
    no depth frame, because it looks current.
    """

    def __init__(self, source: DepthSource, poll_interval: float = 0.0) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.poll_interval = poll_interval
        self._latest = None
        self._lock = threading.Lock()
        # Not `_stop`: threading.Thread already owns that name internally,
        # and shadowing it breaks join().
        self._stop_event = threading.Event()
        self.recorder = None
        self.last_frame_at = 0.0

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def age_s(self) -> float:
        """Seconds since the newest frame arrived; inf if none ever did."""
        if not self.last_frame_at:
            return float("inf")
        return time.perf_counter() - self.last_frame_at

    def run(self) -> None:
        if not self.source.is_open and not self.source.open():
            return
        while not self._stop_event.is_set():
            frame = self.source.read()
            if frame is None:
                # A source that has stopped producing should not spin a core.
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = frame
                self.last_frame_at = time.perf_counter()
            if self.recorder is not None:
                self.recorder.feed(frame)
            if self.poll_interval:
                time.sleep(self.poll_interval)
        self.source.close()

    def stop(self, join_timeout: float = 1.0) -> None:
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=join_timeout)


# ── Monocular estimation ─────────────────────────────────────────────────────

#: Path to the depth network's ONNX file. There is no pinned download URL the
#: way `matting.py` has one for MODNet: FastDepth ships as PyTorch weights from
#: MIT, and the ONNX exports in circulation are third-party with no digest
#: worth trusting. So this backend never downloads -- you point it at a file.
DEPTH_ONNX_ENV_VAR = "VISUAL_AI_DEPTH_ONNX"

#: Metres per output unit. FastDepth is trained on NYU Depth v2, whose targets
#: are metres, so 1.0 is right for a faithful export. An export that baked in
#: its own normalisation needs this corrected, which is why it is a knob and
#: not a constant.
DEPTH_UNIT_ENV_VAR = "VISUAL_AI_DEPTH_SCALE"


class MonocularDepthSource(DepthSource):
    """Depth *estimated* from the ordinary RGB frame by a small CNN.

    Every other backend in this module reports what a sensor measured. This
    one reports what a network guessed from a colour image, and the difference
    matters more than the identical return type suggests: a monocular estimate
    has no way to know absolute scale from first principles, inherits whatever
    indoor bias its training set had, and will hallucinate a confident surface
    where a ToF unit would honestly return 0. It is therefore marked
    ``synthetic`` and kept out of ``AUTO_CANDIDATES`` -- "auto" means hardware,
    and a payload must never claim a sensor this machine does not have.

    What it *is* good for: this repo's RGB path currently fabricates depth as
    ``0.45 + lm_z * 0.6``, a MediaPipe landmark wearing a depth sensor's label.
    A real network's estimate is a strict improvement on that, and unlike the
    three hardware backends it can actually run on a machine with one webcam.

    Unlike a sensor, it has no frame source of its own -- it must not open a
    second capture and fight the pipeline for the camera. Frames are pushed in::

        source = MonocularDepthSource(model_path="fastdepth.onnx")
        source.open()
        source.submit(bgr_frame)          # from the capture already running
        depth = source.read()             # uint16 mm, or None if nothing new

    ``_read`` consumes the submitted frame, so a `DepthStream` polling faster
    than frames arrive gets None rather than the same estimate counted twice.
    """

    name = "monocular"

    # Inferred, not measured. See the class docstring -- this flag is the whole
    # reason the pipeline will not label this "ToF IR Hardware".
    synthetic = True

    #: FastDepth's native input. Feeding it the full capture frame wastes time
    #: on detail the first conv layer immediately throws away.
    DEFAULT_INPUT = (224, 224)

    def __init__(self, model_path: str | None = None,
                 providers: list[str] | None = None,
                 unit_metres: float | None = None) -> None:
        super().__init__()
        self.model_path = model_path or os.environ.get(DEPTH_ONNX_ENV_VAR, "")
        self.providers = providers
        if unit_metres is None:
            raw = os.environ.get(DEPTH_UNIT_ENV_VAR, "")
            unit_metres = float(raw) if raw else 1.0
        self.unit_metres = float(unit_metres)
        self._session = None
        self._input_name = ""
        self._input_size = self.DEFAULT_INPUT
        self._pending: np.ndarray | None = None
        #: Inference-only milliseconds for the last frame, for the HUD and the
        #: bench. Excludes resize and sanitize, which are measured separately.
        self.last_infer_ms = 0.0

    def _open(self) -> None:
        if not self.model_path:
            raise DepthSourceError(
                f"no depth model given -- pass model_path= or set "
                f"{DEPTH_ONNX_ENV_VAR} to a FastDepth ONNX file")
        if not os.path.isfile(self.model_path):
            raise DepthSourceError(f"no such file: {self.model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DepthSourceError(
                "onnxruntime is not installed, so monocular depth is "
                "unavailable (pip install onnxruntime)") from exc
        if cv2 is None:
            raise DepthSourceError("OpenCV is not available")

        providers = self.providers
        if providers is None:
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in available] or available

        session = ort.InferenceSession(self.model_path, providers=providers)
        spec = session.get_inputs()[0]
        if len(spec.shape) != 4:
            raise DepthSourceError(
                f"expected a 4-D NCHW input, got {spec.shape} -- is this "
                f"really a depth network?")

        # Fixed-shape exports fail inside onnxruntime with an opaque rank error
        # if fed anything else, so honour the declared size when there is one.
        height, width = spec.shape[2], spec.shape[3]
        if isinstance(height, int) and isinstance(width, int):
            self._input_size = (height, width)
        self._session = session
        self._input_name = spec.name
        self.name = f"monocular:{os.path.basename(self.model_path)}"

    def submit(self, frame: np.ndarray | None) -> None:
        """Hand over the newest colour frame. Cheap: no inference happens here."""
        self._pending = frame

    def _read(self) -> np.ndarray | None:
        frame = self._pending
        self._pending = None          # consume, so a fast poller gets None
        if frame is None or self._session is None:
            return None

        tensor = self._preprocess(frame)
        started = time.perf_counter()
        output = self._session.run(None, {self._input_name: tensor})[0]
        self.last_infer_ms = (time.perf_counter() - started) * 1000.0

        depth = np.squeeze(np.asarray(output)).astype(np.float32)
        if depth.ndim != 2:
            raise DepthSourceError(
                f"expected a single-channel depth map, got shape "
                f"{np.asarray(output).shape}")

        # Returned at the network's own resolution, *not* the frame's. A real
        # ToF unit's map rarely matches the RGB frame either, and `sample_depth`
        # already rescales from depth-map space into frame space. Upscaling here
        # would cost more than the inference (measured: 7.7ms at 640x360, 42ms
        # at 720p) only for the consumer to resample it straight back down.
        # Then units -> mm, cast here rather than handed to `sanitize` as a
        # float. Its float branch has to *guess* whether a float frame is metres
        # or millimetres, and guesses "metres" whenever the peak is under 100 --
        # so a genuinely close-range or small-unit map would be multiplied by
        # 1000 a second time. This backend knows its own units; nothing is
        # gained by making the sanitiser infer them. `sanitize` still zeroes
        # anything outside MIN/MAX_VALID_MM, which is what turns the network's
        # confident nonsense at the extremes into an honest "no return".
        millimetres = depth * (self.unit_metres * 1000.0)
        return np.clip(millimetres, 0.0, 65535.0).astype(np.uint16)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR uint8 -> NCHW float32, ImageNet-normalised at the model's size."""
        # Shrink first, flip channels second. The other order hands cv2.resize
        # a negative-strided view of the full frame, which it cannot walk
        # linearly -- that alone was most of a 7.2ms preprocess at 640x360.
        height, width = self._input_size
        scaled = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if scaled.ndim == 3:
            scaled = scaled[:, :, ::-1]

        # x/255, then (x - mean)/std, folded into one multiply and one subtract
        # against precomputed per-channel constants, in OpenCV rather than
        # NumPy. Four full float passes become two SIMD ones: 2.3ms -> 0.28ms,
        # agreeing with the naive form to 5e-7. FastDepth's MobileNet-V1
        # encoder carries ImageNet statistics, hence these constants.
        normalized = np.empty((height, width, 3), dtype=np.float32)
        cv2.multiply(scaled, _IMAGENET_SCALE, dst=normalized, dtype=cv2.CV_32F)
        cv2.subtract(normalized, _IMAGENET_OFFSET, dst=normalized)
        return np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[None])

    def _close(self) -> None:
        self._session = None
        self._pending = None


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Folded forms of x/255 then (x - mean)/std, so `_preprocess` runs two passes
# instead of four. Kept beside the constants they come from.
_IMAGENET_SCALE = (1.0 / (255.0 * _IMAGENET_STD)).astype(np.float32)
_IMAGENET_OFFSET = (_IMAGENET_MEAN / _IMAGENET_STD).astype(np.float32)


# ── Discovery ────────────────────────────────────────────────────────────────

#: Tried in order by "auto". Hardware first, then nothing: "auto" must never
#: silently fall back to synthetic depth, or a game would report a sensor it
#: does not have -- the exact failure this module exists to end.
AUTO_CANDIDATES = (
    ("realsense", lambda: RealSenseDepthSource()),
    ("openni2", lambda: OpenNI2DepthSource()),
    ("uvc-y16", lambda: UVCDepthSource(index=1)),
)


def open_depth_source(spec: str | DepthSource | None = "auto",
                      quiet: bool = True) -> DepthSource | None:
    """Build and open a source from a spec string. Returns None if none opens.

    Specs::

        "auto"            first hardware backend that opens, else None
        "none" / ""       no depth
        "synthetic"       the built-in blob, for development and tests
        "replay:PATH"     a recording made by DepthRecorder
        "openni2[:IDX]"   OpenNI2 through OpenCV
        "uvc[:IDX]"       16-bit Y16 UVC device (default index 1)
        "realsense"       pyrealsense2
        "monocular[:ONNX]"  depth estimated from RGB; needs frames pushed in
                            with `submit()`, and is never chosen by "auto"
    """
    if spec is None:
        return None
    if isinstance(spec, DepthSource):
        return spec if spec.open() else None

    raw_spec = str(spec).strip()
    spec = raw_spec.lower()
    if spec in ("", "none", "off", "false"):
        return None

    if spec == "auto":
        for label, build in AUTO_CANDIDATES:
            source = build()
            if source.open():
                return source
            if not quiet:
                print(f"[DepthSource] {label}: {source.last_error}")
        return None

    # Split the *raw* spec, not the lowercased one: every argument up to now
    # was a digit or a path OpenCV opens case-insensitively, but a model path
    # on a case-sensitive filesystem does not survive .lower().
    head, _, arg = raw_spec.partition(":")
    head = head.lower()
    if head == "synthetic":
        source = SyntheticDepthSource()
    elif head == "replay":
        source = ReplayDepthSource(arg)
    elif head == "openni2":
        source = OpenNI2DepthSource(int(arg) if arg else 0)
    elif head == "uvc":
        source = UVCDepthSource(int(arg) if arg else 1)
    elif head == "realsense":
        source = RealSenseDepthSource()
    elif head == "monocular":
        # "monocular:C:\path\to.onnx" -- the drive letter's colon means the
        # single partition above already split it, so rejoin what came after.
        source = MonocularDepthSource(model_path=arg or None)
    else:
        raise ValueError(f"unknown depth source spec: {spec!r}")

    if not source.open():
        if not quiet:
            print(f"[DepthSource] {spec}: {source.last_error}")
        return None
    return source


def probe_depth_sources() -> list[tuple[str, bool, str]]:
    """Try every hardware backend and report. For `play doctor`.

    Returns a list of (name, ok, detail). Opens and immediately closes, so it
    is safe to call while nothing else is using the sensor -- and unsafe while
    something is, which is true of every capture device.
    """
    results = []
    for label, build in AUTO_CANDIDATES:
        source = build()
        ok = source.open()
        detail = ""
        if ok:
            frame = source.read()
            if frame is None:
                ok, detail = False, "opened but produced no frame"
            else:
                h, w = frame.shape
                valid = int(np.count_nonzero(frame))
                detail = (f"{w}x{h}, {100.0 * valid / frame.size:.0f}% valid "
                          f"returns")
            source.close()
        else:
            detail = source.last_error
        results.append((label, ok, detail))
    return results
