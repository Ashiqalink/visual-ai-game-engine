"""
accel.py — which piece of silicon each network runs on.

Three devices are in play on an Intel Core Ultra machine, and they are good at
different things (measured on a Core Ultra 5 225H / Arc 130T / AI Boost NPU,
640x480 capture, one hand):

    hand tracking, end to end   MediaPipe CPU 19.4 ms   iGPU 2.0 ms   NPU 3.4 ms
    CPU cost of the same        1.0 core                2.3 cores     1.4 cores
    face detection, end to end  MediaPipe CPU  5.7 ms   iGPU 1.4 ms   NPU 2.1 ms
    CPU cost of the same        1.0 core                1.2 cores     0.25 core
    MODNet matting 640x352      ORT CPU     48.4 ms     iGPU 11.9 ms  NPU 42.7 ms

So the default policy is hands *and* faces on the NPU — nearly iGPU speed, a
fraction of the CPU, and it leaves the iGPU free for the game's own drawing —
and matting and segmentation on the iGPU, which is 4x the CPU there.

Faces go to the NPU rather than the iGPU, which wins the row above, because
the row is one network running alone and neither of them ever does. Added to
a frame whose hands are already on the NPU (720p fixture, 60 frames,
2026-08-31), face detection costs +0.50 ms on the NPU, +1.52 ms on the iGPU
and +9.59 ms on MediaPipe's CPU graph. The NPU is a serial queue, but these
two networks are small enough that sharing it still beats crossing devices.

Matting stays on the iGPU for speed, not accuracy. Measured against the CPU's
f32 matte (640x352, 2026-08-31), the NPU's FP16 is the *closer* of the two
accelerators — MAE 2e-5 and max 0.019, against the iGPU's 3e-5 and 0.048 —
and simply half its speed. An earlier note here blamed FP16 for an accuracy
cost on the alpha matte; that was wrong.

That policy is the default: with nothing asked for, the preset is "auto". It
costs nothing to try — `resolve` returns None when the preference is "off", when
OpenVINO is not installed, or when the requested device is not present, and the
callers then keep their existing CPU path. What it is *not* is silent: the
device actually in use is on the HUD either way, and `describe` says why when
an explicitly requested device was not the one that won.
"""
from __future__ import annotations

import os

#: Environment keys the launcher sets from ``play <title> --accel ...``.
ENV_ACCEL = "VISUAL_AI_ACCEL"
ENV_HAND_DEVICE = "VISUAL_AI_HAND_DEVICE"
ENV_MATTE_DEVICE = "VISUAL_AI_MATTE_DEVICE"
ENV_FACE_DEVICE = "VISUAL_AI_FACE_DEVICE"

#: What each ``--accel`` preset means for each consumer, in preference order.
#: "auto" is the measured policy above; the single-device presets are escape
#: hatches for benchmarking one part of the machine against another.
_PRESETS: dict[str, dict[str, tuple[str, ...]]] = {
    "off":  {"hand": (), "matte": (), "face": ()},
    "auto": {"hand": ("NPU", "GPU"), "matte": ("GPU",), "face": ("NPU", "GPU")},
    "npu":  {"hand": ("NPU",), "matte": ("NPU",), "face": ("NPU",)},
    "gpu":  {"hand": ("GPU",), "matte": ("GPU",), "face": ("GPU",)},
    "cpu":  {"hand": ("CPU",), "matte": ("CPU",), "face": ("CPU",)},
}

PRESETS = tuple(_PRESETS)

#: The preset when nothing asked for one. "auto" rather than "off" because the
#: CPU path costs 19.4 ms of a 16.7 ms frame budget and every fallback out of
#: "auto" lands back on the CPU code when no accelerator is reachable.  When a
#: GPU is present but the NPU is not, hands run on the iGPU through OpenVINO
#: with a looser tracking gate (0.45 vs MediaPipe's 0.65); this is faster and
#: intentional for iGPUs, but not a no-op.  dGPU behavior at 0.45 is untested.
#: A typo'd preset still falls back to "off" — that one is a mistake, and
#: running it as "auto" would hide it.
DEFAULT_PRESET = "auto"


def openvino_available() -> bool:
    """Is the OpenVINO runtime importable at all?"""
    try:
        import openvino  # noqa: F401
    except Exception:
        return False
    return True


def available_devices() -> list[str]:
    """
    Devices OpenVINO can see, e.g. ``["CPU", "GPU", "NPU"]``.

    Empty when OpenVINO is missing or its runtime fails to enumerate — a
    machine without the NPU driver installed reports no NPU here rather than
    failing later inside ``compile_model``.
    """
    try:
        import openvino as ov

        return list(ov.Core().available_devices)
    except Exception:
        return []


def device_name(device: str) -> str:
    """The device's full product name, for a HUD line. Falls back to the id."""
    try:
        import openvino as ov

        return str(ov.Core().get_property(device, "FULL_DEVICE_NAME"))
    except Exception:
        return device


def preset(explicit: str | None = None) -> str:
    """The active ``--accel`` preset: the argument, then the environment, then
    ``DEFAULT_PRESET``."""
    value = (explicit or os.environ.get(ENV_ACCEL) or DEFAULT_PRESET).strip().lower()
    return value if value in _PRESETS else "off"


def resolve(consumer: str, explicit: str | None = None,
            accel: str | None = None) -> str | None:
    """
    The OpenVINO device for ``consumer`` ("hand", "face" or "matte"), or None for
    "stay on the existing CPU path".

    `explicit` (a device id, or the matching ``VISUAL_AI_*_DEVICE`` variable)
    overrides the preset, including forcing a device on while the preset is
    off. An explicit device that is not present resolves to None: a request for
    an NPU on a machine without one is a fallback, not an error, but it must be
    a *visible* fallback — see `describe`.
    """
    env_key = {"hand": ENV_HAND_DEVICE, "matte": ENV_MATTE_DEVICE,
               "face": ENV_FACE_DEVICE}.get(consumer)
    wanted = (explicit or (os.environ.get(env_key) if env_key else None) or "").strip().upper()
    if wanted in ("", "AUTO"):
        candidates = _PRESETS[preset(accel)].get(consumer, ())
    elif wanted in ("OFF", "NONE"):
        return None
    else:
        candidates = (wanted,)
    if not candidates:
        return None
    present = available_devices()
    for candidate in candidates:
        if candidate in present:
            return candidate
    return None


def describe(consumer: str, resolved: str | None, fallback: str) -> str:
    """
    One short string naming what is actually running, for the HUD.

    Says why when the answer is "not what you asked for", because a silent
    fallback to CPU is indistinguishable in play from an accelerator that is
    working — just slower.
    """
    if resolved:
        return resolved
    # Deliberately the raw environment, not `preset()`: the default is "auto"
    # now, so going through `preset()` would append "(auto unavailable)" to
    # every CPU-only machine's HUD forever - including every frozen .exe, which
    # excludes OpenVINO on purpose. This suffix is for a request that lost, and
    # nobody requested the default.
    device_key = {"hand": ENV_HAND_DEVICE,
                  "matte": ENV_MATTE_DEVICE,
                  "face": ENV_FACE_DEVICE}.get(consumer, "")
    asked = os.environ.get(device_key, "").strip().lower()
    if not asked:
        asked = os.environ.get(ENV_ACCEL, "").strip().lower()
        # A typo'd preset (e.g. "aut") is silently normalized to "off" by
        # preset(); don't report it as if the user asked for something.
        if asked not in _PRESETS:
            asked = ""
    if asked in ("", "off", "none"):
        return fallback
    if not openvino_available():
        return f"{fallback} (openvino not installed)"
    return f"{fallback} ({asked} unavailable)"
