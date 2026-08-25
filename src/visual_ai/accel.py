"""
accel.py — which piece of silicon each network runs on.

Three devices are in play on an Intel Core Ultra machine, and they are good at
different things (measured on a Core Ultra 5 225H / Arc 130T / AI Boost NPU,
640x480 capture, one hand):

    hand tracking, end to end   MediaPipe CPU 19.4 ms   iGPU 2.0 ms   NPU 3.4 ms
    CPU cost of the same        1.0 core                2.3 cores     1.4 cores
    MODNet matting 640x352      ORT CPU     48.4 ms     iGPU 11.9 ms  NPU 42.7 ms

So the default policy is hands on the NPU — nearly iGPU speed, a third of the
CPU, and it leaves the iGPU free for the game's own drawing — and matting and
segmentation on the iGPU, which is 4x the CPU there and where the NPU's FP16
weights cost real accuracy on an alpha matte.

None of this is on unless asked for. `resolve` returns None when the preference
is "off" (the default), when OpenVINO is not installed, or when the requested
device is not present — the callers then keep their existing CPU path, and say
so on the HUD rather than silently running something else.
"""
from __future__ import annotations

import os

#: Environment keys the launcher sets from ``play <title> --accel ...``.
ENV_ACCEL = "VISUAL_AI_ACCEL"
ENV_HAND_DEVICE = "VISUAL_AI_HAND_DEVICE"
ENV_MATTE_DEVICE = "VISUAL_AI_MATTE_DEVICE"

#: What each ``--accel`` preset means for each consumer, in preference order.
#: "auto" is the measured policy above; the single-device presets are escape
#: hatches for benchmarking one part of the machine against another.
_PRESETS: dict[str, dict[str, tuple[str, ...]]] = {
    "off":  {"hand": (), "matte": ()},
    "auto": {"hand": ("NPU", "GPU"), "matte": ("GPU",)},
    "npu":  {"hand": ("NPU",), "matte": ("NPU",)},
    "gpu":  {"hand": ("GPU",), "matte": ("GPU",)},
    "cpu":  {"hand": ("CPU",), "matte": ("CPU",)},
}

PRESETS = tuple(_PRESETS)


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
    """The active ``--accel`` preset: the argument, then the environment, then off."""
    value = (explicit or os.environ.get(ENV_ACCEL) or "off").strip().lower()
    return value if value in _PRESETS else "off"


def resolve(consumer: str, explicit: str | None = None,
            accel: str | None = None) -> str | None:
    """
    The OpenVINO device for ``consumer`` ("hand" or "matte"), or None for
    "stay on the existing CPU path".

    `explicit` (a device id, or the matching ``VISUAL_AI_*_DEVICE`` variable)
    overrides the preset, including forcing a device on while the preset is
    off. An explicit device that is not present resolves to None: a request for
    an NPU on a machine without one is a fallback, not an error, but it must be
    a *visible* fallback — see `describe`.
    """
    env_key = {"hand": ENV_HAND_DEVICE, "matte": ENV_MATTE_DEVICE}.get(consumer)
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
    asked = (os.environ.get({"hand": ENV_HAND_DEVICE,
                             "matte": ENV_MATTE_DEVICE}.get(consumer, ""), "")
             or preset()).strip().lower()
    if asked in ("", "off", "none"):
        return fallback
    if not openvino_available():
        return f"{fallback} (openvino not installed)"
    return f"{fallback} ({asked} unavailable)"
