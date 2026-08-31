"""
bench_npu_zoo.py — every `visual_ai.openvino_zoo` wrapper, plus hands, on
every device: compile time, call time, CPU load, and agreement with the CPU.

    python benchmarks/bench_npu_zoo.py [--devices NPU,GPU,CPU] [--only yolo,pose]
                                       [--camera 0 --seconds 10] [--json out.json]

Offline (the default) the input is `fixtures/hand_motion.mp4` at 640x480, as
`bench_hand_backend.py` uses it. Each wrapper is built on each device and run
over the clip; the CPU run is the reference the other devices are compared
against — top-box IoU for detectors, mean landmark distance in pixels for the
pose nets, top-1 match for the classifiers. A wrapper that fell back reports
the device it actually runs on.

`--camera N` opens that webcam at 640x480 and runs each wrapper live for
`--seconds`, reporting end-to-end fps, capture and call time, and what it
found — the only mode that exercises the capture path, and still not a check
that the results are *right*; that takes a person watching the overlay.

Exit 0 when a device is missing (it says which), 2 when weights are not in
the model cache (it prints the fetch commands; this bench never downloads),
1 when a check fails. Not part of `run_all.py`: it needs the hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import Check, bootstrap, dim, header, table  # noqa: E402

bootstrap()
from visual_ai import accel  # noqa: E402
from visual_ai import openvino_zoo as zoo  # noqa: E402

FIXTURE = HERE / "fixtures" / "hand_motion.mp4"
#: A device that finds the subject on far fewer frames than the CPU is not
#: running the same network, however fast it is.
MIN_FOUND_RATIO = 0.8
MIN_BOX_IOU = 0.85
MAX_LANDMARK_PX = 20.0
TARGETS = {"boxes": f"IoU >= {MIN_BOX_IOU:.2f}", "pose": f"<= {MAX_LANDMARK_PX:.0f} px",
           "parts": f"<= {MAX_LANDMARK_PX:.0f} px", "topk": "top-1 >= 80 %",
           "hands": f"<= {MAX_LANDMARK_PX:.0f} px"}


# ── the wrappers under test ──────────────────────────────────────────────────

def _hands(device):
    from visual_ai.openvino_hands import OpenVINOHands
    return OpenVINOHands(device=device)


#: key -> (label, build(device), call(obj, rgb), kind)
WRAPPERS = {
    "yolo": ("YOLOv10n 640", lambda d: zoo.ObjectDetector(d), lambda o, f: o.detect(f), "boxes"),
    "person": ("person-detection-0200", lambda d: zoo.SSDDetector("person", d),
               lambda o, f: o.detect(f), "boxes"),
    "face-ssd": ("face-detection-0200", lambda d: zoo.SSDDetector("face", d),
                 lambda o, f: o.detect(f), "boxes"),
    "blazeface": ("BlazeFace 128", lambda d: zoo.FaceDetector(d), lambda o, f: o.detect(f), "boxes"),
    "pose": ("MediaPipe Pose", lambda d: zoo.BodyPose(d), lambda o, f: o.process(f), "pose"),
    "pose-gpu-det": ("MediaPipe Pose (NPU: detector on GPU)",
                     lambda d: zoo.BodyPose(d, detector_device="GPU" if d == "NPU" else None),
                     lambda o, f: o.process(f), "pose"),
    "openpose": ("OpenPose 256x456", lambda d: zoo.HeatmapPose(d), lambda o, f: o.detect(f), "parts"),
    "mobilenet": ("MobileNetV2", lambda d: zoo.ImageClassifier("mobilenetv2", d),
                  lambda o, f: o.classify(f), "topk"),
    "resnet": ("ResNet50", lambda d: zoo.ImageClassifier("resnet50", d),
               lambda o, f: o.classify(f), "topk"),
    "hands": ("MediaPipe Hands (openvino_hands)", _hands, lambda o, f: o.process(f), "hands"),
}


def found(kind: str, result) -> bool:
    if kind == "boxes":
        return bool(result)
    if kind == "pose":
        return result is not None
    if kind == "parts":
        return len(result) >= 3
    if kind == "topk":
        return True
    return bool(getattr(result, "multi_hand_landmarks", None))


def describe(kind: str, result) -> str:
    """One line of what a wrapper saw, for the live report."""
    if not found(kind, result):
        return "nothing"
    if kind == "boxes":
        d = result[0]
        return f"{d.label} {d.score:.2f} @ [{d.box[0]:.0f} {d.box[1]:.0f} {d.box[2]:.0f} {d.box[3]:.0f}]"
    if kind == "pose":
        nose, wrist = result.point("nose"), result.point("right_wrist")
        return (f"presence {result.score:.2f}, nose ({nose[0]:.0f}, {nose[1]:.0f}), "
                f"right wrist ({wrist[0]:.0f}, {wrist[1]:.0f})")
    if kind == "parts":
        names = ", ".join(list(result)[:4])
        return f"{len(result)} parts: {names}"
    if kind == "topk":
        return ", ".join(f"{label} {prob:.2f}" for label, prob in result[:3])
    hand = result.multi_hand_landmarks[0].landmark[8]
    return f"{len(result.multi_hand_landmarks)} hand(s), index tip ({hand.x:.2f}, {hand.y:.2f})"


def _hand_tip(result, shape):
    hand = result.multi_hand_landmarks[0].landmark[8]
    x, y = hand.x, hand.y
    if max(abs(x), abs(y)) <= 1.5:                # normalised, MediaPipe style
        x, y = x * shape[1], y * shape[0]
    return np.array([x, y])


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def compare(kind: str, reference: list, candidate: list, shape) -> tuple[str, float | None, bool]:
    """(text, value, passed) for `candidate` against the CPU `reference`."""
    pairs = [(r, c) for r, c in zip(reference, candidate)
             if found(kind, r) and found(kind, c)]
    if not pairs:
        return "no frame found on both", None, False
    if kind == "boxes":
        value = statistics.mean(iou(r[0].box, c[0].box) for r, c in pairs)
        return f"IoU {value:.3f}", value, value >= MIN_BOX_IOU
    if kind == "pose":
        value = statistics.mean(
            float(np.linalg.norm(r.landmarks[:, :2] - c.landmarks[:, :2], axis=1).mean())
            for r, c in pairs)
        return f"{value:.1f} px", value, value <= MAX_LANDMARK_PX
    if kind == "parts":
        distances = [np.hypot(r[k][0] - c[k][0], r[k][1] - c[k][1])
                     for r, c in pairs for k in r if k in c]
        value = statistics.mean(distances) if distances else None
        return (f"{value:.1f} px" if value is not None else "no shared parts",
                value, value is not None and value <= MAX_LANDMARK_PX)
    if kind == "topk":
        value = statistics.mean(r[0][0] == c[0][0] for r, c in pairs)
        return f"top-1 {value * 100:.0f} %", value, value >= 0.8
    value = statistics.mean(float(np.linalg.norm(_hand_tip(r, shape) - _hand_tip(c, shape)))
                            for r, c in pairs)
    return f"index tip {value:.1f} px", value, value <= MAX_LANDMARK_PX


# ── input ────────────────────────────────────────────────────────────────────

def load_frames(size=(640, 480)) -> list[np.ndarray]:
    if not FIXTURE.is_file():
        raise SystemExit(f"missing fixture: {FIXTURE}")
    capture = cv2.VideoCapture(str(FIXTURE))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise SystemExit(f"no frames in {FIXTURE}")
    return frames


def require_cached() -> None:
    cache = zoo._cache_dir()
    missing = [(filename, url) for filename, url, _ in zoo._REGISTRY.values()
               if not os.path.isfile(os.path.join(cache, filename))]
    if missing:
        print("weights missing from the model cache; fetch them first:")
        for filename, url in missing:
            print(f'  curl -L -o "{os.path.join(cache, filename)}" {url}')
        raise SystemExit(2)


def cpu_meter():
    try:
        import psutil
        process = psutil.Process()
        process.cpu_percent(None)
        return lambda: process.cpu_percent(None)
    except Exception:                              # noqa: BLE001 - optional
        return lambda: None


# ── offline ──────────────────────────────────────────────────────────────────

def run_offline(keys, devices, frames, args):
    rows, checks, results = [], [], {}
    shape = frames[0].shape
    # CPU first: it is the reference the accelerators are compared against.
    ordered = [d for d in devices if d == "CPU"] + [d for d in devices if d != "CPU"]
    for key in keys:
        label, build, call, kind = WRAPPERS[key]
        reference = None
        for device in ordered:
            started = time.perf_counter()
            try:
                obj = build(device)
            except Exception as exc:               # noqa: BLE001 - reported
                rows.append([label, device, f"{type(exc).__name__}: {str(exc).splitlines()[0][:50]}",
                             "", "", "", "", "", ""])
                continue
            build_ms = (time.perf_counter() - started) * 1e3
            compile_ms = getattr(obj, "compile_ms", build_ms)
            for frame in frames[:5]:
                call(obj, frame)
            meter = cpu_meter()
            samples, outputs = [], []
            for frame in frames:
                t = time.perf_counter()
                outputs.append(call(obj, frame))
                samples.append((time.perf_counter() - t) * 1e3)
            load = meter()
            if kind == "pose":
                # Tracking carries the ROI across frames and amplifies any device
                # difference into 20+ px even on the GPU, so agreement is judged
                # with the detector re-run on every frame (timings above stay tracked).
                outputs = []
                for frame in frames:
                    obj._rect = None
                    outputs.append(call(obj, frame))
            samples.sort()
            median = statistics.median(samples)
            p95 = samples[max(0, int(len(samples) * 0.95) - 1)]
            hits = sum(found(kind, o) for o in outputs) / len(outputs)
            device_name = getattr(obj, "device_name", getattr(obj, "device", device))
            if device == "CPU":
                reference, agree, value, passed = outputs, "reference", None, True
            elif reference is None:
                agree, value, passed = "no CPU reference", None, True
            else:
                agree, value, passed = compare(kind, reference, outputs, shape)
                checks.append(Check(f"{label} on {device} agrees with CPU", value or 0.0, "",
                                    TARGETS[kind], passed, agree))
                ref_hits = sum(found(kind, o) for o in reference) / len(reference)
                if ref_hits > 0:
                    checks.append(Check(f"{label} on {device} finds the subject", hits / ref_hits, "x",
                                        f">= {MIN_FOUND_RATIO:.1f}x of CPU", hits / ref_hits >= MIN_FOUND_RATIO))
            rows.append([label, device, device_name if device_name != device else "",
                         f"{compile_ms:.0f}", f"{median:.2f}", f"{p95:.2f}",
                         "—" if load is None else f"{load:.0f}", f"{hits * 100:.0f}", agree])
            results[f"{key}/{device}"] = {"compile_ms": compile_ms, "median_ms": median, "p95_ms": p95,
                                          "cpu_percent": load, "found": hits, "agreement": agree,
                                          "device_name": device_name}
            if hasattr(obj, "close"):
                obj.close()
    print(table(["wrapper", "device", "ran on", "compile ms", "med ms", "p95 ms", "CPU %", "found %",
                 "vs CPU"], rows, "lllrrrrrl"))
    print(dim("  med/p95 are the whole call: resize or crop, inference, decode.  CPU % is the process,\n"
              "  one core = 100.  found % is frames with a result; 'vs CPU' compares those frames."))
    return rows, checks, results


# ── live ─────────────────────────────────────────────────────────────────────

def run_live(keys, devices, args):
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, _ = capture.read()
    if not ok:
        print(f"camera {args.camera} gave no frame; nothing to measure")
        return [], [], {}
    rows, results = [], {}
    for key in keys:
        label, build, call, kind = WRAPPERS[key]
        for device in devices:
            try:
                obj = build(device)
            except Exception as exc:               # noqa: BLE001 - reported
                rows.append([label, device, f"{type(exc).__name__}: {str(exc).splitlines()[0][:50]}",
                             "", "", "", "", ""])
                continue
            device_name = getattr(obj, "device_name", getattr(obj, "device", device))
            grabs, calls, hits, last, frames = [], [], 0, "nothing", 0
            meter = cpu_meter()
            started = time.perf_counter()
            while time.perf_counter() - started < args.seconds:
                t = time.perf_counter()
                ok, bgr = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                grabbed = time.perf_counter()
                result = call(obj, rgb)
                calls.append((time.perf_counter() - grabbed) * 1e3)
                grabs.append((grabbed - t) * 1e3)
                frames += 1
                if found(kind, result):
                    hits += 1
                    last = describe(kind, result)
            load = meter()
            elapsed = time.perf_counter() - started
            rows.append([label, device_name, f"{frames / elapsed:.1f}",
                         f"{statistics.median(grabs):.1f}" if grabs else "—",
                         f"{statistics.median(calls):.2f}" if calls else "—",
                         "—" if load is None else f"{load:.0f}",
                         f"{hits / frames * 100:.0f}" if frames else "—", last])
            results[f"{key}/{device}"] = {"fps": frames / elapsed, "call_ms": statistics.median(calls) if calls else None,
                                          "grab_ms": statistics.median(grabs) if grabs else None,
                                          "cpu_percent": load, "found": hits / frames if frames else None,
                                          "last": last, "device_name": device_name}
            if hasattr(obj, "close"):
                obj.close()
    capture.release()
    print(table(["wrapper", "ran on", "fps", "grab ms", "call ms", "CPU %", "found %", "last result"],
                rows, "llrrrrrl"))
    print(dim("  fps is end to end (grab + convert + call), capped by the camera's own rate.\n"
              "  Whether the boxes and landmarks are on the right things needs a person watching."))
    return rows, [], results


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--devices", default="NPU,GPU,CPU")
    parser.add_argument("--only", default="", help="comma-separated subset of: " + ",".join(WRAPPERS))
    parser.add_argument("--camera", type=int, default=None, help="webcam index for the live mode")
    parser.add_argument("--seconds", type=float, default=10.0, help="live seconds per wrapper")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    if not accel.openvino_available():
        print("openvino is not importable; nothing to measure")
        return 0
    present = accel.available_devices()
    wanted = [d.strip().upper() for d in args.devices.split(",") if d.strip()]
    devices = [d for d in wanted if d in present]
    absent = [d for d in wanted if d not in present]
    mode = f"camera {args.camera}, {args.seconds:.0f} s each" if args.camera is not None else "fixture clip"
    print(header("NPU zoo", f"devices: {', '.join(devices) or 'none'}"
                 + (f"   absent: {', '.join(absent)}" if absent else "") + f"   input: {mode}"))
    if not devices:
        return 0
    require_cached()
    keys = [k.strip() for k in args.only.split(",") if k.strip()] or list(WRAPPERS)
    unknown = [k for k in keys if k not in WRAPPERS]
    if unknown:
        raise SystemExit(f"unknown wrapper(s) {unknown}; choose from {list(WRAPPERS)}")

    if args.camera is not None:
        rows, checks, results = run_live(keys, devices, args)
    else:
        if "CPU" not in devices:
            print(dim("  no CPU in --devices: nothing to compare against"))
        rows, checks, results = run_offline(keys, devices, load_frames(), args)

    if checks:
        print(header("checks"))
        for check in checks:
            shown = check.detail or f"{check.value:.3f} {check.unit}".strip()
            print(f"  {check.mark}  {check.name}: {shown}  (target {check.target})")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"devices": devices, "mode": mode, "results": results,
                       "checks": [c.as_dict() for c in checks]}, handle, indent=2, default=str)
        print(dim(f"  wrote {args.json}"))
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
