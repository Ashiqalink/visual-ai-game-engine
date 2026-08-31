"""
bench_npu_traits.py — what the NPU is good and bad at, as numbers.

One experiment per trait, on every OpenVINO device present (CPU, GPU, NPU),
one table each:

  floor        per-inference cost from a 128x128 face net to YOLOv10n@640:
               where the NPU starts to beat the CPU, and by how much
  resolution   YOLOv10n at 320/416/512/640: how each device scales with pixels
  batch        MobileNetV2 at batch 1/2/4/8: ms per image
  throughput   YOLOv10n: LATENCY x1 request vs THROUGHPUT x N async requests
  compile      MobileNetV2 compile ms with the driver/plugin cache bypassed,
               cold, and warm
  preprocess   uint8 input with PrePostProcessor folded into the graph vs
               float32 normalised in numpy
  cpu-load     process CPU % (of one core) while running YOLOv10n on each device
  concurrency  two YOLOv10n on one device; NPU and GPU at the same time
  agreement    FP16 devices vs the CPU: top-box IoU, score delta, top-1 match
  dynamic      a dynamic-output graph compiled for the NPU — in a subprocess,
               because on this driver it kills the interpreter
  sustained    YOLOv10n on the NPU for 20 s: first 2 s against last 2 s

Usage (from the engine directory, with the games venv — it has openvino,
mediapipe and psutil):

    python benchmarks/bench_npu_traits.py [--devices NPU,GPU,CPU] [--quick]
                                          [--only floor,batch] [--json out.json]

Exit 0 when no accelerator is present (it prints which devices it saw), 2
when weights are missing from the model cache — this bench never downloads;
it prints the commands to fetch them. Not part of `run_all.py`: it needs the
hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
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
from visual_ai.openvino_zoo import _Net, _core, mediapipe_model, model_path  # noqa: E402

FIXTURE = HERE / "fixtures" / "hand_motion.mp4"
SECTIONS = ("floor", "resolution", "batch", "throughput", "compile", "preprocess",
            "cpu-load", "concurrency", "agreement", "dynamic", "sustained")
IMAGENET = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


# ── helpers ──────────────────────────────────────────────────────────────────

def load_frames(count: int = 6) -> list[np.ndarray]:
    if not FIXTURE.is_file():
        raise SystemExit(f"missing fixture: {FIXTURE}")
    capture = cv2.VideoCapture(str(FIXTURE))
    frames, index = [], 0
    while len(frames) < count:
        ok, bgr = capture.read()
        if not ok:
            break
        if index % 15 == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        index += 1
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


def timed(fn, iters: int, warmup: int = 5) -> tuple[float, float]:
    """(median, p95) ms of `fn` over `iters` calls after `warmup`."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1e3)
    samples.sort()
    return statistics.median(samples), samples[max(0, int(len(samples) * 0.95) - 1)]


def ms(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def fill(net: _Net, rgb: np.ndarray) -> None:
    cv2.resize(rgb, (net.width, net.height), dst=net.input[0], interpolation=cv2.INTER_LINEAR)


def try_build(build, device: str):
    """`build(device)` or a one-line reason string."""
    try:
        return build(device), None
    except Exception as exc:                       # noqa: BLE001 - reported
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:60]}"


def cpu_percent_around(fn, seconds: float) -> tuple[float | None, float]:
    """Run `fn` in a loop for `seconds`; (process CPU % of one core, calls/s)."""
    try:
        import psutil
    except Exception:                              # noqa: BLE001 - optional
        psutil = None
    process = psutil.Process() if psutil else None
    if process:
        process.cpu_percent(None)
    calls, started = 0, time.perf_counter()
    while time.perf_counter() - started < seconds:
        fn()
        calls += 1
    elapsed = time.perf_counter() - started
    load = process.cpu_percent(None) if process else None
    return load, calls / elapsed


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


# ── traits ───────────────────────────────────────────────────────────────────

def floor(devices, frames, iters):
    nets = [
        ("BlazeFace 128", lambda d: _Net(
            mediapipe_model("face_detection/face_detection_short_range.tflite"), d,
            height=128, width=128, model_layout="NHWC", scale=127.5, mean=1.0)),
        ("hand_recrop 256", lambda d: _Net(
            mediapipe_model("holistic_landmark/hand_recrop.tflite"), d,
            height=256, width=256, model_layout="NHWC", scale=255.0)),
        ("MobileNetV2 224", lambda d: _Net(
            model_path("mobilenetv2"), d, height=224, width=224, **IMAGENET)),
        ("ResNet50 224", lambda d: _Net(
            model_path("resnet50"), d, height=224, width=224, **IMAGENET)),
        ("OpenPose 256x456", lambda d: _Net(
            model_path("human-pose-estimation-0001"), d, height=256, width=456,
            scale=1.0, bgr=True)),
        ("YOLOv10n 640", lambda d: _Net(
            model_path("yolov10n"), d, height=640, width=640)),
    ]
    rows, compile_rows, result = [], [], {}
    for name, build in nets:
        medians, compiles = {}, {}
        for device in devices:
            net, why = try_build(build, device)
            if net is None:
                medians[device] = compiles[device] = None
                rows.append([name, device, why])
                continue
            fill(net, frames[0])
            medians[device], _ = timed(net.infer, iters)
            compiles[device] = net.compile_ms
            del net
        ratio = (medians["CPU"] / medians["NPU"]
                 if medians.get("CPU") and medians.get("NPU") else None)
        rows.append([name] + [ms(medians.get(d)) for d in devices]
                    + [f"{ratio:.2f}x" if ratio else "—"])
        compile_rows.append([name] + [ms(compiles.get(d)) for d in devices])
        result[name] = {"infer_ms": medians, "compile_ms": compiles}
    rows = [r for r in rows if len(r) == len(devices) + 2]
    print(table(["network", *[f"{d} ms" for d in devices], "CPU/NPU"], rows,
                "l" + "r" * (len(devices) + 1)))
    print(dim("  compile ms (warm driver cache where the device has one)"))
    print(table(["network", *devices], compile_rows, "l" + "r" * len(devices)))
    return result


def resolution(devices, frames, iters):
    """MobileNetV2 reshaped to 128..640: pure convolutions, so the input can
    be any size and the cost is the pixels alone. (The YOLOv10n export is
    fixed at 640 — its anchor split constants do not survive a reshape.)"""
    path = model_path("mobilenetv2")
    rows, result = [], {}
    for size in (128, 224, 320, 448, 640):
        row, result[size] = [str(size)], {}
        for device in devices:
            net, why = try_build(
                lambda d: _Net(path, d, height=size, width=size, **IMAGENET), device)
            if net is None:
                row.append(why)
                continue
            fill(net, frames[0])
            median, _ = timed(net.infer, iters)
            row.append(f"{median:.2f}")
            result[size][device] = median
            del net
        base = result[128]
        row.append("  ".join(f"{d} {result[size][d] / base[d]:.1f}x" for d in devices
                             if d in base and d in result[size]))
        rows.append(row)
    print(table(["MobileNetV2 size", *[f"{d} ms" for d in devices], "vs 128"], rows,
                "l" + "r" * len(devices) + "l"))
    return result


def _compile_classifier(path, device, batch, hint="LATENCY", config=None, ppp=True):
    import openvino as ov
    from openvino.preprocess import PrePostProcessor

    core = _core()
    model = core.read_model(path)
    model.reshape({model.inputs[0]: ov.PartialShape([batch, 3, 224, 224])})
    if ppp:
        pre = PrePostProcessor(model)
        pre.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC"))
        pre.input().model().set_layout(ov.Layout("NCHW"))
        (pre.input().preprocess().convert_element_type(ov.Type.f32).scale(255.0)
         .mean(list(IMAGENET["mean"])).scale(list(IMAGENET["std"])))
        model = pre.build()
    started = time.perf_counter()
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": hint, **(config or {})})
    return compiled, (time.perf_counter() - started) * 1e3


def batch(devices, frames, iters):
    import openvino as ov

    path = model_path("mobilenetv2")
    crop = cv2.resize(frames[0], (224, 224))
    rows, result = [], {}
    for size in (1, 2, 4, 8):
        row, result[size] = [str(size)], {}
        for device in devices:
            try:
                compiled, _ = _compile_classifier(path, device, size)
                request = compiled.create_infer_request()
                buffer = np.repeat(crop[None], size, axis=0).copy()
                request.set_input_tensor(ov.Tensor(buffer, shared_memory=True))
                median, _ = timed(request.infer, iters)
            except Exception as exc:               # noqa: BLE001 - reported
                row.append(f"{type(exc).__name__}: {str(exc).splitlines()[0][:40]}")
                continue
            row.append(f"{median / size:.2f} ({median:.1f})")
            result[size][device] = {"ms_per_image": median / size, "ms_per_batch": median}
        rows.append(row)
    print(table(["batch", *[f"{d} ms/img (batch)" for d in devices]], rows,
                "l" + "r" * len(devices)))
    return result


def throughput(devices, frames, seconds):
    import openvino as ov

    path = model_path("yolov10n")
    canvas, *_ = zoo._letterbox(frames[0], 640)
    rows, result = [], {}

    def run(compiled, count):
        # The buffers must outlive the requests: a shared-memory Tensor over a
        # temporary array is a dangling pointer once the temporary is freed,
        # and on the NPU that is a segfault, not an exception.
        buffers = [canvas[None].copy() for _ in range(count)]
        requests = [compiled.create_infer_request() for _ in range(count)]
        for request, buffer in zip(requests, buffers):
            request.set_input_tensor(ov.Tensor(buffer, shared_memory=True))
            request.start_async()
        done, started = 0, time.perf_counter()
        while time.perf_counter() - started < seconds:
            for request in requests:
                request.wait()
                done += 1
                request.start_async()
        for request in requests:
            request.wait()
        return done / (time.perf_counter() - started)

    for device in devices:
        cells, result[device] = [device], {}
        for label, hint, forced in (("LATENCY x1", "LATENCY", 1),
                                    ("THROUGHPUT xN", "THROUGHPUT", None),
                                    ("LATENCY x4", "LATENCY", 4)):
            print(dim(f"  {device} {label} ..."), flush=True)
            try:
                model = zoo._prepare_model(path, height=640, width=640)
                started = time.perf_counter()
                compiled = _core().compile_model(model, device, {"PERFORMANCE_HINT": hint})
                compile_ms = (time.perf_counter() - started) * 1e3
                optimal = int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
                count = forced or optimal
                fps = run(compiled, count)
            except Exception as exc:               # noqa: BLE001 - reported
                cells.append(f"{type(exc).__name__}: {str(exc).splitlines()[0][:30]}")
                continue
            cells.append(f"{fps:.1f} fps (n={count}, {compile_ms:.0f} ms)")
            result[device][label] = {"fps": fps, "requests": count, "compile_ms": compile_ms}
        rows.append(cells)
    print(table(["YOLOv10n 640", "LATENCY x1", "THROUGHPUT x optimal", "LATENCY x4 async"],
                rows, "lrrr"))
    return result


def compile_cache(devices, frames, iters):
    path = model_path("mobilenetv2")
    rows, result = [], {}
    for device in devices:
        cells, result[device] = [device], {}
        if device == "NPU":
            arms = (("bypass driver cache", {"NPU_BYPASS_UMD_CACHING": True}),
                    ("default #1", {}), ("default #2", {}))
        else:
            cache = tempfile.mkdtemp(prefix="ov_cache_")
            arms = (("no CACHE_DIR", {}), ("CACHE_DIR cold", {"CACHE_DIR": cache}),
                    ("CACHE_DIR warm", {"CACHE_DIR": cache}))
        for label, config in arms:
            try:
                _, compile_ms = _compile_classifier(path, device, 1, config=config)
            except Exception as exc:               # noqa: BLE001 - reported
                cells.append(f"{type(exc).__name__}: {str(exc).splitlines()[0][:30]}")
                continue
            cells.append(f"{label}: {compile_ms:.0f}")
            result[device][label] = compile_ms
        rows.append(cells)
    print(table(["MobileNetV2 compile ms", "arm 1", "arm 2", "arm 3"], rows, "lrrr"))
    return result


def preprocess(devices, frames, iters):
    import openvino as ov

    path = model_path("mobilenetv2")
    mean = np.asarray(IMAGENET["mean"], np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET["std"], np.float32).reshape(3, 1, 1)
    rows, result = [], {}
    for device in devices:
        try:
            folded = _Net(path, device, height=224, width=224, **IMAGENET)

            def run_folded():
                cv2.resize(frames[0], (224, 224), dst=folded.input[0])
                folded.infer()

            raw, _ = _compile_classifier(path, device, 1, ppp=False)
            request = raw.create_infer_request()

            def run_numpy():
                crop = cv2.resize(frames[0], (224, 224)).astype(np.float32) / 255.0
                chw = (crop.transpose(2, 0, 1) - mean) / std
                request.infer({0: ov.Tensor(np.ascontiguousarray(chw[None]))})

            folded_ms, _ = timed(run_folded, iters)
            numpy_ms, _ = timed(run_numpy, iters)
        except Exception as exc:                   # noqa: BLE001 - reported
            rows.append([device, f"{type(exc).__name__}: {str(exc).splitlines()[0][:40]}", "", ""])
            continue
        rows.append([device, ms(folded_ms), ms(numpy_ms), f"{numpy_ms - folded_ms:+.2f}"])
        result[device] = {"folded_ms": folded_ms, "numpy_ms": numpy_ms}
    print(table(["MobileNetV2", "PPP folded ms", "numpy f32 ms", "delta"], rows, "lrrr"))
    return result


def cpu_load(devices, frames, seconds):
    rows, result = [], {}
    for device in devices:
        det, why = try_build(lambda d: zoo.ObjectDetector(d), device)
        if det is None:
            rows.append([device, why, ""])
            continue
        load, fps = cpu_percent_around(lambda: det.detect(frames[0]), seconds)
        rows.append([device, "psutil missing" if load is None else f"{load:.0f}", f"{fps:.1f}"])
        result[device] = {"cpu_percent": load, "fps": fps}
        det.close()
    print(table(["YOLOv10n 640", "process CPU % (1 core = 100)", "fps"], rows, "lrr"))
    return result


def concurrency(devices, frames, seconds):
    def worker(det, out, slot):
        samples, started = [], time.perf_counter()
        while time.perf_counter() - started < seconds:
            t = time.perf_counter()
            det.detect(frames[0])
            samples.append((time.perf_counter() - t) * 1e3)
        out[slot] = samples          # by slot, not by finishing order

    def pair(first, second):
        a, why_a = try_build(lambda d: zoo.ObjectDetector(d), first)
        b, why_b = try_build(lambda d: zoo.ObjectDetector(d), second)
        if a is None or b is None:
            return None, why_a or why_b
        solo = {first: timed(lambda: a.detect(frames[0]), 20)[0],
                second: timed(lambda: b.detect(frames[0]), 20)[0]}
        results: list[list[float]] = [[], []]
        threads = [threading.Thread(target=worker, args=(det, results, slot))
                   for slot, det in enumerate((a, b))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        a.close(), b.close()
        both = [statistics.median(s) for s in results]
        fps = sum(len(s) for s in results) / seconds
        return {"solo": solo, "together": both, "fps": fps}, None

    rows, result = [], {}
    configs = [(d, d) for d in devices if d != "CPU"]
    if "NPU" in devices and "GPU" in devices:
        configs.append(("NPU", "GPU"))
    for first, second in configs:
        label = f"{first} + {second}"
        got, why = pair(first, second)
        if got is None:
            rows.append([label, why, "", ""])
            continue
        rows.append([label, f"{ms(got['solo'][first])} / {ms(got['solo'][second])}",
                     f"{ms(got['together'][0])} / {ms(got['together'][1])}", f"{got['fps']:.1f}"])
        result[label] = got
    print(table(["two YOLOv10n", "solo ms", "together ms", "combined fps"], rows, "lrrr"))
    return result


def agreement(devices, frames, iters):
    if "CPU" not in devices:
        print(dim("  needs CPU in --devices as the reference"))
        return {}
    ref_det = zoo.ObjectDetector("CPU")
    ref_cls = zoo.ImageClassifier("mobilenetv2", "CPU")
    ref_boxes = [ref_det.detect(f) for f in frames]
    ref_top = [ref_cls.classify(f)[0] for f in frames]
    rows, result = [], {}
    for device in [d for d in devices if d != "CPU"]:
        det, why = try_build(lambda d: zoo.ObjectDetector(d), device)
        cls, why_c = try_build(lambda d: zoo.ImageClassifier("mobilenetv2", d), device)
        if det is None or cls is None:
            rows.append([device, why or why_c, "", "", ""])
            continue
        ious, deltas, matches, prob_deltas = [], [], 0, []
        for frame, ref_found, (ref_label, ref_prob) in zip(frames, ref_boxes, ref_top):
            found = det.detect(frame)
            if ref_found and found:
                ious.append(iou(ref_found[0].box, found[0].box))
                deltas.append(abs(ref_found[0].score - found[0].score))
            elif bool(ref_found) != bool(found):
                ious.append(0.0)
            label, prob = cls.classify(frame)[0]
            matches += label == ref_label
            prob_deltas.append(abs(prob - ref_prob))
        summary = {"iou": statistics.mean(ious) if ious else None,
                   "score_delta": max(deltas) if deltas else None,
                   "top1": matches / len(frames), "prob_delta": max(prob_deltas)}
        rows.append([device, "—" if summary["iou"] is None else f"{summary['iou']:.3f}",
                     "—" if summary["score_delta"] is None else f"{summary['score_delta']:.3f}",
                     f"{matches}/{len(frames)}", f"{summary['prob_delta']:.3f}"])
        result[device] = summary
        det.close(), cls.close()
    print(table(["vs CPU", "top-box IoU", "max score delta", "top-1 match", "max prob delta"],
                rows, "lrrrr"))
    return result


def dynamic(devices, frames, iters):
    path = os.path.join(zoo._cache_dir(), "ssd_mobilenet_v1_12.onnx")
    if "NPU" not in devices:
        print(dim("  needs the NPU"))
        return {}
    if not os.path.isfile(path):
        print(dim(f"  {path} absent — this negative case is not in the registry; fetch\n"
                  "  https://github.com/onnx/models/raw/main/validated/vision/object_detection_"
                  "segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx to run it"))
        return {"skipped": "model absent"}
    code = ("import openvino as ov, sys\n"
            "core = ov.Core(); model = core.read_model(sys.argv[1])\n"
            "print('outputs', [str(o.get_partial_shape()) for o in model.outputs], flush=True)\n"
            "core.compile_model(model, 'NPU'); print('compiled', flush=True)\n")
    started = time.perf_counter()
    try:
        run = subprocess.run([sys.executable, "-c", code, path], capture_output=True,
                             text=True, timeout=240)
        code_str = f"{run.returncode} (0x{run.returncode & 0xFFFFFFFF:08X})"
        out = (run.stdout.strip().splitlines() or ["(no stdout)"])[-1][:70]
        err = (run.stderr.strip().splitlines() or [""])[0][:70]
        result = {"returncode": run.returncode, "stdout": out, "stderr": err}
    except subprocess.TimeoutExpired:
        code_str, out, err = "timeout", "", ""
        result = {"returncode": None, "timeout": True}
    result["seconds"] = time.perf_counter() - started
    print(table(["dynamic-output SSD on NPU", "exit code", "last stdout", "first stderr"],
                [[f"{result['seconds']:.1f} s", code_str, out, err]], "lrll"))
    return result


def sustained(devices, frames, seconds):
    device = "NPU" if "NPU" in devices else next((d for d in devices if d != "CPU"), None)
    if device is None:
        print(dim("  needs an accelerator"))
        return {}
    det, why = try_build(lambda d: zoo.ObjectDetector(d), device)
    if det is None:
        print(dim(f"  {device}: {why}"))
        return {}
    for _ in range(10):
        det.detect(frames[0])
    samples, started = [], time.perf_counter()
    while True:
        now = time.perf_counter() - started
        if now >= seconds:
            break
        t = time.perf_counter()
        det.detect(frames[len(samples) % len(frames)])
        samples.append((now, (time.perf_counter() - t) * 1e3))
    det.close()
    head = statistics.median(v for t, v in samples if t < 2.0)
    tail = statistics.median(v for t, v in samples if t >= seconds - 2.0)
    values = sorted(v for _, v in samples)
    result = {"device": device, "first_2s_ms": head, "last_2s_ms": tail,
              "p99_ms": values[int(len(values) * 0.99) - 1], "max_ms": values[-1],
              "frames": len(samples), "seconds": seconds}
    print(table([f"YOLOv10n on {device}, {seconds:.0f} s", "first 2 s", "last 2 s", "p99", "max", "frames"],
                [["median ms", ms(head), ms(tail), ms(result["p99_ms"]), ms(result["max_ms"]),
                  str(len(samples))]], "lrrrrr"))
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--devices", default="NPU,GPU,CPU")
    parser.add_argument("--only", default="", help="comma-separated subset of: " + ",".join(SECTIONS))
    parser.add_argument("--quick", action="store_true", help="a third of the iterations, 5 s sustained")
    parser.add_argument("--json", default="", help="write every number here")
    args = parser.parse_args()

    if not accel.openvino_available():
        print("openvino is not importable; nothing to measure")
        return 0
    present = accel.available_devices()
    wanted = [d.strip().upper() for d in args.devices.split(",") if d.strip()]
    devices = [d for d in wanted if d in present]
    absent = [d for d in wanted if d not in present]
    print(header("NPU traits", f"devices: {', '.join(devices) or 'none'}"
                 + (f"   absent: {', '.join(absent)}" if absent else "")))
    if not devices:
        return 0
    if "NPU" not in devices:
        print(dim("  no NPU: the CPU/GPU columns still run, but this bench is about the NPU"))
    require_cached()

    iters = 20 if args.quick else 60
    seconds = 1.5 if args.quick else 3.0
    sustained_seconds = 5.0 if args.quick else 20.0
    frames = load_frames()
    selected = [s.strip() for s in args.only.split(",") if s.strip()] or list(SECTIONS)
    unknown = [s for s in selected if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown section(s) {unknown}; choose from {SECTIONS}")

    runners = {
        "floor": (floor, iters), "resolution": (resolution, iters), "batch": (batch, iters),
        "throughput": (throughput, seconds), "compile": (compile_cache, iters),
        "preprocess": (preprocess, iters), "cpu-load": (cpu_load, seconds),
        "concurrency": (concurrency, seconds), "agreement": (agreement, iters),
        "dynamic": (dynamic, iters), "sustained": (sustained, sustained_seconds),
    }
    results = {}
    for name in selected:
        fn, arg = runners[name]
        print(header(name))
        started = time.perf_counter()
        results[name] = fn(devices, frames, arg)
        print(dim(f"  {time.perf_counter() - started:.1f} s"))

    checks = []
    got = results.get("floor", {})
    face = got.get("BlazeFace 128", {}).get("infer_ms", {})
    if face.get("NPU"):
        checks.append(Check("NPU per-inference floor (BlazeFace 128)", face["NPU"], "ms",
                            "< 2.0 ms", face["NPU"] < 2.0))
    yolo = got.get("YOLOv10n 640", {}).get("infer_ms", {})
    if yolo.get("NPU") and yolo.get("CPU"):
        ratio = yolo["CPU"] / yolo["NPU"]
        checks.append(Check("NPU speed-up over CPU (YOLOv10n 640)", ratio, "x",
                            ">= 1.5x", ratio >= 1.5))
    agree = results.get("agreement", {}).get("NPU")
    if agree:
        if agree["iou"] is not None:
            checks.append(Check("NPU top-box IoU vs CPU (YOLOv10n)", agree["iou"], "",
                                ">= 0.90", agree["iou"] >= 0.90))
        checks.append(Check("NPU top-1 agrees with CPU (MobileNetV2)", agree["top1"] * 100,
                            "%", ">= 80 %", agree["top1"] >= 0.8))
    drift = results.get("sustained", {})
    if drift.get("first_2s_ms"):
        ratio = drift["last_2s_ms"] / drift["first_2s_ms"]
        checks.append(Check(f"sustained drift on {drift['device']} (last/first 2 s)", ratio, "x",
                            "<= 1.25x", ratio <= 1.25))
    if checks:
        print(header("checks"))
        for check in checks:
            print(f"  {check.mark}  {check.name}: {check.value:.3f} {check.unit}  (target {check.target})")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"devices": devices, "results": results,
                       "checks": [c.as_dict() for c in checks]}, handle, indent=2, default=str)
        print(dim(f"  wrote {args.json}"))
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
