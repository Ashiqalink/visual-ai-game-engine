"""
bench_monocular_depth.py — is estimating depth from RGB affordable per frame?

The question this answers is not "how fast is FastDepth" in the abstract. It
is whether `MonocularDepthSource` fits in a frame budget already spent on
MediaPipe hand tracking and, in `avatarcatch`, MODNet matting. So the bench
measures the pieces separately:

  * **overhead** — resize, ImageNet normalise and transpose on the way in;
    metres-to-millimetres and `sanitize` on the way out. Pure NumPy and
    OpenCV, so it runs with no model at all and is measurable today.
  * **inference** — the `onnxruntime` session run. Needs real weights, and is
    reported as unavailable rather than guessed at when there are none.

Run::

    python benchmarks/bench_monocular_depth.py                  # overhead only
    python benchmarks/bench_monocular_depth.py --model f.onnx   # + inference

`--model` also accepts the VISUAL_AI_DEPTH_ONNX environment variable. The
budget check fails loudly when the total exceeds `--budget-ms`, which defaults
to 11 ms — what `matting.py` documents its downsampled MODNet pass as costing
at 640x360, and therefore the bar a second per-frame network has to clear to
be worth having.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import bold, bootstrap, cyan, dim, green, header, red  # noqa: E402

bootstrap()

from visual_ai.depth_source import (  # noqa: E402
    MonocularDepthSource,
    sanitize,
)

#: MODNet's documented downsampled cost at 640x360 (see matting.py). A second
#: per-frame network that costs more than the one already shipping is a hard
#: sell, so this is the default bar.
DEFAULT_BUDGET_MS = 11.0

#: Capture sizes that actually occur here: the pipeline's default and the
#: 720p a webcam hands back when nothing overrides it.
FRAME_SIZES = ((640, 360), (1280, 720))

#: Two titles, because the distinction is the whole point: a proxy answers
#: "can this architecture fit the budget", never "does this model work".
_INFERENCE_TITLE = "Inference — real weights through the real backend"
_INFERENCE_TITLE_PROXY = "Inference — PROXY weights through the real backend"


def _scene(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A BGR frame with real structure, not flat noise.

    Timing on a constant image is a trap: it is representative for a dense CNN
    but not for the resize, and it invites a reader to assume the whole bench
    is measuring nothing. This builds gradients plus a blob plus mild noise so
    every stage does the work it would do on a camera frame.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    base = (xs / max(1, width - 1)) * 140.0 + (ys / max(1, height - 1)) * 60.0
    blob = 90.0 * np.exp(
        -(((xs - width * 0.45) ** 2) / (2 * (width * 0.13) ** 2)
          + ((ys - height * 0.55) ** 2) / (2 * (height * 0.17) ** 2)))
    frame = np.stack([base, base * 0.8 + blob, base * 0.6 + blob * 1.2], axis=-1)
    frame += rng.normal(0.0, 4.0, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _time_ms(fn, repeats: int, warmup: int = 3) -> dict:
    """Median / p95 / min wall-clock ms over `repeats` calls."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
        "min_ms": samples[0],
        "n": repeats,
    }


def bench_overhead(repeats: int) -> list[dict]:
    """Cost of everything around the session run, at each capture size.

    Uses the real `_preprocess` and the real `sanitize` rather than a
    reimplementation, so this cannot drift away from what the backend does.
    """
    rows = []
    source = MonocularDepthSource(model_path="unused")   # never opened
    for width, height in FRAME_SIZES:
        frame = _scene(width, height)
        # A plausible network output at FastDepth's native size, in metres.
        raw = np.random.default_rng(1).uniform(0.4, 4.0, (224, 224)).astype(np.float32)

        pre = _time_ms(lambda: source._preprocess(frame), repeats)

        # The backend returns the map at the network's resolution rather than
        # the frame's, so postprocess is metres->mm plus `sanitize` at 224x224
        # and nothing else. It is deliberately independent of frame size; if a
        # future change reintroduces an upscale, this column stops being flat
        # and the regression is visible here.
        def post():
            return sanitize(raw * 1000.0)

        post_stats = _time_ms(post, repeats)
        rows.append({
            "scenario": f"{width}x{height}",
            "preprocess_ms": pre["median_ms"],
            "postprocess_ms": post_stats["median_ms"],
            "overhead_ms": pre["median_ms"] + post_stats["median_ms"],
            "preprocess_p95_ms": pre["p95_ms"],
            "postprocess_p95_ms": post_stats["p95_ms"],
        })
    return rows


def bench_inference(model_path: str, repeats: int) -> dict:
    """End-to-end `submit` + `read` through the real backend, or why it could not run."""
    source = MonocularDepthSource(model_path=model_path)
    if not source.open():
        return {"ok": False, "reason": source.last_error}

    rows = []
    try:
        for width, height in FRAME_SIZES:
            frame = _scene(width, height)

            # `last_infer_ms` is overwritten every call, so sample it as the
            # runs happen. Reporting only the final frame's value made the
            # inference column swing by 5ms between runs while the total held
            # steady -- one sample of a noisy CPU, not a measurement.
            infer_samples = []

            def one():
                source.submit(frame)
                result = source.read()
                infer_samples.append(source.last_infer_ms)
                return result

            probe = one()
            if probe is None:
                return {"ok": False,
                        "reason": f"backend produced no frame: {source.last_error}"}
            # Native network resolution, not the frame's -- see `_read`.
            if probe.shape != source._input_size:
                return {"ok": False,
                        "reason": f"expected {source._input_size}, got {probe.shape}"}

            stats = _time_ms(one, repeats)
            valid = float(np.count_nonzero(probe)) / probe.size
            rows.append({
                "scenario": f"{width}x{height}",
                "total_ms": stats["median_ms"],
                "total_p95_ms": stats["p95_ms"],
                "infer_ms": statistics.median(infer_samples),
                "valid_fraction": valid,
                "fps": 1000.0 / stats["median_ms"] if stats["median_ms"] else 0.0,
            })
    finally:
        source.close()
    return {"ok": True, "rows": rows, "input_size": source._input_size}


def _print_overhead(rows: list[dict]) -> None:
    print(header("Overhead — everything except the session run"))
    print(f"  {'frame':<12}{'preprocess':>13}{'postprocess':>14}{'total':>11}")
    for row in rows:
        print(f"  {row['scenario']:<12}"
              f"{row['preprocess_ms']:>11.2f}ms"
              f"{row['postprocess_ms']:>12.2f}ms"
              f"{row['overhead_ms']:>9.2f}ms")
    print(dim("  preprocess = resize + ImageNet normalise + transpose"))
    print(dim("  postprocess = metres->mm + sanitize, at the network resolution"))


def _print_inference(result: dict, budget_ms: float, proxy: bool = False) -> bool:
    print(header(_INFERENCE_TITLE_PROXY if proxy else _INFERENCE_TITLE))
    if not result["ok"]:
        print(red(f"  UNAVAILABLE: {result['reason']}"))
        print(dim("  No number is reported here rather than a guessed one."))
        return True                     # not a failure, just not measured

    print(dim(f"  model input {result['input_size'][1]}x{result['input_size'][0]}"))
    print(f"  {'frame':<12}{'infer':>11}{'total':>11}{'p95':>11}"
          f"{'fps':>9}{'valid':>9}")
    passed = True
    for row in result["rows"]:
        over = row["total_ms"] > budget_ms
        passed = passed and not over
        mark = red("OVER") if over else green("ok")
        print(f"  {row['scenario']:<12}"
              f"{row['infer_ms']:>9.2f}ms"
              f"{row['total_ms']:>9.2f}ms"
              f"{row['total_p95_ms']:>9.2f}ms"
              f"{row['fps']:>9.1f}"
              f"{row['valid_fraction'] * 100:>8.0f}%  {mark}")
    print(dim(f"  budget {budget_ms:.1f}ms — MODNet's documented downsampled cost"))
    print(dim("  valid = pixels surviving sanitize; a low number means the "
              "model's range disagrees with MIN/MAX_VALID_MM"))
    if proxy:
        print(red("  random weights: these milliseconds are real, the depth "
                  "values and the valid fraction are not"))
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("VISUAL_AI_DEPTH_ONNX", ""),
                        help="FastDepth ONNX file; without one only overhead is measured")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS)
    parser.add_argument("--proxy", action="store_true",
                        help="the model is make_depth_proxy's random-weight "
                             "architecture stand-in, not a trained network")
    args = parser.parse_args(argv)

    print(bold(cyan("monocular depth — per-frame cost")))
    overhead = bench_overhead(args.repeats)
    _print_overhead(overhead)

    if args.model:
        passed = _print_inference(bench_inference(args.model, args.repeats),
                                  args.budget_ms, proxy=args.proxy)
    else:
        print(header(_INFERENCE_TITLE))
        print(red("  NOT RUN: no --model given and VISUAL_AI_DEPTH_ONNX is unset."))
        print(dim("  Overhead above is real; the inference cost is unmeasured."))
        passed = True

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
