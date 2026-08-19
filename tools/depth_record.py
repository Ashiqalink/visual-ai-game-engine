"""Record depth frames from a sensor so they can be replayed without it.

This is how hardware gets into the test suite. On the machine that has the
sensor:

    python tools/depth_record.py --source auto --seconds 20 --out lab.npz

and then anywhere, with no sensor attached:

    python tools/depth_record.py --inspect lab.npz
    VisionPipeline(..., depth_source="replay:lab.npz")

A recording carries the sensor's real noise, real dropouts and real
resolution, which is exactly what the synthetic source cannot imitate and what
the ToF consumer code most needs to be tested against.
"""

import os
import sys
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from visual_ai.depth_source import (          # noqa: E402
    DepthRecorder,
    open_depth_source,
    probe_depth_sources,
)


def describe(frame):
    valid = frame[frame > 0]
    if not valid.size:
        return "no valid returns"
    return (f"{frame.shape[1]}x{frame.shape[0]}  "
            f"{100.0 * valid.size / frame.size:5.1f}% valid  "
            f"{valid.min() / 1000.0:.2f}-{valid.max() / 1000.0:.2f} m  "
            f"median {np.median(valid) / 1000.0:.2f} m")


def do_probe():
    print("\ndepth backends:\n")
    for name, ok, detail in probe_depth_sources():
        print(f"  {name:12s} {'OK  ' if ok else 'no  '} {detail}")
    print()
    return 0


def do_inspect(path):
    data = np.load(path)
    frames = data["frames"]
    print(f"\n{path}: {len(frames)} frames, {frames.dtype}, "
          f"{frames.nbytes / 1e6:.1f} MB in memory\n")
    for i in (0, len(frames) // 2, len(frames) - 1):
        print(f"  frame {i:5d}  {describe(frames[i])}")
    holes = float(np.mean(frames == 0)) * 100.0
    print(f"\n  {holes:.1f}% of all pixels are no-return\n")
    return 0


def do_record(spec, seconds, out, max_frames):
    source = open_depth_source(spec, quiet=False)
    if source is None:
        print(f"no depth source for {spec!r}. Try --probe to see why.")
        return 2
    print(f"recording from {source.name} for {seconds:.0f}s -- Ctrl+C to stop early")

    recorder = DepthRecorder(max_frames=max_frames)
    deadline = time.perf_counter() + seconds
    try:
        while time.perf_counter() < deadline and not recorder.full:
            frame = source.read()
            if frame is None:
                continue
            recorder.feed(frame)
            if len(recorder.frames) % 30 == 0:
                print(f"  {len(recorder.frames)} frames  {describe(frame)}")
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        source.close()

    if not recorder.frames:
        print("nothing recorded -- the source opened but produced no frames")
        return 2
    recorder.save(out)
    size = os.path.getsize(out) / 1e6
    print(f"\nwrote {len(recorder.frames)} frames to {out} ({size:.1f} MB)")
    print(f'replay with: VisionPipeline(..., depth_source="replay:{out}")')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto",
                    help='depth source spec (default "auto")')
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--out", default="depth_recording.npz")
    ap.add_argument("--max-frames", type=int, default=900,
                    help="hard cap; depth is 2 bytes per pixel per frame")
    ap.add_argument("--probe", action="store_true",
                    help="list depth backends and why each did or did not open")
    ap.add_argument("--inspect", metavar="PATH",
                    help="summarise an existing recording")
    args = ap.parse_args(argv)

    if args.probe:
        return do_probe()
    if args.inspect:
        return do_inspect(args.inspect)
    return do_record(args.source, args.seconds, args.out, args.max_frames)


if __name__ == "__main__":
    sys.exit(main())
