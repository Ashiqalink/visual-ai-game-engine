"""
scratch.py - the 3D scratch slot. Overwrite this file.

`demo_3d.py` is a demo: it is supposed to keep working, so changing it costs a
decision. This one costs nothing. Nothing imports it, no test covers it, and
no other title reads anything from it - it exists so that trying an idea
against the renderer is one command (`play3d scratch`) rather than a new file,
a new sys.path dance and a new argument parser every time.

What it starts as: one mesh on a turntable, optionally with the hand payload
driving it, with the launcher's 3D overrides applying to it like any other
title - `play3d scratch --wireframe --fov 35 --camz 800`.

    play3d scratch                  a window, driven by the clock
    play3d scratch --hands          driven by the pipeline instead
    play3d scratch --mesh torus     start from a different primitive
    play3d scratch --headless 300   no window, no camera, 300 frames

Headless is not an afterthought here: it is what makes a change to the
renderer checkable without a camera in front of you, and it is the mode the
run harness uses.
"""

from __future__ import annotations

import argparse
import queue
import time

import cv2
import numpy as np

from visual_ai import (
    Camera3D,
    Material,
    Mesh3D,
    Renderer3D,
    Transform3D,
    VisionPipeline,
    voxel,
)

MESHES = {
    "cube": lambda: Mesh3D.create_cube(size=80.0),
    "sphere": lambda: Mesh3D.create_sphere(radius=45.0, rings=10, sectors=16),
    "torus": lambda: Mesh3D.create_torus(ring_radius=40.0, tube_radius=14.0),
    "capsule": lambda: Mesh3D.create_capsule(radius=25.0, height=60.0),
    "icosahedron": lambda: Mesh3D.create_icosahedron(radius=45.0),
    "pyramid": lambda: Mesh3D.create_pyramid(width=70.0, height=90.0),
    "cylinder": lambda: Mesh3D.create_cylinder(radius=35.0, height=80.0, segments=12),
    "prism": lambda: Mesh3D.create_prism(width=50.0, height=50.0),
}

#: And the same shapes as voxel models, under `voxel-<name>`.
MESHES.update({f"voxel-{name}": builder
               for name, builder in voxel.MODELS.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="3D scratch slot.")
    parser.add_argument("--mesh", default="torus", choices=sorted(MESHES))
    parser.add_argument("--material", default="gold")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hands", action="store_true",
                        help="drive the turntable from the hand payload")
    parser.add_argument("--headless", type=int, metavar="N",
                        help="render N frames with no window and no camera")
    args = parser.parse_args(argv)

    width, height = args.width, args.height
    mesh = MESHES[args.mesh]()
    material = Material.preset(args.material)
    # fov and the camera Z are what `play3d --fov / --camz` rewrite; they are
    # passed as keywords for that reason, not positionally.
    camera = Camera3D(fov=60.0, screen_width=width, screen_height=height,
                      position=(0.0, 0.0, 500.0))
    renderer = Renderer3D(camera=camera)
    transform = Transform3D(sx=1.0, sy=1.0, sz=1.0)

    pipeline = None
    payload_queue: queue.Queue | None = None
    if args.hands and args.headless is None:
        payload_queue = queue.Queue(maxsize=2)
        pipeline = VisionPipeline(result_queue=payload_queue,
                                  width=width, height=height, max_hands=1)
        pipeline.start()

    headless = args.headless is not None
    total = args.headless if headless else None
    frames = 0
    costs: list[float] = []
    started = time.perf_counter()

    try:
        while total is None or frames < total:
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            if pipeline is not None:
                try:
                    payload = payload_queue.get_nowait()
                except queue.Empty:
                    payload = None
                if payload and payload.get("hand_visible"):
                    # index_pos is pixels, not a normalised centre - the
                    # payload has no normalised key, and dividing here is
                    # closer to what a title actually does with it.
                    px, py = payload["index_pos"]
                    transform.ry = (px / max(1, width) - 0.5) * 360.0
                    transform.rx = (py / max(1, height) - 0.5) * 180.0
            else:
                transform.ry = (frames * 1.4) % 360.0
                transform.rx = (frames * 0.6) % 360.0

            tick = time.perf_counter()
            renderer.render_scene(frame, [(mesh, transform, material)])
            costs.append((time.perf_counter() - tick) * 1000.0)

            frames += 1
            if headless:
                continue

            cv2.putText(frame, f"{args.mesh}  {costs[-1]:.1f} ms",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (200, 220, 255), 2, cv2.LINE_AA)
            cv2.imshow("3D scratch", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if pipeline is not None:
            pipeline.stop()
        if not headless:
            cv2.destroyAllWindows()

    if costs:
        costs.sort()
        elapsed = time.perf_counter() - started
        print(f"  {args.mesh}: {frames} frames in {elapsed:,.1f}s - "
              f"render mean {sum(costs) / len(costs):.2f} ms, "
              f"p95 {costs[int(len(costs) * 0.95) - 1]:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
