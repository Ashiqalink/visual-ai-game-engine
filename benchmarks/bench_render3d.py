"""
bench_render3d.py - what one frame of Renderer3D costs, per mesh.

The renderer is a pure-Python software rasteriser, so its cost is the game's
frame budget: at 60 fps everything else in a title has to fit in what this
does not use. Sculptor measures the same thing under a live camera, which is
the honest number but not a repeatable one - it moves with the lighting, the
tracker, and what the hands happen to be doing. This is the repeatable half:
no camera, no window, no pipeline, one spinning mesh drawn into a black frame.

What it reports per mesh: faces submitted, mean, p50 and p95 milliseconds per
render_scene call, and the fps that mean implies. p95 matters more than the
mean - a renderer that averages 8 ms and spikes to 30 ms drops frames, and the
mean will not say so.

    play3d bench                        every mesh at 1280x720
    play3d bench --width 1920 --height 1080
    play3d bench --frames 200 --mesh sphere
    play3d bench --hands                add two hand-sized meshes to the scene
    play3d bench --json                 machine-readable

The rotation is not decoration: a fixed pose lets the same faces cull the same
way every frame, which measures one pose rather than the mesh.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from visual_ai import (
    Camera3D,
    Material,
    Mesh3D,
    Renderer3D,
    Transform3D,
    voxel,
)

#: Every primitive the engine builds, at the size a title would use it.
MESHES = {
    "cube": lambda: Mesh3D.create_cube(size=70.0),
    "pyramid": lambda: Mesh3D.create_pyramid(width=60.0, height=80.0),
    "sphere": lambda: Mesh3D.create_sphere(radius=35.0, rings=8, sectors=12),
    "sphere-hi": lambda: Mesh3D.create_sphere(radius=35.0, rings=16, sectors=24),
    "cylinder": lambda: Mesh3D.create_cylinder(radius=30.0, height=70.0, segments=10),
    "capsule": lambda: Mesh3D.create_capsule(radius=20.0, height=40.0),
    "torus": lambda: Mesh3D.create_torus(ring_radius=30.0, tube_radius=10.0),
    "prism": lambda: Mesh3D.create_prism(width=40.0, height=40.0),
    "icosahedron": lambda: Mesh3D.create_icosahedron(radius=30.0),
}

#: The same shapes as occupancy grids, meshed into exposed quads only. Two
#: orders of magnitude more faces than the primitives above — a smooth cube is
#: 6 faces, a voxel one 864 — which is the range this bench is for.
MESHES.update({f"voxel-{name}": builder
               for name, builder in voxel.MODELS.items()})


def _face_count(mesh) -> int:
    faces = getattr(mesh, "faces", None)
    if faces is None:
        return 0
    return len(faces)


def _hand_items(material):
    """
    Two hands' worth of small meshes, one per landmark.

    Sculptor draws both hands as geometry through the same depth sort as the
    model, and that - not the model - is where its face count comes from. A
    bench that only ever draws one mesh would be measuring a scene no title
    actually renders.
    """
    items = []
    blob = Mesh3D.create_cube(size=8.0)
    for hand in range(2):
        for landmark in range(21):
            angle = landmark / 21.0 * 6.28318
            transform = Transform3D(
                x=(-1.0 if hand == 0 else 1.0) * 120.0 + 40.0 * np.cos(angle),
                y=40.0 * np.sin(angle),
                z=-60.0 + 4.0 * landmark,
            )
            items.append((blob, transform, material))
    return items


def bench_one(name: str, builder, width: int, height: int, frames: int,
              warmup: int, hands: bool) -> dict:
    mesh = builder()
    material = Material.preset("gold")
    camera = Camera3D(fov=60.0, screen_width=width, screen_height=height,
                      position=(0.0, 0.0, 500.0))
    renderer = Renderer3D(camera=camera)
    transform = Transform3D(x=0.0, y=0.0, z=0.0, sx=2.0, sy=2.0, sz=2.0)
    extra = _hand_items(material) if hands else []

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    samples: list[float] = []
    for index in range(warmup + frames):
        transform.rx = (index * 1.7) % 360.0
        transform.ry = (index * 2.3) % 360.0
        frame[:] = 0
        started = time.perf_counter()
        renderer.render_scene(frame, [(mesh, transform, material), *extra])
        elapsed = (time.perf_counter() - started) * 1000.0
        if index >= warmup:                    # the first frames pay for numpy
            samples.append(elapsed)            # warming up, not for the mesh

    samples.sort()
    faces = _face_count(mesh) + len(extra) * _face_count(Mesh3D.create_cube(size=8.0))
    mean = statistics.fmean(samples)
    return {
        "mesh": name,
        "faces": faces,
        "mean_ms": round(mean, 3),
        "p50_ms": round(samples[len(samples) // 2], 3),
        "p95_ms": round(samples[min(len(samples) - 1, int(len(samples) * 0.95))], 3),
        "max_ms": round(samples[-1], 3),
        "fps_at_mean": round(1000.0 / mean, 1) if mean else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-mesh Renderer3D frame cost.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=120,
                        help="measured frames per mesh (default 120)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="unmeasured frames first (default 10)")
    parser.add_argument("--mesh", action="append", metavar="NAME",
                        help="only this mesh; repeatable")
    parser.add_argument("--hands", action="store_true",
                        help="also draw 42 landmark blobs, as sculptor does")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    args = parser.parse_args(argv)

    wanted = args.mesh or list(MESHES)
    unknown = [name for name in wanted if name not in MESHES]
    if unknown:
        print(f"unknown mesh: {', '.join(unknown)}", file=sys.stderr)
        print(f"known: {', '.join(MESHES)}", file=sys.stderr)
        return 2

    rows = [bench_one(name, MESHES[name], args.width, args.height,
                      args.frames, args.warmup, args.hands)
            for name in wanted]

    if args.json:
        print(json.dumps({"width": args.width, "height": args.height,
                          "frames": args.frames, "hands": args.hands,
                          "results": rows}, indent=2))
        return 0

    scene = "mesh + 42 hand blobs" if args.hands else "one mesh"
    print(f"\n  Renderer3D at {args.width}x{args.height}, {args.frames} frames, {scene}\n")
    print(f"  {'mesh'.ljust(13)} {'faces'.rjust(6)} {'mean'.rjust(8)} "
          f"{'p50'.rjust(8)} {'p95'.rjust(8)} {'max'.rjust(8)} {'fps'.rjust(7)}")
    for row in rows:
        print(f"  {row['mesh'].ljust(13)} {str(row['faces']).rjust(6)} "
              f"{row['mean_ms']:8.2f} {row['p50_ms']:8.2f} {row['p95_ms']:8.2f} "
              f"{row['max_ms']:8.2f} {row['fps_at_mean']:7.1f}")
    print(f"\n  {'ms per render_scene call; fps is what the mean alone would allow'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
