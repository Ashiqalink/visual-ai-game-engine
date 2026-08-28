# `visual_ai.three_d` — the 3D workspace

Scope label: **engine SDK / 3D**. Everything three-dimensional in the engine lives
here; the 2D side (`pipeline`, `fallback_engine`, `noise_filter`, `display`, …) stays
in the package root. New 3D work — loaders, scene graph, lighting, a compiled
rasteriser — goes under this directory, with its tests in `tests/three_d/` and its
demos in `examples/three_d/`.

## Layout

| module | holds |
| --- | --- |
| `transform.py` | `Transform3D` — position, Euler rotation (degrees), scale; `transform_points` |
| `camera.py` | `Camera3D` — perspective projection, position + focal length only |
| `mesh.py` | `Mesh3D` — vertices/faces, the `create_*` primitives, `face_groups()` cache |
| `renderer.py` | `Renderer3D` — depth-sorted flat-shaded software rasteriser onto an OpenCV frame |
| `__init__.py` | the public surface: the four classes above plus `Material`, `ShaderType` |

`visual_ai/render3d.py` is a shim that re-exports this package so existing
`from visual_ai.render3d import …` lines keep working. `visual_ai/__init__.py`
re-exports the same names, which is how the games import them.

## What is shared, and stays outside

- **`Material` / `ShaderType`** (`visual_ai/material.py`): used by the 2D engine's
  `Block` / `Debris` / `Entity` and by `add_block(material=)`, so it is not 3D-only.
  Re-exported here for convenience; do not move it.
- **`Entity` rotation / angular velocity / `add_3d_element`** (`src/engine.hpp`,
  `src/bridge.cpp`, `fallback_engine.py`): the physics for a 3D element runs in the
  shared `GameEngine`, in both the C++ core and the Python fallback. The queue payload
  schema, the pybind11 bindings and the physics constants need approval before they
  change (`CLAUDE.md`). A separate C++ 3D core would go in `src/three_d/` and mirror
  into this package the same way `engine.cpp` mirrors `fallback_engine.py`.
- **`math_utils.py`** Transform3D-adjacent helpers predate this package; leave them.

## Rules

- Nothing in `three_d/` imports from `pipeline`, `display`, or the games. The renderer
  takes an `np.ndarray` frame and returns it; how the frame got on screen is the 2D
  side's business.
- Add a public name in three places together: its module, `three_d/__init__.py`, and
  `visual_ai/__init__.py` (with `__all__`). The `render3d` shim only needs it if a
  downstream file imports it by that old path.
- Tests: `python -m pytest tests/three_d` (interpreter: `D:\visual\.venv`).
- Sculptor (`visual ai games/sculptor`) is the soak test for this renderer; `Sling/block.py`
  is its other consumer. Run `play sculptor` after a rendering change.
