# Visual AI Game Engine SDK (`visual_ai`)

High-performance computer vision AI tracking and physics SDK for game developers.

`visual_ai` allows game developers to easily capture webcam face/gesture coordinates in a background thread and bind them to real-time game entity physics using C++ high-speed pybind11 modules (with zero-config Python fallback support).

---

## 📁 Project Structure

```
visual_ai_game_engine/
├── pyproject.toml          # PEP 517 packaging (setuptools + pybind11)
├── setup.py                # Builds the engine_core extension
├── CMakeLists.txt          # CMake build of the same extension
├── pytest.ini              # pythonpath=src, testpaths=tests
├── src/
│   ├── engine.hpp/.cpp     # C++ physics core (GameEngine, blocks, debris)
│   ├── bridge.cpp          # pybind11 bindings for the physics core
│   ├── raster3d.*          # C++ 3D rasteriser (+ raster3d_bind.cpp)
│   ├── voxelmesh.*         # C++ voxel mesher (+ voxelmesh_bind.cpp)
│   ├── gridops.*           # C++ grid morphology (+ gridops_bind.cpp)
│   └── visual_ai/          # Python package
│       ├── __init__.py     # Public API; picks engine_core, else the Python fallback
│       ├── pipeline.py     # VisionPipeline: threaded camera face/hand tracking
│       ├── fallback_engine.py # Pure-Python physics engine (parity with engine.cpp)
│       ├── capture.py, accel.py, openvino_*.py  # Camera backends, accelerator policy
│       ├── depth_*.py, tof_stabilizer.py        # Depth sources and stabilisers
│       ├── gesture_*.py, noise_filter.py        # Hand-sign math, MLP classifier, filters
│       ├── imaging.py, low_light.py, matting.py, segment.py, spritegen.py
│       ├── display/        # SDL / null display backends and frame pacing
│       └── three_d/        # Camera, transform, mesh, voxel, software renderer
├── tests/                  # pytest suite (test_*.py)
├── benchmarks/             # Per-stage cost, accelerator and filter benches
├── examples/               # demo.py, air_draw.py, playground.py, three_d/
├── tools/                  # Gesture sample capture / MLP training, depth recording
├── docs/                   # index.md manual, opencv_integration_challenges.md
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Installation

### 1. Install Library

Clone the repository and install in editable mode:

```bash
pip install -e .
```

### 2. Build C++ Core Extension (Optional for max FPS)

```bash
python setup.py build_ext --inplace
```

---

## 🧪 Running Tests

The suite is plain `pytest`; `pytest.ini` puts `src/` on the path so it runs
against the source tree, not an installed copy:

```bash
python -m pytest -q
```

---

## 💻 Usage Example

Integrating `visual_ai` into your game loop is simple:

```python
import queue
from visual_ai import VisionPipeline, GameEngine

# 1. Initialize Thread-Safe Queue and Engine
ai_queue = queue.Queue(maxsize=2)
engine = GameEngine(width=800.0, height=600.0)

# 2. Start Vision Thread
pipeline = VisionPipeline(result_queue=ai_queue, width=800, height=600)
pipeline.start()

# 3. Game Loop
try:
    while True:
        # Fetch latest AI vision coordinates if ready
        try:
            ai_data = ai_queue.get_nowait()
            engine.set_target_position(ai_data["target_x"], ai_data["target_y"])
        except queue.Empty:
            pass

        # Update physics
        engine.update(dt=0.016)

        # Get entity position for rendering
        entity_x = engine.get_x()
        entity_y = engine.get_y()
finally:
    pipeline.stop()
```

---

## 📜 License

**Source-available, personal-use license. Not open source.**

You may download, build and run this software yourself, for your own
non-commercial use. You may not copy, redistribute, publish, sublicense,
sell, or create derivative works from it, in whole or in part. All other
rights are reserved by the copyright holder.

There is deliberately no LICENSE file: every license GitHub offers grants
redistribution rights, which this one does not.
