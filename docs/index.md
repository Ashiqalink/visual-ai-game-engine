# Visual AI Game Engine Documentation

A high-performance Computer Vision and AI physics tracking SDK for game developers.

## Architecture

The SDK bridges camera-based vision tracking with game engine mechanics:

- **C++ Physics Engine (`src/engine.cpp`, `src/engine.hpp`)**: Native 60+ FPS physics engine compiled via `pybind11` (`engine_core`).
- **Python Fallback Engine (`src/visual_ai/fallback_engine.py`)**: Pure Python fallback when C++ binaries are not compiled.
- **Vision Pipeline (`src/visual_ai/pipeline.py`)**: Multithreaded MediaPipe / OpenCV camera processor delivering real-time target coordinates.

```
+---------------------+        Thread-Safe Queue        +-------------------------+
|   VisionPipeline    |  ============================>  |       GameEngine        |
|  (MediaPipe/OpenCV) |    {"target_x", "target_y"}     | (C++ Core / Py Fallback)|
+---------------------+                                 +-------------------------+
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/user/visual_ai_game_engine.git
cd visual_ai_game_engine

# Install library in editable mode
pip install -e .
```

### Running the Demo

```bash
python docs/examples/demo.py
```

### Running Unit & Integration Tests

```bash
python -m unittest discover -s tests
```

## API Reference

### `visual_ai.GameEngine(width: float, height: float)`
Primary game engine physics class. Automatically uses C++ acceleration if `engine_core` binary is available, otherwise uses `PythonFallbackEngine`.

#### Methods:
- `set_target_position(x: float, y: float)`: Updates the target coordinates for gravitational attraction.
- `update(dt: float)`: Advances physics loop by `dt` seconds.
- `get_x() -> float`: Returns sprite X coordinate.
- `get_y() -> float`: Returns sprite Y coordinate.
- `get_target_x() -> float`: Returns target X coordinate.
- `get_target_y() -> float`: Returns target Y coordinate.

### `visual_ai.VisionPipeline(result_queue: queue.Queue, width: int = 800, height: int = 600)`
Background worker thread capturing camera feed and performing real-time facial landmark detection.

#### Methods:
- `start()`: Launches thread.
- `stop()`: Signals background thread to stop cleanly.
