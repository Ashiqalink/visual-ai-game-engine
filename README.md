# Visual AI Game Engine SDK (`visual_ai`)

High-performance computer vision AI tracking and physics SDK for game developers.

`visual_ai` allows game developers to easily capture webcam face/gesture coordinates in a background thread and bind them to real-time game entity physics using C++ high-speed pybind11 modules (with zero-config Python fallback support).

---

## 📁 Project Structure

```
visual_ai_game_engine/
├── pyproject.toml          # Modern PEP 517 packaging configuration
├── setup.py                # Legacy setuptools build configuration
├── CMakeLists.txt          # CMake build script for C++ core module
├── src/                    # Source directory (C++ core & Python library)
│   ├── engine.hpp          # C++ core header
│   ├── engine.cpp          # Core physics & game logic
│   ├── bridge.cpp          # pybind11 C++/Python bindings
│   └── visual_ai/          # Python library package
│       ├── __init__.py     # SDK entry point & engine selector
│       ├── pipeline.py     # Threaded camera vision detector
│       └── fallback_engine.py # Python fallback engine
├── tests/                  # Automated test suite
│   ├── test_engine.py      # Unit tests for physics engine
│   └── integration_test.py # Full pipeline integration tests
├── docs/                   # Documentation & developer examples
│   ├── index.md            # Comprehensive user manual
│   └── examples/
│       └── demo.py         # Developer interactive demo
├── requirements.txt        # Package dependencies
└── README.md               # Project overview & quickstart guide
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

Run the automated test suite using `unittest`:

```bash
python -m unittest discover -s tests
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

Distributed under the MIT License. See `LICENSE` for details.
