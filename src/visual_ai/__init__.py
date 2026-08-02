"""
Visual AI Game Engine Library
High-performance computer vision and AI physics tracking SDK for game developers.
"""

from visual_ai.pipeline import VisionPipeline
from visual_ai.fallback_engine import PythonFallbackEngine
from visual_ai.noise_filter import NoiseFilter, FilteredGestureDetector, PipelineNoiseFilter

try:
    import engine_core
    CPP_ENGINE_AVAILABLE = True
    GameEngine = engine_core.GameEngine
except ImportError:
    CPP_ENGINE_AVAILABLE = False
    GameEngine = PythonFallbackEngine

__version__ = "0.2.0"
__all__ = [
    "VisionPipeline",
    "PythonFallbackEngine",
    "GameEngine",
    "CPP_ENGINE_AVAILABLE",
    "NoiseFilter",
    "FilteredGestureDetector",
    "PipelineNoiseFilter",
]
