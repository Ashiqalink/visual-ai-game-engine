"""
Visual AI Game Engine Library
High-performance computer vision and AI physics tracking SDK for game developers.
"""

from visual_ai.pipeline import VisionPipeline
from visual_ai.fallback_engine import PythonFallbackEngine, Entity
from visual_ai.material import Material, ShaderType
from visual_ai.noise_filter import NoiseFilter, FilteredGestureDetector, PipelineNoiseFilter
from visual_ai.render3d import Transform3D, Camera3D, Mesh3D, Renderer3D

try:
    import engine_core
    CPP_ENGINE_AVAILABLE = True
    GameEngine = engine_core.GameEngine
except ImportError:
    CPP_ENGINE_AVAILABLE = False
    GameEngine = PythonFallbackEngine

__version__ = "0.3.0"
__all__ = [
    "VisionPipeline",
    "PythonFallbackEngine",
    "GameEngine",
    "CPP_ENGINE_AVAILABLE",
    "NoiseFilter",
    "FilteredGestureDetector",
    "PipelineNoiseFilter",
    "Material",
    "ShaderType",
    "Entity",
    "Transform3D",
    "Camera3D",
    "Mesh3D",
    "Renderer3D",
]
