"""
Visual AI Game Engine Library
High-performance computer vision and AI physics tracking SDK for game developers.
"""

from visual_ai.pipeline import VisionPipeline
from visual_ai.fallback_engine import PythonFallbackEngine, Entity
from visual_ai.material import Material, ShaderType
from visual_ai.noise_filter import NoiseFilter, FilteredGestureDetector, PipelineNoiseFilter, GenericStreamFilter
from visual_ai.render3d import Transform3D, Camera3D, Mesh3D, Renderer3D
from visual_ai.math_utils import (
    Vector2,
    Vector3,
    Transform2D,
    map_range,
    remap_camera_roi_to_game,
    lerp,
    slerp_quaternion,
    ease_in_quad,
    ease_out_quad,
    ease_in_out_sine,
    spring,
    integrate_euler,
    calculate_drag_force,
    intersect_aabb_aabb,
    intersect_circle_circle,
    intersect_circle_aabb,
    predict_projectile_trajectory,
    FixedTimestepAccumulator,
    Tween,
    SeededRNG,
    weighted_choice,
    perlin_noise_1d,
)
from visual_ai.gesture_math import (
    get_landmark_distance,
    get_finger_angle,
    get_hand_center_and_radius,
    get_landmark_velocity,
)

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
    "GenericStreamFilter",
    "Material",
    "ShaderType",
    "Entity",
    "Transform3D",
    "Camera3D",
    "Mesh3D",
    "Renderer3D",
    # Core Math Utils
    "Vector2",
    "Vector3",
    "Transform2D",
    "map_range",
    "remap_camera_roi_to_game",
    "lerp",
    "slerp_quaternion",
    "ease_in_quad",
    "ease_out_quad",
    "ease_in_out_sine",
    "spring",
    "integrate_euler",
    "calculate_drag_force",
    "intersect_aabb_aabb",
    "intersect_circle_circle",
    "intersect_circle_aabb",
    "predict_projectile_trajectory",
    "FixedTimestepAccumulator",
    "Tween",
    "SeededRNG",
    "weighted_choice",
    "perlin_noise_1d",
    # Gesture Math
    "get_landmark_distance",
    "get_finger_angle",
    "get_hand_center_and_radius",
    "get_landmark_velocity",
]
