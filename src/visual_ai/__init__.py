"""
Visual AI Game Engine Library
High-performance computer vision and AI physics tracking SDK for game developers.
"""

from visual_ai.pipeline import VisionPipeline
from visual_ai.fallback_engine import PythonFallbackEngine, Entity
from visual_ai.material import Material, ShaderType
from visual_ai.noise_filter import (
    NoiseFilter,
    FilteredGestureDetector,
    PipelineNoiseFilter,
    GenericStreamFilter,
    OneEuroFilter,
    ema_alpha_to_cutoff,
)
from visual_ai.render3d import Transform3D, Camera3D, Mesh3D, Renderer3D
from visual_ai.tof_stabilizer import DepthStabilizer, ToFStabilizer
from visual_ai.jitter_analyzer import JitterAnalyzer
from visual_ai.low_light import LowLightBoost, measure_luma
from visual_ai.capture import default_backend, open_camera
from visual_ai.depth_source import (
    DepthRecorder,
    DepthSource,
    DepthStream,
    ReplayDepthSource,
    SyntheticDepthSource,
    open_depth_source,
    probe_depth_sources,
)
from visual_ai.segment import PersonSegmenter
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
from visual_ai.gesture_mlp import (
    GestureMLP,
    landmarks_to_features,
)
from visual_ai.spritegen import (
    CreatureSpec,
    render_creature,
    render_views,
    spec_from_dict,
    spec_to_dict,
    cast_by_name,
    DEFAULT_CAST,
    BODY_SHAPES,
    VIEWS,
)
from visual_ai import imaging
from visual_ai.imaging import (
    load_image,
    save_png,
    to_rgba,
    bgr_to_rgb,
    rgb_to_bgr,
    chroma_key,
    remove_background,
    normalize_lighting,
    clipped_fraction,
    clean_sprite,
    autocrop,
    pad_to,
    resize,
    bleed_edges,
    background_uniformity,
    composite_over,
    blit_sprite,
    blit_ellipse_alpha,
    invalidate_sprite_cache,
)
from visual_ai.matting import (
    PortraitMatter,
    portrait_matte,
    cut_out_person,
    MODNET_AVAILABLE,
)


def __getattr__(name: str):
    # REMBG_AVAILABLE is served lazily: resolving it actually imports rembg
    # (~2 s of onnxruntime), which every game used to pay at startup whether
    # or not it ever removed a background. Importing it eagerly above would
    # defeat imaging's own deferral.
    if name == "REMBG_AVAILABLE":
        return imaging.rembg_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Re-exported third-party surface.
#
# Games are meant to depend on this package and nothing else, so the pieces of
# the wider ecosystem they genuinely need are surfaced here rather than being
# imported directly downstream. That keeps a game's requirements file one line
# long and puts version pinning in one place.
#
# This is a convenience layer, not encapsulation: ndarrays are already part of
# the public contract (queue payloads carry frames, `render_creature` returns
# one), so the engine could not hide NumPy even if it wanted to.
import numpy as np

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
    "OneEuroFilter",
    "ema_alpha_to_cutoff",
    "DepthStabilizer",
    "LowLightBoost",
    "measure_luma",
    "default_backend",
    "open_camera",
    "DepthSource",
    "DepthStream",
    "DepthRecorder",
    "SyntheticDepthSource",
    "ReplayDepthSource",
    "open_depth_source",
    "probe_depth_sources",
    "ToFStabilizer",
    "JitterAnalyzer",
    "PersonSegmenter",
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
    # Sprite generation
    "CreatureSpec",
    "render_creature",
    "render_views",
    "spec_from_dict",
    "spec_to_dict",
    "cast_by_name",
    "DEFAULT_CAST",
    "BODY_SHAPES",
    "VIEWS",
    # Imaging
    "imaging",
    "load_image",
    "save_png",
    "to_rgba",
    "bgr_to_rgb",
    "rgb_to_bgr",
    "chroma_key",
    "remove_background",
    "normalize_lighting",
    "clipped_fraction",
    "clean_sprite",
    "autocrop",
    "pad_to",
    "resize",
    "bleed_edges",
    "background_uniformity",
    "composite_over",
    "blit_sprite",
    "blit_ellipse_alpha",
    "invalidate_sprite_cache",
    "REMBG_AVAILABLE",
    # Portrait matting (MODNet)
    "PortraitMatter",
    "portrait_matte",
    "cut_out_person",
    "MODNET_AVAILABLE",
    # Re-exported dependencies
    "np",
]
