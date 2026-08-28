"""
Compatibility shim: the renderer moved to :mod:`visual_ai.three_d`.

``visual_ai.render3d`` was one 640-line module holding the transform, camera,
mesh and rasteriser together. They now live one class per module under
``visual_ai/three_d/`` so the 3D side has its own workspace. Existing imports
of this name keep working; new code imports from ``visual_ai.three_d``.
"""

from visual_ai.three_d import (  # noqa: F401 - re-exported on purpose
    Camera3D,
    Material,
    Mesh3D,
    Renderer3D,
    Transform3D,
)

__all__ = ["Camera3D", "Material", "Mesh3D", "Renderer3D", "Transform3D"]
