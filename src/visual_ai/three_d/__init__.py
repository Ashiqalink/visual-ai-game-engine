"""
visual_ai.three_d - the engine's 3D workspace.

Everything three-dimensional in the SDK lives under this package, apart from
what the 2D side shares with it (see README.md next to this file). Consumers
import the public names from ``visual_ai`` as before::

    from visual_ai import Camera3D, Mesh3D, Renderer3D, Transform3D

or from here directly. ``visual_ai.render3d`` still resolves, as a shim over
this package, so nothing downstream has to move on the same day.
"""

from visual_ai.material import Material, ShaderType
from visual_ai.three_d.camera import Camera3D
from visual_ai.three_d.mesh import Mesh3D
from visual_ai.three_d.renderer import Renderer3D
from visual_ai.three_d.transform import Transform3D
from visual_ai.three_d import voxel

__all__ = [
    "Camera3D",
    "Material",
    "Mesh3D",
    "Renderer3D",
    "ShaderType",
    "Transform3D",
    "voxel",
]
