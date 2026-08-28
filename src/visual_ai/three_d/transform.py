"""Transform3D - position, Euler rotation (degrees) and scale of one 3D object."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Transform3D:
    """
    3D Transformation representing position, rotation (degrees), and scale.

    The nine components are the storage; `position`, `rotation` and `scale`
    are the same three triples by name. Both spellings existed in callers
    already — every consumer here passed components to the constructor, and
    sculptor assigned the triples — but only the components were real, so
    sculptor's turntable set three attributes that nothing read and its model
    sat at the origin, unrotated and unscaled, for as long as the game has
    existed.

    ``slots=True`` is the half that keeps this fixed: an assignment to a name
    that is not one of these now raises instead of quietly creating a field the
    renderer will never look at.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    sz: float = 1.0

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @position.setter
    def position(self, value) -> None:
        self.x, self.y, self.z = (float(v) for v in value)

    @property
    def rotation(self) -> tuple[float, float, float]:
        """Euler angles in degrees, in the order the matrix applies them."""
        return (self.rx, self.ry, self.rz)

    @rotation.setter
    def rotation(self, value) -> None:
        self.rx, self.ry, self.rz = (float(v) for v in value)

    @property
    def scale(self) -> tuple[float, float, float]:
        return (self.sx, self.sy, self.sz)

    @scale.setter
    def scale(self, value) -> None:
        self.sx, self.sy, self.sz = (float(v) for v in value)

    def get_rotation_matrix(self) -> np.ndarray:
        """Calculate 3x3 rotation matrix from Euler angles (in degrees)."""
        rad_x = math.radians(self.rx)
        rad_y = math.radians(self.ry)
        rad_z = math.radians(self.rz)

        cx, sx = math.cos(rad_x), math.sin(rad_x)
        cy, sy = math.cos(rad_y), math.sin(rad_y)
        cz, sz = math.cos(rad_z), math.sin(rad_z)

        # Rx * Ry * Rz
        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ], dtype=np.float64)

        Ry = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ], dtype=np.float64)

        Rz = np.array([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        return Rz @ Ry @ Rx

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform N x 3 array of local points to world coordinates.
        points: np.ndarray of shape (N, 3)
        Returns: np.ndarray of shape (N, 3)
        """
        if len(points) == 0:
            return points.copy()

        # Scale
        scaled = points * np.array([self.sx, self.sy, self.sz], dtype=np.float64)
        # Rotate
        R = self.get_rotation_matrix()
        rotated = (R @ scaled.T).T
        # Translate
        translated = rotated + np.array([self.x, self.y, self.z], dtype=np.float64)
        return translated
