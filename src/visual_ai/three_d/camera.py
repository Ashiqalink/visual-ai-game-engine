"""Camera3D - perspective projection of world points onto the 2D frame."""

import math

import numpy as np


class Camera3D:
    """
    3D Perspective Camera for projecting 3D world coordinates onto 2D image plane.
    """
    def __init__(
        self,
        fov: float = 60.0,
        near: float = 0.1,
        position: tuple[float, float, float] = (0.0, 0.0, 500.0),
        screen_width: float = 800.0,
        screen_height: float = 600.0,
    ):
        # The projection is position + focal length only — no look-at, no far
        # clip. The old aspect_ratio/far/target/up parameters were stored and
        # never read, silently ignoring whatever callers passed.
        self.fov = fov
        self.near = near
        self.position = np.array(position, dtype=np.float64)
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Focal distance factor derived from FOV
        self.focal_length = (self.screen_width / 2.0) / math.tan(math.radians(self.fov / 2.0))

    def project_point(self, point: tuple[float, float, float]) -> tuple[int, int, float] | None:
        """
        Project a single 3D world point (x, y, z) into 2D screen coordinates (px_x, px_y, z_depth).
        Returns None if behind camera near plane.

        A one-row call into :meth:`project_points`, so the projection maths
        lives in exactly one place.
        """
        coords, depths, valid = self.project_points(
            np.array([point], dtype=np.float64))
        if not valid[0]:
            return None
        return (int(round(coords[0, 0])), int(round(coords[0, 1])), float(depths[0]))

    def project_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project array of N x 3 points.
        Returns (screen_coords [N x 2], depths [N], valid_mask [N])
        """
        N = len(points)
        screen_coords = np.zeros((N, 2), dtype=np.float64)
        depths = np.zeros(N, dtype=np.float64)
        valid = np.zeros(N, dtype=bool)

        if N == 0:
            return screen_coords, depths, valid

        rel_x = points[:, 0] - self.position[0]
        rel_y = points[:, 1] - self.position[1]
        rel_z = self.position[2] - points[:, 2]

        valid_mask = rel_z > self.near
        valid[:] = valid_mask

        # Avoid div by zero for invalid points
        safe_z = np.where(valid_mask, rel_z, 1.0)

        screen_coords[:, 0] = (rel_x * self.focal_length / safe_z) + (self.screen_width / 2.0)
        screen_coords[:, 1] = (-rel_y * self.focal_length / safe_z) + (self.screen_height / 2.0)
        depths[:] = rel_z

        return screen_coords, depths, valid
