"""Renderer3D - software rasteriser: depth-sorted, flat-shaded faces onto an OpenCV frame."""

import math

import cv2
import numpy as np

from visual_ai.material import Material
from visual_ai.three_d.camera import Camera3D
from visual_ai.three_d.mesh import Mesh3D
from visual_ai.three_d.transform import Transform3D


class Renderer3D:
    """
    Software 3D mesh renderer: depth-sorted, shaded primitives drawn onto
    OpenCV image frames.
    """
    def __init__(self, camera: Camera3D | None = None, light_angle_deg: float = 45.0,
                 ambient_intensity: float = 0.45, light_intensity: float = 0.85):
        self.camera = camera if camera is not None else Camera3D()
        self.ambient_intensity = ambient_intensity
        self.light_intensity = light_intensity
        self.set_light_angle(light_angle_deg)

    def set_light_angle(self, angle_deg: float):
        """Set directional light vector based on angle in degrees."""
        rad = math.radians(angle_deg)
        lx = math.cos(rad)
        ly = math.sin(rad)
        lz = 0.85
        self.light_dir = np.array([lx, ly, lz], dtype=np.float64)
        self.light_dir /= np.linalg.norm(self.light_dir)

    def _face_intensity(self, world_verts: np.ndarray, vertex_idx: np.ndarray) -> np.ndarray:
        """
        Flat-shading intensity for a whole bucket of faces of equal vertex count.

        ``vertex_idx`` is (F, K); the normal comes from the first three
        vertices of each face, as the per-face version always did.
        """
        v0 = world_verts[vertex_idx[:, 0]]
        edge1 = world_verts[vertex_idx[:, 1]] - v0
        edge2 = world_verts[vertex_idx[:, 2]] - v0

        normals = np.cross(edge1, edge2)
        lengths = np.linalg.norm(normals, axis=1)
        degenerate = lengths <= 1e-6
        normals[~degenerate] /= lengths[~degenerate, None]
        normals[degenerate] = (0.0, 0.0, 1.0)

        # Directional diffuse lighting with configurable ambient floor
        dot_val = np.abs(normals @ self.light_dir)
        return np.maximum(self.ambient_intensity, np.minimum(
            1.0, self.ambient_intensity
            + (1.0 - self.ambient_intensity) * dot_val * self.light_intensity))

    def render_mesh(
        self,
        frame: np.ndarray,
        mesh: Mesh3D,
        transform: Transform3D,
        material: Material | None = None,
        wireframe: bool = False,
    ) -> np.ndarray:
        """
        Project and render a 3D Mesh onto an OpenCV image canvas.
        """
        self._paint(frame, self._collect_faces(mesh, transform, material,
                                               wireframe, item=0))
        return frame

    def render_scene(self, frame: np.ndarray, items, wireframe: bool = False) -> np.ndarray:
        """
        Render several meshes with ONE depth sort across all of them.

        ``items`` is an iterable of ``(mesh, transform, material)``.

        Calling `render_mesh` twice cannot do this: each call sorts and paints
        its own faces, so the second mesh lands entirely on top of the first
        however far behind it is. Sorting the faces of every mesh together is
        what lets one object pass through another — a hand whose fingers go
        behind a cup while the thumb stays in front of it. The sort is still
        the painter's algorithm, so two faces that actually intersect are
        ordered by their average depth and one of them wins the whole overlap;
        separate objects are what this is for.
        """
        faces = []
        for index, (mesh, transform, material) in enumerate(items):
            faces.extend(self._collect_faces(mesh, transform, material,
                                             wireframe, item=index))
        self._paint(frame, faces)
        return frame

    def _collect_faces(
        self,
        mesh: Mesh3D,
        transform: Transform3D,
        material: Material | None = None,
        wireframe: bool = False,
        item: int = 0,
    ) -> list:
        """
        One mesh's visible faces, shaded and projected, ready for `_paint`.

        Each entry is ``(depth, item, face number, screen points, colour,
        opacity, wireframe)``. Shading is resolved here rather than at paint
        time so a painted face carries no reference back to its mesh — which
        is what lets `render_scene` interleave faces from several of them.
        """
        if material is None:
            material = Material()

        # 1. Transform vertices to world coordinates
        world_verts = transform.transform_points(mesh.vertices)

        # 2. Project points with camera
        screen_coords, depths, valid = self.camera.project_points(world_verts)

        # Parse base color (RGB 0-255)
        color_rgba = material.base_color
        bgr = (int(color_rgba[2] * 255), int(color_rgba[1] * 255), int(color_rgba[0] * 255))

        # 3. Depth, backface cull and shading, a bucket of same-sized faces at
        #    a time (Painter's algorithm). Everything here used to be a numpy
        #    call per face per frame.
        screen_int = screen_coords.astype(np.int32)
        render_faces = []   # see the docstring for the entry shape

        for vertex_idx, face_numbers in mesh.face_groups():
            on_screen = valid[vertex_idx].all(axis=1)
            if not on_screen.any():
                continue
            vertex_idx = vertex_idx[on_screen]
            face_numbers = face_numbers[on_screen]
            pts = screen_int[vertex_idx]                       # (F, K, 2)

            if pts.shape[1] >= 3:
                # Backface culling in 2D screen space (skip polygons wound
                # counter-clockwise / facing away)
                cross_z = ((pts[:, 1, 0] - pts[:, 0, 0]) * (pts[:, 2, 1] - pts[:, 0, 1])
                           - (pts[:, 1, 1] - pts[:, 0, 1]) * (pts[:, 2, 0] - pts[:, 0, 0]))
                facing = cross_z > 0
                if not facing.any():
                    continue
                vertex_idx = vertex_idx[facing]
                face_numbers = face_numbers[facing]
                pts = pts[facing]
                shades = None if wireframe else self._face_intensity(world_verts, vertex_idx)
            else:
                # Too few vertices to have a normal — drawn unshaded, as before.
                shades = None if wireframe else np.ones(len(vertex_idx))

            avg_depth = depths[vertex_idx].mean(axis=1)
            for i in range(len(vertex_idx)):
                intensity = 1.0 if shades is None else shades[i]
                shaded_bgr = bgr if wireframe else (
                    int(min(255, bgr[0] * intensity)),
                    int(min(255, bgr[1] * intensity)),
                    int(min(255, bgr[2] * intensity)),
                )
                render_faces.append((avg_depth[i], item, face_numbers[i],
                                     pts[i], shaded_bgr, material.opacity,
                                     wireframe))
        return render_faces

    def _paint(self, frame: np.ndarray, render_faces: list) -> np.ndarray:
        """Draw collected faces back to front. See `_collect_faces`."""
        # Sort faces back to front (largest depth first); equal depths keep
        # mesh order, which is what the old stable sort over faces gave, and
        # then the order the meshes were handed over in.
        render_faces.sort(key=lambda face: (-face[0], face[1], face[2]))

        frame_h, frame_w = frame.shape[:2]
        for _, _, _, pts, shaded_bgr, opacity, wireframe in render_faces:
            if wireframe:
                cv2.polylines(frame, [pts], isClosed=True, color=shaded_bgr,
                              thickness=1, lineType=cv2.LINE_AA)
            else:
                if opacity < 0.99:
                    min_x = max(0, int(pts[:, 0].min()))
                    max_x = min(frame_w, int(pts[:, 0].max()) + 1)
                    min_y = max(0, int(pts[:, 1].min()))
                    max_y = min(frame_h, int(pts[:, 1].max()) + 1)
                    if max_x > min_x and max_y > min_y:
                        sub_pts = pts - np.array([min_x, min_y], dtype=np.int32)
                        roi = frame[min_y:max_y, min_x:max_x]
                        overlay = roi.copy()
                        cv2.fillPoly(overlay, [sub_pts], shaded_bgr, lineType=cv2.LINE_AA)
                        cv2.addWeighted(overlay, opacity, roi,
                                        1.0 - opacity, 0, roi)
                else:
                    cv2.fillPoly(frame, [pts], shaded_bgr, lineType=cv2.LINE_AA)

        return frame
