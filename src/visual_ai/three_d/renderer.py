"""Renderer3D - software rasteriser: depth-sorted, flat-shaded faces onto an OpenCV frame."""

import math

import cv2
import numpy as np

from visual_ai.material import Material
from visual_ai.three_d.camera import Camera3D
from visual_ai.three_d.mesh import Mesh3D
from visual_ai.three_d.transform import Transform3D


class _FaceBatch:
    """
    Collected faces as parallel sequences, one entry per face.

    ``depth``, ``item`` and ``face_no`` are arrays because they are the sort
    keys: the painter's order is then one `np.lexsort` rather than a Python
    sort over tuples. ``pts``, ``colors``, ``opacity`` and ``wireframe`` are
    lists the paint loop indexes directly, each built one call per bucket —
    which is what keeps `_collect_faces` free of per-face Python work.

    ``len`` is the face count, so a caller can still ask how many faces
    survived the cull.
    """

    __slots__ = ("depth", "item", "face_no", "pts", "colors", "opacity",
                 "wireframe")

    def __init__(self, depth, item, face_no, pts, colors, opacity, wireframe):
        self.depth = depth              # (F,) float: average depth
        self.item = item                # (F,) int: position in the scene
        self.face_no = face_no          # (F,) int: position in mesh.faces
        self.pts = pts                  # F arrays of (K, 2) int32 screen points
        self.colors = colors            # F shaded BGR triples
        self.opacity = opacity          # F floats
        self.wireframe = wireframe      # F bools

    def __len__(self) -> int:
        return len(self.pts)

    @classmethod
    def empty(cls) -> "_FaceBatch":
        return cls(np.empty(0, dtype=np.float64), np.empty(0, dtype=np.intp),
                   np.empty(0, dtype=np.intp), [], [], [], [])

    @classmethod
    def concat(cls, batches) -> "_FaceBatch":
        """One batch out of several, keeping the order they were given in."""
        batches = [batch for batch in batches if len(batch)]
        if not batches:
            return cls.empty()
        if len(batches) == 1:
            return batches[0]
        pts, colors, opacity, wireframe = [], [], [], []
        for batch in batches:
            pts.extend(batch.pts)
            colors.extend(batch.colors)
            opacity.extend(batch.opacity)
            wireframe.extend(batch.wireframe)
        return cls(np.concatenate([b.depth for b in batches]),
                   np.concatenate([b.item for b in batches]),
                   np.concatenate([b.face_no for b in batches]),
                   pts, colors, opacity, wireframe)


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
        self._paint(frame, _FaceBatch.concat([
            self._collect_faces(mesh, transform, material, wireframe, item=index)
            for index, (mesh, transform, material) in enumerate(items)]))
        return frame

    def _collect_faces(
        self,
        mesh: Mesh3D,
        transform: Transform3D,
        material: Material | None = None,
        wireframe: bool = False,
        item: int = 0,
    ) -> "_FaceBatch":
        """
        One mesh's visible faces, shaded and projected, ready for `_paint`.

        The result is a `_FaceBatch`: parallel sequences of depth, item, face
        number, screen points, colour, opacity and wireframe flag. Shading is
        resolved here rather than at paint time so a painted face carries no
        reference back to its mesh — which is what lets `render_scene`
        interleave faces from several of them.
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
        batches = []        # one _FaceBatch per bucket, joined on the way out

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
            count = len(vertex_idx)
            if shades is None:
                # Wireframe: the line is the material's colour, undimmed.
                colors = [bgr] * count
            else:
                colors = np.minimum(
                    255.0, np.asarray(bgr, dtype=np.float64) * shades[:, None]
                ).astype(np.int64).tolist()
            batches.append(_FaceBatch(
                avg_depth, np.full(count, item, dtype=np.intp),
                face_numbers.astype(np.intp, copy=False), list(pts), colors,
                [material.opacity] * count, [wireframe] * count))
        return _FaceBatch.concat(batches)

    def _paint(self, frame: np.ndarray, batch: "_FaceBatch") -> np.ndarray:
        """Draw collected faces back to front. See `_collect_faces`."""
        if not len(batch):
            return frame
        # Sort faces back to front (largest depth first); equal depths keep
        # mesh order, which is what the old stable sort over faces gave, and
        # then the order the meshes were handed over in. lexsort takes its
        # primary key last, and is stable, so this is that same ordering.
        order = np.lexsort((batch.face_no, batch.item, -batch.depth))

        frame_h, frame_w = frame.shape[:2]
        pts_all, colors = batch.pts, batch.colors
        opacity_all, wireframe_all = batch.opacity, batch.wireframe
        for i in order.tolist():
            pts, shaded_bgr = pts_all[i], colors[i]
            opacity, wireframe = opacity_all[i], wireframe_all[i]
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
