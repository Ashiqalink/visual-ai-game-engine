"""
3D Element Rendering, Perspective Projection, and Geometry SDK for Visual AI Game Engine.
"""

import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from visual_ai.material import Material, ShaderType


@dataclass
class Transform3D:
    """3D Transformation representing position, rotation (degrees), and scale."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    sz: float = 1.0

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


class Camera3D:
    """
    3D Perspective Camera for projecting 3D world coordinates onto 2D image plane.
    """
    def __init__(
        self,
        fov: float = 60.0,
        aspect_ratio: float = 4.0 / 3.0,
        near: float = 0.1,
        far: float = 1000.0,
        position: Tuple[float, float, float] = (0.0, 0.0, 500.0),
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        screen_width: float = 800.0,
        screen_height: float = 600.0,
    ):
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far
        self.position = np.array(position, dtype=np.float64)
        self.target = np.array(target, dtype=np.float64)
        self.up = np.array(up, dtype=np.float64)
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Focal distance factor derived from FOV
        self.focal_length = (self.screen_width / 2.0) / math.tan(math.radians(self.fov / 2.0))

    def project_point(self, point: Tuple[float, float, float]) -> Optional[Tuple[int, int, float]]:
        """
        Project a single 3D world point (x, y, z) into 2D screen coordinates (px_x, px_y, z_depth).
        Returns None if behind camera near plane.
        """
        px, py, pz = point[0], point[1], point[2]
        
        # Camera-relative translation (assuming camera facing along -Z)
        rel_x = px - self.position[0]
        rel_y = py - self.position[1]
        rel_z = self.position[2] - pz  # depth away from camera

        if rel_z <= self.near:
            return None

        # Perspective projection
        screen_x = (rel_x * self.focal_length / rel_z) + (self.screen_width / 2.0)
        screen_y = (-rel_y * self.focal_length / rel_z) + (self.screen_height / 2.0)

        return (int(round(screen_x)), int(round(screen_y)), rel_z)

    def project_points(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


@dataclass
class Mesh3D:
    """
    3D Geometry representation holding vertices, faces, and normals.
    """
    vertices: np.ndarray  # N x 3 float64
    faces: List[List[int]]  # List of face vertex index lists
    normals: Optional[np.ndarray] = None  # Face or vertex normals

    @classmethod
    def create_cube(cls, size: float = 50.0) -> "Mesh3D":
        """Create a 3D Cube centered at origin."""
        s = size / 2.0
        verts = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # Front
            [-s, -s,  s], [s, -s,  s], [s, s,  s], [-s, s,  s],  # Back
        ], dtype=np.float64)

        faces = [
            [0, 1, 2, 3],  # Front
            [5, 4, 7, 6],  # Back
            [4, 0, 3, 7],  # Left
            [1, 5, 6, 2],  # Right
            [4, 5, 1, 0],  # Bottom
            [3, 2, 6, 7],  # Top
        ]
        return cls(vertices=verts, faces=faces)

    @classmethod
    def create_pyramid(cls, width: float = 50.0, height: float = 60.0) -> "Mesh3D":
        """Create a 3D Pyramid centered at origin."""
        w = width / 2.0
        h = height / 2.0
        verts = np.array([
            [-w, -h, -w], [w, -h, -w], [w, -h, w], [-w, -h, w],  # Base
            [0, h, 0]  # Apex
        ], dtype=np.float64)

        faces = [
            # Base wound to face down/outward — [0, 1, 2, 3] passed the screen-
            # space cull from below, so the underside painted through the mesh.
            [3, 2, 1, 0],  # Base
            [0, 1, 4],     # Side 1
            [1, 2, 4],     # Side 2
            [2, 3, 4],     # Side 3
            [3, 0, 4],     # Side 4
        ]
        return cls(vertices=verts, faces=faces)

    @classmethod
    def create_sphere(cls, radius: float = 30.0, rings: int = 8, sectors: int = 12) -> "Mesh3D":
        """Create a UV Sphere mesh."""
        verts = []
        faces = []

        R = 1.0 / float(rings - 1)
        S = 1.0 / float(sectors - 1)

        for r in range(rings):
            for s in range(sectors):
                y = math.sin(-math.pi / 2 + math.pi * r * R)
                x = math.cos(2 * math.pi * s * S) * math.sin(math.pi * r * R)
                z = math.sin(2 * math.pi * s * S) * math.sin(math.pi * r * R)
                verts.append([x * radius, y * radius, z * radius])

        for r in range(rings - 1):
            for s in range(sectors - 1):
                i1 = r * sectors + s
                i2 = r * sectors + (s + 1)
                i3 = (r + 1) * sectors + (s + 1)
                i4 = (r + 1) * sectors + s
                faces.append([i1, i2, i3, i4])

        return cls(vertices=np.array(verts, dtype=np.float64), faces=faces)

    @classmethod
    def create_cylinder(cls, radius: float = 25.0, height: float = 60.0, segments: int = 12) -> "Mesh3D":
        """Create a 3D Cylinder mesh."""
        verts = []
        faces = []
        half_h = height / 2.0

        # Top center (0) & Bottom center (1)
        verts.append([0.0, half_h, 0.0])
        verts.append([0.0, -half_h, 0.0])

        # Top ring, then bottom ring. The rings must be contiguous blocks:
        # the face indices below address top vertex i as 2 + i and bottom
        # vertex i as 2 + segments + i. The old interleaved append (top,
        # bottom, top, bottom, …) silently made every cap and side face a
        # zigzag mix of top and bottom vertices — the "caps" were not even
        # planar, so no winding could render them right.
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            verts.append([x, half_h, z])   # 2 + i
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            verts.append([x, -half_h, z])  # 2 + segments + i

        for i in range(segments):
            next_i = (i + 1) % segments
            top1 = 2 + i
            top2 = 2 + next_i
            bot1 = 2 + segments + i
            bot2 = 2 + segments + next_i

            # Cap windings match the side quads' outward orientation — the old
            # [0, top2, top1] / [1, bot1, bot2] order failed the screen-space
            # cull, so looking down at a cylinder hid the top disc and painted
            # the bottom one through the body.
            faces.append([0, top1, top2])
            # Bottom cap
            faces.append([1, bot2, bot1])
            # Side quad, wound to match the caps' outward orientation
            faces.append([top2, top1, bot1, bot2])

        return cls(vertices=np.array(verts, dtype=np.float64), faces=faces)

    @classmethod
    def create_capsule(cls, radius: float = 20.0, height: float = 40.0, rings: int = 6, sectors: int = 12) -> "Mesh3D":
        """Create a 3D Capsule (pill-shaped) character mesh."""
        verts = []
        faces = []
        half_h = height / 2.0

        # Hemisphere tops and bottoms
        R = 1.0 / float(rings - 1)
        S = 1.0 / float(sectors - 1)

        for r in range(rings):
            v_lat = r * R  # 0 to 1
            lat = -math.pi / 2 + math.pi * v_lat
            y_offset = half_h if lat >= 0 else -half_h
            for s in range(sectors):
                y = math.sin(lat) * radius + y_offset
                x = math.cos(2 * math.pi * s * S) * math.cos(lat) * radius
                z = math.sin(2 * math.pi * s * S) * math.cos(lat) * radius
                verts.append([x, y, z])

        for r in range(rings - 1):
            for s in range(sectors - 1):
                i1 = r * sectors + s
                i2 = r * sectors + (s + 1)
                i3 = (r + 1) * sectors + (s + 1)
                i4 = (r + 1) * sectors + s
                faces.append([i1, i2, i3, i4])

        return cls(vertices=np.array(verts, dtype=np.float64), faces=faces)

    @classmethod
    def create_torus(cls, ring_radius: float = 30.0, tube_radius: float = 10.0, ring_segments: int = 16, tube_segments: int = 8) -> "Mesh3D":
        """Create a 3D Torus (ring) character mesh."""
        verts = []
        faces = []

        for i in range(ring_segments):
            u = 2.0 * math.pi * i / ring_segments
            cos_u, sin_u = math.cos(u), math.sin(u)
            for j in range(tube_segments):
                v = 2.0 * math.pi * j / tube_segments
                cos_v, sin_v = math.cos(v), math.sin(v)

                x = (ring_radius + tube_radius * cos_v) * cos_u
                y = tube_radius * sin_v
                z = (ring_radius + tube_radius * cos_v) * sin_u
                verts.append([x, y, z])

        for i in range(ring_segments):
            next_i = (i + 1) % ring_segments
            for j in range(tube_segments):
                next_j = (j + 1) % tube_segments
                idx1 = i * tube_segments + j
                idx2 = next_i * tube_segments + j
                idx3 = next_i * tube_segments + next_j
                idx4 = i * tube_segments + next_j
                faces.append([idx1, idx2, idx3, idx4])

        return cls(vertices=np.array(verts, dtype=np.float64), faces=faces)

    @classmethod
    def create_prism(cls, width: float = 40.0, height: float = 40.0, depth: float = 40.0) -> "Mesh3D":
        """Create a 3D Triangular Prism character mesh."""
        w = width / 2.0
        h = height / 2.0
        d = depth / 2.0

        verts = np.array([
            [-w, -h, -d], [w, -h, -d], [0, h, -d],  # Front triangle
            [-w, -h,  d], [w, -h,  d], [0, h,  d],  # Back triangle
        ], dtype=np.float64)

        faces = [
            [0, 1, 2],        # Front
            [5, 4, 3],        # Back
            [0, 3, 4, 1],     # Bottom
            [1, 4, 5, 2],     # Right slant
            [2, 5, 3, 0],     # Left slant
        ]
        return cls(vertices=verts, faces=faces)

    @classmethod
    def create_icosahedron(cls, radius: float = 30.0) -> "Mesh3D":
        """Create a 3D Icosahedron (polyhedral sphere) character mesh."""
        t = (1.0 + math.sqrt(5.0)) / 2.0
        verts = np.array([
            [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
            [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
            [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
        ], dtype=np.float64)

        # Scale to target radius
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        verts = (verts / norms) * radius

        # The textbook CCW-outward winding is backwards under this renderer's
        # screen-space cull convention (every other primitive here winds the
        # opposite way): all 20 camera-facing triangles were culled and the 20
        # back faces drawn, so the mesh rendered inside-out. Each face is
        # reversed to match the cube/sphere/prism orientation.
        faces = [
            [5, 11, 0], [1, 5, 0], [7, 1, 0], [10, 7, 0], [11, 10, 0],
            [9, 5, 1], [4, 11, 5], [2, 10, 11], [6, 7, 10], [8, 1, 7],
            [4, 9, 3], [2, 4, 3], [6, 2, 3], [8, 6, 3], [9, 8, 3],
            [5, 9, 4], [11, 4, 2], [10, 2, 6], [7, 6, 8], [1, 8, 9]
        ]
        return cls(vertices=verts, faces=faces)

    def apply_extrusion_depth(self, depth_factor: float = 1.0) -> "Mesh3D":
        """Scale Z-axis vertices by depth_factor to adjust 3D extrusion thickness."""
        new_verts = self.vertices.copy()
        new_verts[:, 2] *= float(depth_factor)
        return Mesh3D(vertices=new_verts, faces=self.faces, normals=self.normals)

    def scale_non_uniform(self, sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> "Mesh3D":
        """Apply non-uniform 3D scaling to mesh vertices."""
        new_verts = self.vertices.copy()
        new_verts *= np.array([sx, sy, sz], dtype=np.float64)
        return Mesh3D(vertices=new_verts, faces=self.faces, normals=self.normals)


class Renderer3D:
    """
    Software 3D Mesh Renderer for drawing depth-sorted shaded 3D primitives onto OpenCV image frames.
    """
    def __init__(self, camera: Optional[Camera3D] = None, light_angle_deg: float = 45.0, ambient_intensity: float = 0.45, light_intensity: float = 0.85):
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

    def render_mesh(
        self,
        frame: np.ndarray,
        mesh: Mesh3D,
        transform: Transform3D,
        material: Optional[Material] = None,
        wireframe: bool = False,
    ) -> np.ndarray:
        """
        Project and render a 3D Mesh onto an OpenCV image canvas.
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

        # 3. Process faces for depth sorting (Painter's algorithm)
        render_faces = []
        for face in mesh.faces:
            face_indices = np.array(face)
            if not np.all(valid[face_indices]):
                continue

            # Average depth of face
            avg_depth = np.mean(depths[face_indices])
            render_faces.append((avg_depth, face_indices))

        # Sort faces back to front (largest depth first)
        render_faces.sort(key=lambda item: item[0], reverse=True)

        # 4. Render faces onto frame
        frame_h, frame_w = frame.shape[:2]
        for avg_depth, face_indices in render_faces:
            pts = screen_coords[face_indices].astype(np.int32)

            # Backface culling in 2D screen space (skip polygons wound counter-clockwise / facing away)
            if len(pts) >= 3:
                cross_z = (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) - (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[0][0])
                if cross_z <= 0:
                    continue

            if wireframe:
                cv2.polylines(frame, [pts], isClosed=True, color=bgr, thickness=1, lineType=cv2.LINE_AA)
            else:
                # Flat Shading calculation using normal
                if len(face_indices) >= 3:
                    v0 = world_verts[face_indices[0]]
                    v1 = world_verts[face_indices[1]]
                    v2 = world_verts[face_indices[2]]

                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    normal = np.cross(edge1, edge2)
                    norm_len = np.linalg.norm(normal)
                    if norm_len > 1e-6:
                        normal /= norm_len
                    else:
                        normal = np.array([0.0, 0.0, 1.0])

                    # Directional diffuse lighting with configurable ambient floor
                    dot_val = abs(float(np.dot(normal, self.light_dir)))
                    intensity = max(self.ambient_intensity, min(1.0, self.ambient_intensity + (1.0 - self.ambient_intensity) * dot_val * self.light_intensity))
                else:
                    intensity = 1.0

                shaded_bgr = (
                    int(min(255, bgr[0] * intensity)),
                    int(min(255, bgr[1] * intensity)),
                    int(min(255, bgr[2] * intensity)),
                )

                if material.opacity < 0.99:
                    min_x = max(0, int(pts[:, 0].min()))
                    max_x = min(frame_w, int(pts[:, 0].max()) + 1)
                    min_y = max(0, int(pts[:, 1].min()))
                    max_y = min(frame_h, int(pts[:, 1].max()) + 1)
                    if max_x > min_x and max_y > min_y:
                        sub_pts = pts - np.array([min_x, min_y], dtype=np.int32)
                        roi = frame[min_y:max_y, min_x:max_x]
                        overlay = roi.copy()
                        cv2.fillPoly(overlay, [sub_pts], shaded_bgr, lineType=cv2.LINE_AA)
                        cv2.addWeighted(overlay, material.opacity, roi, 1.0 - material.opacity, 0, roi)
                else:
                    cv2.fillPoly(frame, [pts], shaded_bgr, lineType=cv2.LINE_AA)

        return frame
