"""Mesh3D - vertices, faces, and the primitive constructors."""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh3D:
    """
    3D Geometry representation holding vertices and faces. Face normals are
    computed by the renderer per draw (see _face_intensity), not stored here.
    """
    vertices: np.ndarray  # N x 3 float64
    faces: list[list[int]]  # List of face vertex index lists

    # Built on first render, never by hand — see face_groups().
    _face_groups: list[tuple[np.ndarray, np.ndarray]] | None = field(
        default=None, init=False, repr=False, compare=False)
    # Likewise, for the compiled rasteriser — see face_arrays().
    _face_arrays: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False)

    def face_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Faces flattened to ``(indices, offsets)``, face ``i`` being
        ``indices[offsets[i]:offsets[i + 1]]``.

        This is what the compiled rasteriser takes. `face_groups` buckets by
        vertex count because numpy needs rectangular arrays; a C++ loop does
        not, and walking the faces in their original order means a face's
        number is its position, so no separate array of those is needed. Cached
        for the same reason the buckets are: faces never change after a mesh is
        built.
        """
        if self._face_arrays is None:
            offsets = np.zeros(len(self.faces) + 1, dtype=np.int32)
            if len(self.faces):
                offsets[1:] = np.cumsum([len(f) for f in self.faces],
                                        dtype=np.int32)
            flat = np.fromiter((v for face in self.faces for v in face),
                               dtype=np.int32, count=int(offsets[-1]))
            self._face_arrays = (flat, offsets)
        return self._face_arrays

    def face_groups(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Faces bucketed by vertex count: ``(vertex indices (F, K), face numbers (F,))``.

        The renderer needs an index array per face; building one per face per
        frame was the bulk of its Python work. Faces never change after a mesh
        is built, so the buckets are worked out once and reused. ``face
        numbers`` are positions in ``self.faces``, kept so the draw order can
        still tie-break the way a stable sort over the original list would.
        """
        if self._face_groups is None:
            buckets: dict[int, list[int]] = {}
            for i, face in enumerate(self.faces):
                buckets.setdefault(len(face), []).append(i)
            self._face_groups = [
                (np.array([self.faces[i] for i in positions], dtype=np.intp),
                 np.array(positions, dtype=np.intp))
                for positions in buckets.values()
            ]
        return self._face_groups

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
    def create_cylinder(cls, radius: float = 25.0, height: float = 60.0,
                        segments: int = 12) -> "Mesh3D":
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
    def create_capsule(cls, radius: float = 20.0, height: float = 40.0,
                       rings: int = 6, sectors: int = 12) -> "Mesh3D":
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
    def create_torus(cls, ring_radius: float = 30.0, tube_radius: float = 10.0,
                     ring_segments: int = 16, tube_segments: int = 8) -> "Mesh3D":
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
    def create_prism(cls, width: float = 40.0, height: float = 40.0,
                     depth: float = 40.0) -> "Mesh3D":
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
        return Mesh3D(vertices=new_verts, faces=self.faces)

    def scale_non_uniform(self, sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> "Mesh3D":
        """Apply non-uniform 3D scaling to mesh vertices."""
        new_verts = self.vertices.copy()
        new_verts *= np.array([sx, sy, sz], dtype=np.float64)
        return Mesh3D(vertices=new_verts, faces=self.faces)
