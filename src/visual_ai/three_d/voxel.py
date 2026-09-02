"""
voxel - occupancy grids meshed into the quads that are actually exposed.

A voxel model is a boolean grid indexed ``[x, y, z]``, Y up like every other
mesh in the package, meshed as the quads between an occupied cell and an empty
one. Only those quads exist: a solid 3x3x3 block is 54 faces, not 162, so the
renderer never sorts a face nothing could see. Each quad is wound exactly as
`Mesh3D.create_cube` winds the matching side, so the screen-space backface cull
keeps them.

`MODELS` holds one builder per shape the smooth primitives offer, at the same
size and each at its own cell size. They are heavier than the primitives on
purpose — 480 to 1750 faces against a smooth mesh's dozens — which is what
makes them the renderer's soak::

    from visual_ai.three_d import voxel

    mesh = voxel.MODELS["gear"]()          # 1024 faces
    mesh = voxel.torus(ring_radius=40.0, tube_radius=14.0, cell=5.0)
    mesh = voxel.voxel_mesh(my_bool_grid, cell=2.5)

`region_mesh` meshes one box of a grid while judging exposure against the whole
of it, which is what lets a caller re-mesh a chunk after an edit without a seam
appearing between it and its neighbours.
"""

import functools
import math

import numpy as np

from visual_ai.three_d.mesh import Mesh3D

try:
    import engine_core as _engine_core
    if not hasattr(_engine_core, "region_mesh"):
        # A .pyd built before the mesher landed: importable, but without the
        # symbol. Treat it as absent rather than letting the call fail later.
        _engine_core = None
except ImportError:
    _engine_core = None

#: Whether the compiled mesher is the one `region_mesh` will use. The pure
#: Python path below stays authoritative — it is what runs without it, and it
#: is what tests/three_d/test_voxelmesh.py compares the compiled one against.
CPP_MESHER_AVAILABLE = _engine_core is not None

#: Likewise for the grid morphology below, which landed after the mesher and
#: so can be missing from a .pyd that has `region_mesh`.
CPP_GRIDOPS_AVAILABLE = (_engine_core is not None
                         and hasattr(_engine_core, "grid_erode"))

#: Per side: the neighbour offset that must be empty for the face to exist,
#: and the four corner offsets (cell units from the cell's minimum corner) in
#: create_cube's order for that side.
_SIDES = (
    ((0, 0, -1), ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))),   # -Z
    ((0, 0, 1),  ((1, 0, 1), (0, 0, 1), (0, 1, 1), (1, 1, 1))),   # +Z
    ((-1, 0, 0), ((0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1))),   # -X
    ((1, 0, 0),  ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0))),   # +X
    ((0, -1, 0), ((0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0))),   # -Y
    ((0, 1, 0),  ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))),   # +Y
)


#: The six neighbour offsets, taken from the side table so the two cannot drift
#: apart. Used to find the cells that have an empty face — the surface.
NEIGHBOURS = tuple(offset for offset, _ in _SIDES)


def padded_grid(grid: np.ndarray) -> np.ndarray:
    """The grid with one empty cell of margin, so a neighbour is always in it."""
    nx, ny, nz = grid.shape
    padded = np.zeros((nx + 2, ny + 2, nz + 2), dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = grid
    return padded


def erode(grid: np.ndarray) -> np.ndarray:
    """Occupied cells all six of whose neighbours are occupied.

    The grid's rim always peels: a rim cell has a neighbour off the grid, and
    what is off the grid counts as empty. Repeated, this is a depth map in
    disguise — `wall_layers` is what uses it that way.
    """
    if CPP_GRIDOPS_AVAILABLE:
        return _engine_core.grid_erode(grid)
    return _erode_py(grid)


def _erode_py(grid: np.ndarray) -> np.ndarray:
    """`erode` in pure numpy — the fallback, and the reference."""
    pad = np.pad(grid, 1, constant_values=False)
    return (grid
            & pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1]
            & pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1]
            & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2])


def wall_layers(grid: np.ndarray, depth: int,
                skin_depth: int) -> tuple[np.ndarray, np.ndarray]:
    """
    ``(core, skin)``: what hollowing a lump takes out, and the shell it must not.

    One erosion peels the cells that touch air, so ``depth`` of them leave the
    cells more than ``depth`` from any surface — the core, which a hollow lump
    has none of. ``skin`` is the outermost ``skin_depth`` layers: the part of
    the wall that is the model as far as anyone looking at it is concerned.

    Depth from *any* surface, not from the outside specifically. Once a lump is
    hollow the cavity is a surface too, so hollowing it again finds no core and
    leaves the wall alone rather than eating it a layer at a time.
    """
    if CPP_GRIDOPS_AVAILABLE:
        return _engine_core.grid_wall_layers(grid, int(depth), int(skin_depth))
    return _wall_layers_py(grid, depth, skin_depth)


def _wall_layers_py(grid: np.ndarray, depth: int,
                    skin_depth: int) -> tuple[np.ndarray, np.ndarray]:
    """`wall_layers` in pure numpy — the fallback, and the reference."""
    core = grid
    for _ in range(depth):
        core = _erode_py(core)
    skin = grid
    for _ in range(skin_depth):
        skin = _erode_py(skin)
    return core, grid & ~skin


def downsample(grid: np.ndarray, factor: int) -> np.ndarray:
    """
    ``grid`` in blocks of ``factor``, a block filled when at least half of it is.

    A majority vote rather than an `any`: `any` dilates the surface by up to a
    block and would fill a one-cell scratch back in, and an `all` would eat a
    hollow wall. Ties go to filled, so a wall ``factor`` cells thick survives.
    Cells past the last whole block are dropped.
    """
    if CPP_GRIDOPS_AVAILABLE:
        return _engine_core.grid_downsample(grid, int(factor))
    return _downsample_py(grid, factor)


def _downsample_py(grid: np.ndarray, factor: int) -> np.ndarray:
    """`downsample` in pure numpy — the fallback, and the reference."""
    shape = tuple(size // factor for size in grid.shape)
    blocks = grid[:shape[0] * factor, :shape[1] * factor, :shape[2] * factor]
    blocks = blocks.reshape(shape[0], factor, shape[1], factor,
                            shape[2], factor)
    return blocks.sum(axis=(1, 3, 5)) * 2 >= factor ** 3


def region_mesh(grid: np.ndarray, padded: np.ndarray,
                lo: tuple[int, int, int], hi: tuple[int, int, int],
                cell: float) -> Mesh3D | None:
    """
    Mesh the cells ``grid[lo:hi]`` only, or None if that box holds nothing.

    Exposure is judged against ``padded`` — the *whole* grid — not against the
    box. That is what makes chunking work: a cell on a chunk's boundary sees
    the occupied cell in the next chunk and raises no face, so the seam between
    two chunks is invisible. Meshed in isolation, each chunk would instead grow
    a wall on every side it shares with a neighbour, which measured as 3.5x the
    faces.

    Corner ids are linear over the *full* lattice, so two chunks that share a
    corner put it at the same position and the sculpt tools — which move
    vertices — cannot tear them apart.
    """
    if _engine_core is not None:
        built = _engine_core.region_mesh(grid, padded, tuple(lo), tuple(hi),
                                         cell)
        if built is None:
            return None
        vertices, quads = built
        mesh = Mesh3D(vertices=vertices, faces=quads.tolist())
        # The mesher already has the faces flat, and every one is a quad, so
        # hand the rasteriser its CSR form directly rather than making
        # face_arrays() walk the list of lists back into one.
        mesh._face_arrays = (quads.reshape(-1),
                             np.arange(0, 4 * len(quads) + 1, 4,
                                       dtype=np.int32))
        return mesh
    return _region_mesh_py(grid, padded, lo, hi, cell)


def _region_mesh_py(grid: np.ndarray, padded: np.ndarray,
                    lo: tuple[int, int, int], hi: tuple[int, int, int],
                    cell: float) -> Mesh3D | None:
    """`region_mesh` in pure numpy — the fallback, and the reference."""
    core = grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if not core.any():
        return None
    nx, ny, nz = grid.shape
    origin = np.array([-nx * cell / 2.0, -ny * cell / 2.0, -nz * cell / 2.0])

    #: The corner lattice is one cell wider than the grid in every axis.
    span_y, span_z = ny + 1, nz + 1
    quads = []
    for (dx, dy, dz), offsets in _SIDES:
        neighbour = padded[1 + lo[0] + dx:1 + hi[0] + dx,
                           1 + lo[1] + dy:1 + hi[1] + dy,
                           1 + lo[2] + dz:1 + hi[2] + dz]
        i, j, k = np.nonzero(core & ~neighbour)
        if not len(i):
            continue
        gi, gj, gk = i + lo[0], j + lo[1], k + lo[2]
        quad = np.empty((len(i), 4), dtype=np.int64)
        for corner, (ox, oy, oz) in enumerate(offsets):
            quad[:, corner] = ((gi + ox) * span_y + (gj + oy)) * span_z + (gk + oz)
        quads.append(quad)
    if not quads:
        return None

    # `unique` returns the used corner ids in order and, as `inverse`, each
    # face's corners renumbered against them — which is the sharing.
    used, inverse = np.unique(np.concatenate(quads), return_inverse=True)
    lattice = np.stack(np.unravel_index(used, (nx + 1, span_y, span_z)), axis=1)
    return Mesh3D(vertices=origin + lattice * cell,
                  faces=inverse.reshape(-1, 4).tolist())


def voxel_mesh(grid: np.ndarray, cell: float) -> Mesh3D:
    """
    Mesh a boolean grid ``[x, y, z]`` into exposed quads, ``cell`` units each.

    Corners are shared between neighbouring faces, so the vertex array stays
    small and the sculpt tools — which move vertices, not faces — keep the
    surface watertight. The grid is centred on the origin.

    Sharing used to be a dict keyed on the corner's grid position, filled one
    corner at a time. That loop was the whole cost of a remesh, and a remesh is
    the whole cost of a brush stroke. Corners live on a regular lattice, so
    each one has a linear id — ``(i * (ny + 1) + j) * (nz + 1) + k`` — and
    ``np.unique`` shares them all in one call instead. Same faces, in the same
    order, with the same corners in the same positions; 3-5x faster from N=16
    to N=128.

    The whole grid is just the one-chunk case of `region_mesh`.
    """
    grid = np.asarray(grid, dtype=bool)
    if grid.ndim != 3:
        raise ValueError(f"voxel grid must be 3-D, got shape {grid.shape}")
    mesh = region_mesh(grid, padded_grid(grid), (0, 0, 0), grid.shape,
                              cell)
    if mesh is None:
        return Mesh3D(vertices=np.empty((0, 3), dtype=np.float64), faces=[])
    return mesh


def chunk_bounds(shape: tuple[int, int, int], size: int) -> tuple:
    """Every chunk's ``(lo, hi)`` cell bounds, in a fixed order."""
    return tuple(
        ((i, j, k), (min(i + size, shape[0]), min(j + size, shape[1]),
                     min(k + size, shape[2])))
        for i in range(0, shape[0], size)
        for j in range(0, shape[1], size)
        for k in range(0, shape[2], size))


def ellipsoid(shape: tuple[int, int, int], centre, radii) -> np.ndarray:
    """Cells of a ``shape`` grid inside an axis-aligned ellipsoid."""
    idx = np.indices(shape).astype(np.float64)
    return sum(((idx[a] - centre[a]) / radii[a]) ** 2 for a in range(3)) <= 1.0


def star_points(outer: float, inner: float,
                points: int) -> list[tuple[float, float]]:
    out = []
    for i in range(points * 2):
        radius = outer if i % 2 == 0 else inner
        theta = math.pi * i / points - math.pi / 2.0
        out.append((radius * math.cos(theta), radius * math.sin(theta)))
    return out


def gear_points(outer: float, root: float,
                teeth: int) -> list[tuple[float, float]]:
    """A tooth is a flat-topped square, not a spike — four points per tooth."""
    out = []
    for i in range(teeth):
        base = 2.0 * math.pi * i / teeth
        step = 2.0 * math.pi / teeth
        for fraction, radius in ((0.00, root), (0.15, outer),
                                 (0.35, outer), (0.50, root)):
            theta = base + step * fraction
            out.append((radius * math.cos(theta), radius * math.sin(theta)))
    return out


def tree_grid(n=14, cell=9.5) -> tuple[np.ndarray, float]:
    """A trunk under a lumpy canopy — the voxel model with concave bits."""
    c = (n - 1) / 2.0
    shape = (n, n, n)
    idx = np.indices(shape)
    trunk = (((idx[0] - c) ** 2 + (idx[2] - c) ** 2) <= 1.6 ** 2) & (idx[1] <= 5)
    canopy = ellipsoid(shape, (c, 8.5, c), (5.6, 4.2, 5.6))
    canopy |= ellipsoid(shape, (c - 3.0, 7.0, c + 2.5), (3.2, 2.8, 3.2))
    canopy |= ellipsoid(shape, (c + 3.2, 7.5, c - 2.0), (3.0, 2.6, 3.0))
    canopy |= ellipsoid(shape, (c, 11.5, c), (2.4, 2.2, 2.4))
    return trunk | canopy, cell


# ── Voxel models ───────────────────────────────────────────────────────────
# Every entry in `MESH_BUILDERS` is an occupancy grid now rather than a smooth
# mesh, so a run soaks `voxel_mesh` and the painter's sort on fourteen
# silhouettes instead of two. The smooth builders above are left in place: a
# model is flipped back by naming one of them in the table again.
#
# Each model keeps the size it had as a smooth mesh and picks its *own* cell,
# because one shared resolution would test one resolution. The spread is the
# point — `prism` at 9.0 units a cell is a staircase, `gear` at 7.0 has to hold
# nine teeth, `capsule` at 6.5 is the finest curve — and each knob sits in the
# signature of the model that uses it, so changing one moves nothing else.
#
# Each model is written as a `*_grid` returning ``(grid, cell)``, and the mesh
# builder of the same name is `_meshed` around it. The grid is the primary form
# because a caller that wants to *edit* a model — the sculptor's brush does —
# needs the occupancy, and meshing it first throws that away. Nothing about the
# mesh builders changed: same names, same signatures, same meshes.


def _meshed(grid_fn):
    """A mesh builder from a grid builder — same signature, same docstring."""
    @functools.wraps(grid_fn)
    def build(*args, **kwargs) -> Mesh3D:
        return voxel_mesh(*grid_fn(*args, **kwargs))
    build.__name__ = grid_fn.__name__.removesuffix("_grid")
    build.__qualname__ = build.__name__
    return build


def axes(shape: tuple[int, int, int], cell: float) -> tuple:
    """Cell-centre coordinates in model units, per axis, centred on the origin.

    `voxel_mesh` centres the grid it is handed, so a model built against these
    coordinates lands where the smooth mesh it replaces did, at the same size.
    """
    idx = np.indices(shape).astype(np.float64)
    return tuple((idx[a] - (shape[a] - 1) / 2.0) * cell for a in range(3))


def grid_for(size, cell: float, margin: int = 1) -> tuple[int, ...]:
    """Grid dimensions holding a model ``size`` units across, plus margin.

    The margin is a cell of empty on every side, so no model is wedged against
    the grid wall and every axis keeps a symmetric box about the origin.
    """
    return tuple(int(math.ceil(s / cell)) + 2 * margin for s in size)


def polygon_voxels_grid(points, depth: float,
                        cell: float) -> tuple[np.ndarray, float]:
    """A 2-D outline in X-Y, extruded along Z, filled by the even-odd rule.

    Even-odd rather than a fan, so a concave outline — the star's notches, the
    gear's roots — comes out as a notch instead of being filled over. Same
    point lists the smooth prisms are built from, so the two agree on shape.
    """
    span = (2.0 * max(abs(p[0]) for p in points),
            2.0 * max(abs(p[1]) for p in points), depth)
    shape = grid_for(span, cell)
    x, y, z = axes(shape, cell)
    inside = np.zeros(shape, dtype=bool)
    for i, (ax, ay) in enumerate(points):
        bx, by = points[(i + 1) % len(points)]
        if ay == by:
            continue
        crosses = (ay > y) != (by > y)
        inside ^= crosses & (x < ax + (y - ay) * (bx - ax) / (by - ay))
    return inside & (np.abs(z) <= depth / 2.0), cell


def cube_grid(size=90.0, cell=7.5) -> tuple[np.ndarray, float]:
    """A box: twelve cells an edge, and the one model with no stairs to hide."""
    shape = grid_for((size,) * 3, cell)
    x, y, z = axes(shape, cell)
    h = size / 2.0
    return (np.abs(x) <= h) & (np.abs(y) <= h) & (np.abs(z) <= h), cell


def pyramid_grid(width=95.0, height=115.0,
                 cell=6.5) -> tuple[np.ndarray, float]:
    """A square pyramid — a taper on two axes at once, stepped."""
    shape = grid_for((width, height, width), cell)
    x, y, z = axes(shape, cell)
    h = height / 2.0
    taper = np.clip((h - y) / height, 0.0, 1.0) * (width / 2.0)
    return ((np.abs(x) <= taper) & (np.abs(z) <= taper)
            & (np.abs(y) <= h)), cell


def sphere_grid(radius=62.0, cell=7.75) -> tuple[np.ndarray, float]:
    """A sphere, coarser than `voxel-sphere` and bigger — the two differ."""
    shape = grid_for((2.0 * radius,) * 3, cell)
    x, y, z = axes(shape, cell)
    return x * x + y * y + z * z <= radius * radius, cell


def cylinder_grid(radius=52.0, height=115.0,
                  cell=8.0) -> tuple[np.ndarray, float]:
    """A round side and two flat caps: the mix a stair-step cull gets wrong."""
    shape = grid_for((2.0 * radius, height, 2.0 * radius), cell)
    x, y, z = axes(shape, cell)
    return ((x * x + z * z <= radius * radius)
            & (np.abs(y) <= height / 2.0)), cell


def capsule_grid(radius=44.0, height=80.0,
                 cell=6.5) -> tuple[np.ndarray, float]:
    """`height` is the straight part; a hemisphere caps each end."""
    shape = grid_for((2.0 * radius, height + 2.0 * radius, 2.0 * radius), cell)
    x, y, z = axes(shape, cell)
    dy = np.clip(np.abs(y) - height / 2.0, 0.0, None)
    return x * x + dy * dy + z * z <= radius * radius, cell


def torus_grid(ring_radius=62.0, tube_radius=24.0,
               cell=8.0) -> tuple[np.ndarray, float]:
    """The hole is the test: faces on the far side of it must sort behind."""
    span = 2.0 * (ring_radius + tube_radius)
    shape = grid_for((span, 2.0 * tube_radius, span), cell)
    x, y, z = axes(shape, cell)
    q = np.hypot(x, z) - ring_radius
    return q * q + y * y <= tube_radius * tube_radius, cell


def cone_grid(radius=58.0, height=118.0, cell=8.0) -> tuple[np.ndarray, float]:
    """A taper to a single cell, where the smooth cone tapered to a vertex."""
    shape = grid_for((2.0 * radius, height, 2.0 * radius), cell)
    x, y, z = axes(shape, cell)
    h = height / 2.0
    taper = np.clip((h - y) / height, 0.0, 1.0) * radius
    return (x * x + z * z <= taper * taper) & (np.abs(y) <= h), cell


def icosahedron_grid(radius=66.0, cell=7.5) -> tuple[np.ndarray, float]:
    """The intersection of its twenty face half-spaces.

    The normals are the dodecahedron's vertices — the icosahedron's dual — so
    `radius` is the distance to a face, not to a corner, and the solid is
    about 1.26 times that across its corners.
    """
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    normals = [(sx, sy, sz) for sx in (-1.0, 1.0)
               for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
    for s in (-1.0, 1.0):
        for t in (-1.0, 1.0):
            normals += [(0.0, s / phi, t * phi), (s / phi, t * phi, 0.0),
                        (s * phi, 0.0, t / phi)]
    shape = grid_for((2.6 * radius,) * 3, cell)
    x, y, z = axes(shape, cell)
    inside = np.ones(shape, dtype=bool)
    for ax, ay, az in normals:
        length = math.sqrt(ax * ax + ay * ay + az * az)
        inside &= (x * ax + y * ay + z * az) / length <= radius
    return inside, cell


def tube_grid(radius=52.0, height=110.0, thickness=18.0,
              cell=9.0) -> tuple[np.ndarray, float]:
    """An open pipe, walled two cells thick at this resolution.

    Thicker than the smooth tube's 9 units on purpose: a one-cell wall puts
    its outer and inner faces on the same cell and reads as a sheet rather
    than as something with an inside.
    """
    shape = grid_for((2.0 * radius, height, 2.0 * radius), cell)
    x, y, z = axes(shape, cell)
    r2 = x * x + z * z
    inner = max(cell, radius - thickness)
    return ((r2 <= radius * radius) & (r2 >= inner * inner)
            & (np.abs(y) <= height / 2.0)), cell


def cup_grid(radius=46.0, height=105.0, thickness=15.0,
             cell=7.5) -> tuple[np.ndarray, float]:
    """A pot with a floor and a handle — the model with an inside to see into.

    The grid is sized for the handle, which hangs off +X, so the pot itself
    sits left of the middle exactly as it did on the smooth mesh.
    """
    ring_c, ring_r, side = radius * 0.92, height * 0.30, 9.0
    shape = grid_for((2.0 * (ring_c + ring_r + side), height, 2.0 * radius),
                      cell)
    x, y, z = axes(shape, cell)
    h, r2 = height / 2.0, x * x + z * z
    inner = max(cell, radius - thickness)
    wall = (r2 <= radius * radius) & (r2 >= inner * inner) & (np.abs(y) <= h)
    floor = (r2 <= radius * radius) & (y >= -h) & (y <= -h + thickness)
    q = np.hypot(x - ring_c, y) - ring_r
    handle = (q * q + z * z <= side * side) & (x >= ring_c)
    return wall | floor | handle, cell


def prism_grid(width=84.0, height=84.0, depth=84.0,
               cell=9.0) -> tuple[np.ndarray, float]:
    """A triangular prism, and the coarsest cell in the table."""
    w, h = width / 2.0, height / 2.0
    return polygon_voxels_grid(((-w, -h), (w, -h), (0.0, h)), depth, cell)


def star_grid(outer=72.0, inner=30.0, points=6, depth=44.0,
              cell=7.5) -> tuple[np.ndarray, float]:
    """Six notches, each one a place the even-odd fill has to leave empty."""
    return polygon_voxels_grid(star_points(outer, inner, points), depth, cell)


def gear_grid(outer=70.0, root=52.0, teeth=9, depth=38.0,
              cell=7.0) -> tuple[np.ndarray, float]:
    """Nine flat-topped teeth: the finest outline the table asks for."""
    return polygon_voxels_grid(gear_points(outer, root, teeth), depth, cell)


polygon_voxels = _meshed(polygon_voxels_grid)
cube = _meshed(cube_grid)
pyramid = _meshed(pyramid_grid)
sphere = _meshed(sphere_grid)
cylinder = _meshed(cylinder_grid)
capsule = _meshed(capsule_grid)
torus = _meshed(torus_grid)
cone = _meshed(cone_grid)
icosahedron = _meshed(icosahedron_grid)
tube = _meshed(tube_grid)
cup = _meshed(cup_grid)
prism = _meshed(prism_grid)
star = _meshed(star_grid)
gear = _meshed(gear_grid)
tree = _meshed(tree_grid)


#: Every model by name. The engine's own tables prefix these with ``voxel-``
#: and keep the smooth primitive under the bare name, so the two can be
#: measured against each other.
MODELS = {
    "cube": cube,
    "pyramid": pyramid,
    "sphere": sphere,
    "cylinder": cylinder,
    "capsule": capsule,
    "torus": torus,
    "prism": prism,
    "icosahedron": icosahedron,
    "cup": cup,
    "cone": cone,
    "tube": tube,
    "star": star,
    "gear": gear,
    "tree": tree,
}

#: The same models as ``(grid, cell)`` builders: ``MODELS[name]()`` is
#: ``voxel_mesh(*MODEL_GRIDS[name]())``. The grid is what a caller needs to
#: *edit* a model rather than only draw it.
MODEL_GRIDS = {
    "cube": cube_grid,
    "pyramid": pyramid_grid,
    "sphere": sphere_grid,
    "cylinder": cylinder_grid,
    "capsule": capsule_grid,
    "torus": torus_grid,
    "prism": prism_grid,
    "icosahedron": icosahedron_grid,
    "cup": cup_grid,
    "cone": cone_grid,
    "tube": tube_grid,
    "star": star_grid,
    "gear": gear_grid,
    "tree": tree_grid,
}
