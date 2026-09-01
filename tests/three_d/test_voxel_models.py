"""The voxel mesher and the models built on it.

The point of a voxel model is that it is *only* the exposed quads: a solid
block's interior faces are never built, so the painter's sort never touches a
face nothing could see. Most of what follows checks that property from a
different angle, because a mesher that emitted every face of every cell would
still look right on screen and cost three times as much.
"""

import math

import numpy as np

from visual_ai.three_d import Mesh3D, voxel

#: Each model's own cell size, repeated here rather than read back off the
#: signature - the spread is deliberate (a run sweeps the mesher across
#: resolutions instead of measuring one), so a default quietly changing to
#: match its neighbour should fail rather than pass.
MODEL_CELLS = {
    "cube": 7.5, "pyramid": 6.5, "sphere": 7.75, "cylinder": 8.0,
    "capsule": 6.5, "torus": 8.0, "prism": 9.0, "icosahedron": 7.5,
    "cup": 7.5, "cone": 8.0, "tube": 9.0, "star": 7.5, "gear": 7.0,
    "tree": 9.5,
}

#: Above this a model stops being a soak and starts being a stall: at the
#: renderer's ~4.3 ms per 1000 faces, 2000 is ~8.6 ms, most of a 60 Hz frame.
FACE_BUDGET = 2000


def unit_square_faces(mesh, cell):
    """Is every face an axis-aligned square of exactly one cell's side?"""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    for face in mesh.faces:
        if len(face) != 4:
            return False
        quad = verts[list(face)]
        spread = quad.max(axis=0) - quad.min(axis=0)
        flat = np.isclose(spread, 0.0)
        if flat.sum() != 1:
            return False                      # not axis-aligned, or degenerate
        if not np.allclose(spread[~flat], cell):
            return False                      # right shape, wrong size
    return True


def test_model_table_matches_the_module():
    assert set(voxel.MODELS) == set(MODEL_CELLS)


def test_models_do_not_share_one_resolution():
    # Six distinct cells across fourteen models. One shared cell would mean a
    # cycle measured a single resolution fourteen times.
    assert len(set(MODEL_CELLS.values())) >= 6


def test_a_solid_block_is_its_shell_only():
    mesh = voxel.voxel_mesh(np.ones((3, 3, 3), dtype=bool), cell=2.0)
    # 6 sides x 9 cells. Every face of every cell would be 162.
    assert len(mesh.faces) == 54
    # 4x4x4 lattice corners, each shared by the quads that meet at it, minus
    # the 8 that only the (absent) interior would use... which is to say: far
    # fewer than four corners per face.
    assert len(mesh.vertices) < 2 * len(mesh.faces)


def test_a_hollow_shell_meshes_both_surfaces():
    grid = np.ones((3, 3, 3), dtype=bool)
    grid[1, 1, 1] = False
    mesh = voxel.voxel_mesh(grid, cell=2.0)
    # The outer 54 plus the six faces of the cavity now exposed to it.
    assert len(mesh.faces) == 60


def test_quads_wind_the_way_create_cube_winds():
    # A single cell must be the cube primitive, corner for corner and face for
    # face - the renderer's screen-space backface cull keeps or drops a quad on
    # its winding alone, so a mirrored side would simply vanish.
    voxelled = voxel.voxel_mesh(np.ones((1, 1, 1), dtype=bool), cell=40.0)
    smooth = Mesh3D.create_cube(size=40.0)
    assert len(voxelled.faces) == 6

    def rings(mesh):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        out = set()
        for face in mesh.faces:
            loop = [tuple(np.round(verts[i], 6)) for i in face]
            start = loop.index(min(loop))     # same loop, one starting corner
            out.add(tuple(loop[start:] + loop[:start]))
        return out

    assert rings(voxelled) == rings(smooth)


def test_region_mesh_leaves_no_seam():
    # Re-meshing one chunk has to judge exposure against the whole grid, not
    # against the chunk - otherwise every chunk boundary grows a wall of faces
    # that the full mesh does not have.
    rng = np.random.default_rng(7)
    grid = rng.random((8, 8, 8)) < 0.5
    padded = voxel.padded_grid(grid)
    whole = voxel.voxel_mesh(grid, cell=2.0)

    pieces = 0
    for lo, hi in voxel.chunk_bounds(grid.shape, 4):
        part = voxel.region_mesh(grid, padded, lo, hi, 2.0)
        if part is not None:
            pieces += len(part.faces)
    assert pieces == len(whole.faces)


def test_polygon_voxels_keeps_a_concave_notch_empty():
    # Even-odd crossing, not a triangle fan: the star's notches are outside the
    # outline and must stay empty. A fan would fill them and the model would
    # come out a disc.
    points = voxel.star_points(outer=60.0, inner=20.0, points=5)
    mesh = voxel.polygon_voxels(points, depth=20.0, cell=4.0)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    radius = np.hypot(verts[:, 0], verts[:, 1])
    # Something has to sit well inside the outer radius, in the notches.
    assert radius.min() < 30.0
    assert radius.max() <= 60.0 + 4.0


def test_every_model_builds_a_clean_shell():
    for name, cell in MODEL_CELLS.items():
        mesh = voxel.MODELS[name]()
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        assert len(mesh.faces) > 0, name
        assert np.isfinite(verts).all(), name
        assert unit_square_faces(mesh, cell), name
        assert all(len(set(f)) == 4 for f in mesh.faces), name
        assert len(verts) < 2 * len(mesh.faces), name   # corners are shared


def test_every_model_stays_inside_the_face_budget():
    over = {name: len(voxel.MODELS[name]().faces) for name in MODEL_CELLS
            if len(voxel.MODELS[name]().faces) > FACE_BUDGET}
    assert not over, over


def test_models_are_the_size_of_the_primitives_they_shadow():
    # A voxel model replaces a smooth one in a scene; if it came out at half
    # the size the swap would be visible, not just heavier.
    pairs = [
        (voxel.cube(), Mesh3D.create_cube(size=90.0)),
        (voxel.sphere(), Mesh3D.create_sphere(radius=62.0, rings=8, sectors=12)),
        (voxel.cylinder(),
         Mesh3D.create_cylinder(radius=52.0, height=115.0, segments=12)),
    ]
    for rough, smooth in pairs:
        a = np.asarray(rough.vertices, dtype=np.float64)
        b = np.asarray(smooth.vertices, dtype=np.float64)
        extent_a = a.max(axis=0) - a.min(axis=0)
        extent_b = b.max(axis=0) - b.min(axis=0)
        # Within one cell either way: a grid cannot land on the exact surface.
        assert np.all(np.abs(extent_a - extent_b) <= 10.0), (extent_a, extent_b)


def test_a_model_takes_its_own_knobs():
    # Each size and cell sits in the signature of the model that uses it, so a
    # caller can ask for a coarser or finer one without touching the module.
    coarse = voxel.torus(ring_radius=62.0, tube_radius=24.0, cell=12.0)
    fine = voxel.torus(ring_radius=62.0, tube_radius=24.0, cell=6.0)
    assert len(coarse.faces) < len(fine.faces)
    assert unit_square_faces(coarse, 12.0)
    assert unit_square_faces(fine, 6.0)


def test_ellipsoid_fills_the_radii_it_is_given():
    grid = voxel.ellipsoid((21, 21, 21), (10, 10, 10), (10.0, 4.0, 10.0))
    xs, ys, zs = np.where(grid)
    assert np.ptp(xs) > np.ptp(ys)          # wide in x and z, flat in y
    assert np.ptp(zs) > np.ptp(ys)
    assert grid[10, 10, 10]             # and solid through the middle


def test_gear_points_alternate_tooth_and_root():
    points = voxel.gear_points(outer=70.0, root=52.0, teeth=9)
    radii = sorted({round(math.hypot(x, y), 6) for x, y in points})
    assert radii == [52.0, 70.0]
    assert len(points) == 4 * 9
