"""
The compiled voxel mesher against the numpy one it replaces.

`voxel.region_mesh` routes to `engine_core.region_mesh` when the extension is
built and to `voxel._region_mesh_py` when it is not, so both have to produce
the same mesh — not merely the same surface. Same vertices in the same order,
same faces in the same order, same winding: the renderer's draw order is a
sort over face position, and the sculpt tools address vertices by index, so a
reordering that looks equivalent here would not be.

Every parity assertion has a mutation test next to it that makes the same
assertion about a deliberately wrong mesh, because a comparison of two things
that are both empty passes without checking anything.
"""

import numpy as np
import pytest

from visual_ai.three_d import voxel
from visual_ai.three_d.voxel import CPP_MESHER_AVAILABLE

pytestmark = pytest.mark.skipif(
    not CPP_MESHER_AVAILABLE,
    reason="engine_core.region_mesh is not built; only the numpy path exists")

import engine_core  # noqa: E402  (skipped above when it is absent)


def _lump(n=24):
    """A grid with an outside, an inside, a hole and a detached piece."""
    c = (n - 1) / 2.0
    grid = voxel.ellipsoid((n, n, n), (c, c, c), (c * 0.85,) * 3)
    i, j, k = np.indices((n, n, n))
    grid &= ~(((i - c) ** 2 + (j - c) ** 2 + (k - c) ** 2) < 5.0 ** 2)  # hollow
    grid &= ~((np.abs(i - c) < 2) & (np.abs(k - c) < 2))                # bore
    grid |= (i < 3) & (j < 3) & (k < 3)                                 # island
    return grid


def _both(grid, lo, hi, cell=2.5):
    padded = voxel.padded_grid(grid)
    return (voxel.region_mesh(grid, padded, lo, hi, cell),
            voxel._region_mesh_py(grid, padded, lo, hi, cell))


def _assert_same(cpp, py):
    assert (cpp is None) == (py is None)
    if cpp is None:
        return
    assert len(py.faces) > 0, "meshed nothing, so this compared nothing"
    np.testing.assert_array_equal(cpp.vertices, py.vertices)
    assert [list(f) for f in cpp.faces] == [list(f) for f in py.faces]


def test_side_table_matches_python():
    """The C++ side table is a transcription; compare it, do not trust it."""
    compiled = [(tuple(d), tuple(tuple(c) for c in corners))
                for d, corners in engine_core.voxel_sides()]
    assert compiled == [(tuple(d), tuple(tuple(c) for c in corners))
                        for d, corners in voxel._SIDES]


def test_whole_grid_parity():
    grid = _lump()
    _assert_same(*_both(grid, (0, 0, 0), grid.shape))


def test_whole_grid_parity_can_fail():
    """Flip one interior cell: the two meshes must then differ."""
    grid = _lump()
    padded = voxel.padded_grid(grid)
    good = voxel.region_mesh(grid, padded, (0, 0, 0), grid.shape, 2.5)

    moved = grid.copy()
    moved[4, 12, 12] = not moved[4, 12, 12]
    bad = voxel._region_mesh_py(moved, voxel.padded_grid(moved),
                                (0, 0, 0), moved.shape, 2.5)
    with pytest.raises(AssertionError):
        _assert_same(good, bad)


@pytest.mark.parametrize("size", [6, 8, 16])
def test_chunk_parity_over_every_chunk(size):
    """Chunked, the mesh of every box must match — seams included."""
    grid = _lump()
    seen = 0
    for lo, hi in voxel.chunk_bounds(grid.shape, size):
        cpp, py = _both(grid, lo, hi)
        assert (cpp is None) == (py is None)
        if cpp is None:
            continue
        seen += 1
        np.testing.assert_array_equal(cpp.vertices, py.vertices)
        assert [list(f) for f in cpp.faces] == [list(f) for f in py.faces]
    assert seen >= 2, f"only {seen} chunk(s) held anything; not a chunk test"


def test_chunking_raises_no_seam():
    """
    Chunks share their boundary corners, and raise no wall between them.

    The face count is the check: a chunk meshed in isolation grows a wall on
    every side it touches a neighbour on, so the chunked total would exceed the
    whole-grid total. It must instead equal it exactly.
    """
    grid = _lump()
    padded = voxel.padded_grid(grid)
    whole = voxel.region_mesh(grid, padded, (0, 0, 0), grid.shape, 2.5)
    chunked = [m for m in (voxel.region_mesh(grid, padded, lo, hi, 2.5)
                           for lo, hi in voxel.chunk_bounds(grid.shape, 8))
               if m is not None]
    assert len(chunked) > 1
    assert sum(len(m.faces) for m in chunked) == len(whole.faces)


def test_chunking_raises_no_seam_can_fail():
    """The same count, meshing each chunk against itself, must be larger."""
    grid = _lump()
    whole = voxel.region_mesh(grid, voxel.padded_grid(grid), (0, 0, 0),
                              grid.shape, 2.5)
    isolated = 0
    for lo, hi in voxel.chunk_bounds(grid.shape, 8):
        box = np.ascontiguousarray(grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        mesh = voxel.region_mesh(box, voxel.padded_grid(box), (0, 0, 0),
                                 box.shape, 2.5)
        isolated += 0 if mesh is None else len(mesh.faces)
    assert isolated > len(whole.faces)


def test_empty_and_full_boxes():
    grid = np.zeros((8, 8, 8), dtype=bool)
    assert voxel.region_mesh(grid, voxel.padded_grid(grid), (0, 0, 0),
                             grid.shape, 1.0) is None

    grid[:] = True
    padded = voxel.padded_grid(grid)
    _assert_same(voxel.region_mesh(grid, padded, (0, 0, 0), grid.shape, 1.0),
                 voxel._region_mesh_py(grid, padded, (0, 0, 0), grid.shape, 1.0))

    # A box that holds nothing inside a grid that does: still None, and the
    # empty interior of a hollow shell is the same case.
    grid[2:6, 2:6, 2:6] = False
    assert voxel.region_mesh(grid, voxel.padded_grid(grid), (3, 3, 3),
                             (5, 5, 5), 1.0) is None


def test_single_cell_is_a_closed_cube():
    grid = np.zeros((3, 3, 3), dtype=bool)
    grid[1, 1, 1] = True
    padded = voxel.padded_grid(grid)
    mesh = voxel.region_mesh(grid, padded, (0, 0, 0), grid.shape, 2.0)
    assert len(mesh.faces) == 6
    assert len(mesh.vertices) == 8, "corners were not shared between the sides"
    _assert_same(mesh, voxel._region_mesh_py(grid, padded, (0, 0, 0),
                                             grid.shape, 2.0))


def test_face_arrays_are_primed_and_correct():
    """The mesher hands over the CSR faces; they must match the list of lists."""
    grid = _lump(12)
    mesh = voxel.region_mesh(grid, voxel.padded_grid(grid), (0, 0, 0),
                             grid.shape, 2.5)
    flat, offsets = mesh.face_arrays()
    assert offsets[-1] == 4 * len(mesh.faces)
    np.testing.assert_array_equal(
        flat, np.array([v for f in mesh.faces for v in f], dtype=np.int32))
    np.testing.assert_array_equal(
        offsets, np.arange(0, 4 * len(mesh.faces) + 1, 4, dtype=np.int32))


def test_non_contiguous_grid_still_meshes():
    """A strided view is a real caller — a downsample often leaves one."""
    big = _lump(24)
    grid = big[::2, ::2, ::2]
    assert not grid.flags["C_CONTIGUOUS"]
    padded = voxel.padded_grid(grid)
    _assert_same(voxel.region_mesh(grid, padded, (0, 0, 0), grid.shape, 2.5),
                 voxel._region_mesh_py(grid, padded, (0, 0, 0), grid.shape, 2.5))


def test_voxel_mesh_matches_the_models():
    """Every table model, through the public entry point."""
    for name, builder in voxel.MODEL_GRIDS.items():
        grid, cell = builder()
        grid = np.asarray(grid, dtype=bool)
        padded = voxel.padded_grid(grid)
        py = voxel._region_mesh_py(grid, padded, (0, 0, 0), grid.shape, cell)
        cpp = voxel.MODELS[name]()
        assert len(cpp.faces) > 100, f"{name} meshed to almost nothing"
        np.testing.assert_array_equal(cpp.vertices, py.vertices)
        assert [list(f) for f in cpp.faces] == [list(f) for f in py.faces], name
