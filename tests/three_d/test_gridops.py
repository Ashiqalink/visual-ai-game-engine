"""
The compiled grid morphology against the numpy it replaces.

`voxel.erode`, `voxel.wall_layers` and `voxel.downsample` route to
`engine_core` when the extension is built and to their `_*_py` twins when it is
not, so the two must agree cell for cell — these decide what a hollow takes out
of a model and what the player sees, and a one-cell disagreement is a hole in
the wall or a scratch filled back in.

Each parity test is paired with one that shows the comparison can fail, and
each op also has a property that pins what it means rather than only that two
implementations match — two identical mistakes would pass the parity tests on
their own.
"""

import numpy as np
import pytest

from visual_ai.three_d import voxel
from visual_ai.three_d.voxel import CPP_GRIDOPS_AVAILABLE

pytestmark = pytest.mark.skipif(
    not CPP_GRIDOPS_AVAILABLE,
    reason="engine_core.grid_erode is not built; only the numpy path exists")


def _lump(n=24, hollow=True):
    """A grid with an outside, a cavity, a bore and a detached island."""
    c = (n - 1) / 2.0
    grid = voxel.ellipsoid((n, n, n), (c, c, c), (c * 0.85,) * 3)
    i, j, k = np.indices((n, n, n))
    if hollow:
        grid &= ~(((i - c) ** 2 + (j - c) ** 2 + (k - c) ** 2) < 4.0 ** 2)
    grid &= ~((np.abs(i - c) < 2) & (np.abs(k - c) < 2))
    grid |= (i < 3) & (j < 3) & (k < 3)
    return grid


def test_erode_parity():
    grid = _lump()
    got = voxel.erode(grid)
    assert got.any() and not got.all(), "eroded to nothing or to everything"
    np.testing.assert_array_equal(got, voxel._erode_py(grid))


def test_erode_parity_can_fail():
    """The same assertion against a grid one cell different must fail."""
    grid = _lump()
    other = grid.copy()
    other[8, 8, 8] = not other[8, 8, 8]
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(voxel.erode(grid),
                                      voxel._erode_py(other))


def test_erode_peels_the_rim_and_keeps_the_middle():
    """What erode means, not merely that two versions agree on it."""
    grid = np.ones((5, 5, 5), dtype=bool)
    got = voxel.erode(grid)
    assert got[1:-1, 1:-1, 1:-1].all(), "the interior should survive"
    assert not got[0].any() and not got[-1].any(), "the rim should peel"
    assert got.sum() == 27

    # A cell missing one neighbour goes, and takes only itself.
    grid = np.ones((5, 5, 5), dtype=bool)
    grid[0, 2, 2] = False
    assert not voxel.erode(grid)[1, 2, 2]
    assert voxel.erode(grid).sum() == 26


@pytest.mark.parametrize("depth,skin_depth", [(0, 0), (1, 1), (3, 2), (2, 5)])
def test_wall_layers_parity(depth, skin_depth):
    grid = _lump(28, hollow=False)
    core, skin = voxel.wall_layers(grid, depth, skin_depth)
    py_core, py_skin = voxel._wall_layers_py(grid, depth, skin_depth)
    np.testing.assert_array_equal(core, py_core)
    np.testing.assert_array_equal(skin, py_skin)
    if skin_depth:
        assert skin.any(), "no skin at all; this compared two empty grids"


def test_wall_layers_parity_can_fail():
    grid = _lump(28, hollow=False)
    core, _ = voxel.wall_layers(grid, 3, 2)
    off_by_one, _ = voxel._wall_layers_py(grid, 4, 2)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(core, off_by_one)


def test_wall_layers_depth_zero_is_the_lump_itself():
    grid = _lump(20, hollow=False)
    core, skin = voxel.wall_layers(grid, 0, 0)
    np.testing.assert_array_equal(core, grid)
    assert not skin.any(), "zero layers of skin is no skin"


def test_wall_layers_core_and_skin_do_not_overlap_and_nest():
    """The shell is outside the core, and deeper eats more."""
    grid = _lump(28, hollow=False)
    core, skin = voxel.wall_layers(grid, 3, 3)
    assert not (core & skin).any()
    assert (core | skin == grid).all(), "core plus skin should be the lump"
    deeper, _ = voxel.wall_layers(grid, 5, 3)
    assert deeper.sum() < core.sum()
    assert (deeper & ~core).sum() == 0, "a deeper core must nest inside"


def test_hollowing_a_hollow_lump_finds_no_core():
    """The reason depth is measured from any surface, cavity included."""
    solid = _lump(28, hollow=False)
    core, _ = voxel.wall_layers(solid, 2, 2)
    assert core.any()
    hollowed = solid & ~core
    again, _ = voxel.wall_layers(hollowed, 2, 2)
    assert not again.any(), "a second hollow would thin the wall"


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_downsample_parity(factor):
    grid = _lump()
    got = voxel.downsample(grid, factor)
    assert got.shape == tuple(s // factor for s in grid.shape)
    assert got.any() and not got.all()
    np.testing.assert_array_equal(got, voxel._downsample_py(grid, factor))


def test_downsample_parity_can_fail():
    grid = _lump()
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(voxel.downsample(grid, 2),
                                      voxel._downsample_py(grid, 4))


def test_downsample_votes_by_majority_with_ties_filled():
    """Half a block filled must fill; one under must not."""
    grid = np.zeros((2, 2, 2), dtype=bool)
    grid[0, :, :] = True                       # 4 of 8 — a tie
    assert voxel.downsample(grid, 2)[0, 0, 0]
    grid[0, 1, 1] = False                      # 3 of 8
    assert not voxel.downsample(grid, 2)[0, 0, 0]


def test_downsample_drops_the_ragged_edge():
    """A grid that does not divide keeps only its whole blocks."""
    grid = np.ones((7, 4, 5), dtype=bool)
    got = voxel.downsample(grid, 2)
    assert got.shape == (3, 2, 2)
    np.testing.assert_array_equal(got, voxel._downsample_py(grid, 2))


def test_non_contiguous_input():
    """A strided view is a real caller — every one of these takes one."""
    big = _lump(32)
    grid = big[::2, ::2, ::2]
    assert not grid.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(voxel.erode(grid), voxel._erode_py(grid))
    np.testing.assert_array_equal(voxel.downsample(grid, 2),
                                  voxel._downsample_py(grid, 2))
    core, skin = voxel.wall_layers(grid, 2, 1)
    py_core, py_skin = voxel._wall_layers_py(grid, 2, 1)
    np.testing.assert_array_equal(core, py_core)
    np.testing.assert_array_equal(skin, py_skin)


def test_thin_grids_do_not_walk_off_the_end():
    """Every axis smaller than the neighbourhood the ops read."""
    for shape in [(1, 1, 1), (1, 5, 5), (2, 2, 2), (3, 1, 3)]:
        grid = np.ones(shape, dtype=bool)
        np.testing.assert_array_equal(voxel.erode(grid), voxel._erode_py(grid))
        core, skin = voxel.wall_layers(grid, 1, 1)
        py_core, py_skin = voxel._wall_layers_py(grid, 1, 1)
        np.testing.assert_array_equal(core, py_core)
        np.testing.assert_array_equal(skin, py_skin)
