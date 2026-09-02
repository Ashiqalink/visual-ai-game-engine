// gridops.hpp - the boolean-grid morphology the sculpt tools run on.
//
// Three operations on an occupancy grid indexed [x, y, z], the same layout
// voxelmesh takes: peel the cells that touch air, split a lump into the core a
// hollow takes out and the skin it must not, and vote a grid down to a coarser
// one for display. They live here rather than in the game that uses them
// because they are grid arithmetic with nothing of a game in them, and because
// the mesher next door already owns this grid layout.
//
// Each has a numpy twin in visual_ai.three_d.voxel that is the fallback when
// the extension is not built and the reference tests/three_d/test_gridops.py
// compares this against. Nothing here touches Python; gridops_bind.cpp does.
#pragma once


namespace gridops {

//: Occupied cells all six of whose neighbours are occupied. A cell on the
//: grid's rim has a neighbour outside it, which counts as empty, so the rim
//: always peels. `out` and `grid` are shape[0]*shape[1]*shape[2] in C order
//: and must not alias.
void erode(const bool* grid, bool* out, const int shape[3]);

//: `core` = the cells more than `depth` erosions in, `skin` = the first
//: `skin_depth` layers of the lump.
//
// Depth is measured from ANY surface, a cavity's included, so hollowing an
// already hollow lump finds no core and leaves the wall alone instead of
// thinning it a layer per stroke. Both are eroded off the same grid, so the
// shared prefix of the two erosion chains is walked once.
void wall_layers(const bool* grid, int depth, int skin_depth,
                 const int shape[3], bool* core, bool* skin);

//: `grid` in blocks of `factor`, a block filled when at least half of it is.
//
// A majority vote, not an `any`: `any` dilates the surface by up to a block
// and fills a one-cell scratch back in, `all` eats the hollow wall. Ties go to
// filled, so a wall `factor` cells thick survives. Trailing cells that do not
// fill a whole block are dropped, which is the numpy reshape's behaviour.
// `out` holds (shape[a] / factor) cells per axis.
void downsample(const bool* grid, bool* out, const int shape[3], int factor);

}  // namespace gridops
