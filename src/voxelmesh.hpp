// voxelmesh.hpp - the voxel surface mesher's C++ core.
//
// This is the compiled half of visual_ai.three_d.voxel.region_mesh: given an
// occupancy grid and a box of it, raise a quad for every face of an occupied
// cell whose neighbour is empty, share the corners between the quads that meet
// at them, and hand back a vertex array and a face array.
//
// Exposure is judged against the padded copy of the WHOLE grid, never against
// the box, which is what lets a caller re-mesh one chunk after an edit without
// a wall appearing along the seam it shares with its neighbours. The Python
// path stays and stays authoritative: it runs when the extension is not built,
// and tests/three_d/test_voxelmesh.py compares the two array for array.
//
// Nothing in this header touches Python. The pybind11 layer is
// voxelmesh_bind.cpp, so this can be compiled and tested as plain C++.
#pragma once

#include <cstdint>
#include <vector>

namespace voxelmesh {

//: One side of a cell: the neighbour offset that must be empty for the quad to
//: exist, and its four corner offsets in cell units from the cell's minimum
//: corner. The winding matches Mesh3D.create_cube's for the matching side, so
//: the renderer's screen-space backface cull keeps these quads. Kept in the
//: same order as voxel._SIDES because faces come out grouped by side and the
//: two paths must agree on that order.
struct Side {
    int d[3];
    int corner[4][3];
};

extern const Side SIDES[6];

//: A meshed region. `faces` is 4 * `face_count` corner indices into
//: `vertices`, quads only - a voxel face is always a quad.
struct Mesh {
    std::vector<double> vertices;      // 3 per vertex
    std::vector<std::int32_t> faces;   // 4 per face
    int face_count = 0;
};

//: Mesh grid[lo:hi] against `padded`, `cell` units to a cell.
//
// `grid` is shape[0] * shape[1] * shape[2] booleans in C order; `padded` is
// the same grid with one empty cell of margin on every side, so a neighbour
// lookup is always in bounds. The result is centred on the origin the way the
// whole grid would be, not the way the box would be, so two chunks of one grid
// line up. An empty box gives a Mesh with no faces.
Mesh region_mesh(const bool* grid, const bool* padded, const int shape[3],
                 const int lo[3], const int hi[3], double cell);

}  // namespace voxelmesh
