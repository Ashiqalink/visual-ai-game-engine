#include "voxelmesh.hpp"

#include <algorithm>
#include <cstddef>

namespace voxelmesh {

// Transcribed from voxel._SIDES. The offsets there are written as tuples in
// -Z, +Z, -X, +X, -Y, +Y order and each side's four corners in create_cube's
// winding for that side; changing either here changes the mesh, so the two
// tables are compared cell by cell in tests/three_d/test_voxelmesh.py rather
// than trusted to stay in step by eye.
const Side SIDES[6] = {
    {{0, 0, -1}, {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}},   // -Z
    {{0, 0, 1},  {{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {1, 1, 1}}},   // +Z
    {{-1, 0, 0}, {{0, 0, 1}, {0, 0, 0}, {0, 1, 0}, {0, 1, 1}}},   // -X
    {{1, 0, 0},  {{1, 0, 0}, {1, 0, 1}, {1, 1, 1}, {1, 1, 0}}},   // +X
    {{0, -1, 0}, {{0, 0, 1}, {1, 0, 1}, {1, 0, 0}, {0, 0, 0}}},   // -Y
    {{0, 1, 0},  {{0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}}},   // +Y
};

Mesh region_mesh(const bool* grid, const bool* padded, const int shape[3],
                 const int lo[3], const int hi[3], double cell) {
    Mesh out;
    const int nx = shape[0], ny = shape[1], nz = shape[2];
    if (lo[0] >= hi[0] || lo[1] >= hi[1] || lo[2] >= hi[2]) return out;

    const std::ptrdiff_t sx = static_cast<std::ptrdiff_t>(ny) * nz, sy = nz;
    const std::ptrdiff_t psx = static_cast<std::ptrdiff_t>(ny + 2) * (nz + 2),
                         psy = nz + 2;

    // A corner is named by its position in the box's OWN corner lattice, one
    // wider than the box on each axis. The Python path names corners by a
    // linear id over the whole grid's lattice and hands them to np.unique,
    // which sorts; row-major order over the box lattice is that same ascending
    // order, because both ids are lexicographic in (i, j, k) and the box's
    // corners are a sub-box of the grid's. So sweeping this lattice in order
    // reproduces np.unique's vertex order without sorting anything - which is
    // the whole reason a C++ mesher is faster than the numpy one at all.
    const int cx = hi[0] - lo[0] + 1, cy = hi[1] - lo[1] + 1,
              cz = hi[2] - lo[2] + 1;
    const std::ptrdiff_t lsx = static_cast<std::ptrdiff_t>(cy) * cz, lsy = cz;
    std::vector<std::int32_t> slot(static_cast<std::size_t>(cx) * cy * cz, -1);

    // Faces first, holding lattice positions; they become vertex indices once
    // the sweep below has numbered the corners that are actually used. Order
    // is side-major then row-major within a side, which is the order
    // np.nonzero walks the box in.
    std::vector<std::int32_t> corners;
    for (const Side& side : SIDES) {
        for (int i = lo[0]; i < hi[0]; ++i) {
            for (int j = lo[1]; j < hi[1]; ++j) {
                const bool* row = grid + i * sx + j * sy;
                const bool* nrow = padded + (1 + i + side.d[0]) * psx
                                          + (1 + j + side.d[1]) * psy
                                          + (1 + side.d[2]);
                const std::ptrdiff_t base =
                    (i - lo[0]) * lsx + (j - lo[1]) * lsy - lo[2];
                for (int k = lo[2]; k < hi[2]; ++k) {
                    if (!row[k] || nrow[k]) continue;
                    for (const auto& c : side.corner) {
                        const std::ptrdiff_t at =
                            base + c[0] * lsx + c[1] * lsy + k + c[2];
                        slot[static_cast<std::size_t>(at)] = 0;
                        corners.push_back(static_cast<std::int32_t>(at));
                    }
                }
            }
        }
    }
    if (corners.empty()) return out;

    // Number the used corners in lattice order and place them. Corner ids are
    // linear over the FULL lattice on the Python side for a reason - two
    // chunks that share a corner must put it at the same position - and that
    // still holds here: the position comes from the global (i, j, k), only the
    // numbering is local.
    const double ox = -nx * cell / 2.0, oy = -ny * cell / 2.0,
                 oz = -nz * cell / 2.0;
    std::int32_t next = 0;
    for (int a = 0; a < cx; ++a) {
        for (int b = 0; b < cy; ++b) {
            std::int32_t* row = slot.data() + a * lsx + b * lsy;
            for (int c = 0; c < cz; ++c) {
                if (row[c] < 0) continue;
                row[c] = next++;
                out.vertices.push_back(ox + (lo[0] + a) * cell);
                out.vertices.push_back(oy + (lo[1] + b) * cell);
                out.vertices.push_back(oz + (lo[2] + c) * cell);
            }
        }
    }

    out.face_count = static_cast<int>(corners.size() / 4);
    out.faces.resize(corners.size());
    for (std::size_t n = 0; n < corners.size(); ++n) {
        out.faces[n] = slot[static_cast<std::size_t>(corners[n])];
    }
    return out;
}

}  // namespace voxelmesh
