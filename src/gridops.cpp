#include "gridops.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>

namespace gridops {

void erode(const bool* grid, bool* out, const int shape[3]) {
    const int nx = shape[0], ny = shape[1], nz = shape[2];
    const std::ptrdiff_t sx = static_cast<std::ptrdiff_t>(ny) * nz, sy = nz;
    std::memset(out, 0, static_cast<std::size_t>(nx) * ny * nz);
    // The rim is skipped rather than tested: a rim cell has a neighbour off
    // the grid, that counts as empty, so it can never survive. Which leaves
    // the interior with all six neighbours in bounds and no bounds test in the
    // loop - the numpy version gets the same by padding, at the cost of a
    // copy of the whole grid.
    for (int i = 1; i < nx - 1; ++i) {
        for (int j = 1; j < ny - 1; ++j) {
            const std::ptrdiff_t base = i * sx + j * sy;
            const bool* c = grid + base;
            const bool* xm = c - sx;
            const bool* xp = c + sx;
            const bool* ym = c - sy;
            const bool* yp = c + sy;
            bool* o = out + base;
            for (int k = 1; k < nz - 1; ++k) {
                o[k] = c[k] & xm[k] & xp[k] & ym[k] & yp[k]
                       & c[k - 1] & c[k + 1];
            }
        }
    }
}

void wall_layers(const bool* grid, int depth, int skin_depth,
                 const int shape[3], bool* core, bool* skin) {
    const std::size_t n =
        static_cast<std::size_t>(shape[0]) * shape[1] * shape[2];

    // Both chains erode the same grid, so walk one chain and take a copy of it
    // at each of the two depths rather than walking it twice.
    const int deepest = std::max(depth, skin_depth);
    if (deepest == 0) {
        std::memcpy(core, grid, n);
        std::memset(skin, 0, n);
        return;
    }
    std::unique_ptr<bool[]> a(new bool[n]), b(new bool[n]);
    std::memcpy(a.get(), grid, n);
    for (int step = 0; step <= deepest; ++step) {
        if (step == depth) {
            std::memcpy(core, a.get(), n);
        }
        if (step == skin_depth) {
            // `skin` is the lump minus what is left after skin_depth erosions:
            // the outermost layers, which is the part anyone looking at the
            // model sees. Written through unsigned char with `&` rather than
            // as `grid[x] && !a[x]`: `&&` short-circuits, which is a branch per
            // cell that the compiler will not vectorise over half a million of
            // them.
            const unsigned char* g = reinterpret_cast<const unsigned char*>(grid);
            const unsigned char* left = reinterpret_cast<const unsigned char*>(a.get());
            unsigned char* out = reinterpret_cast<unsigned char*>(skin);
            for (std::size_t x = 0; x < n; ++x) out[x] = g[x] & (left[x] ^ 1u);
        }
        if (step == deepest) break;
        erode(a.get(), b.get(), shape);
        a.swap(b);
    }
}

void downsample(const bool* grid, bool* out, const int shape[3], int factor) {
    const int nx = shape[0] / factor, ny = shape[1] / factor,
              nz = shape[2] / factor;
    const std::ptrdiff_t sx = static_cast<std::ptrdiff_t>(shape[1]) * shape[2],
                         sy = shape[2];
    const int cells = factor * factor * factor;  // doubled below, so ties fill
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            bool* row = out + (static_cast<std::ptrdiff_t>(i) * ny + j) * nz;
            for (int k = 0; k < nz; ++k) {
                int count = 0;
                for (int a = 0; a < factor; ++a) {
                    for (int b = 0; b < factor; ++b) {
                        const bool* src = grid + (i * factor + a) * sx
                                               + (j * factor + b) * sy
                                               + k * factor;
                        for (int c = 0; c < factor; ++c) count += src[c];
                    }
                }
                row[k] = count * 2 >= cells;
            }
        }
    }
}

}  // namespace gridops
