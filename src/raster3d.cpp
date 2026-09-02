#include "raster3d.hpp"

#include <algorithm>
#include <cmath>

namespace raster3d {
namespace {

//: Screen coordinates are cast to int the way numpy's .astype(np.int32) does -
//: truncation toward zero, not floor. A vertex just in front of the near plane
//: projects arbitrarily far out, though, and casting a double outside int32 is
//: undefined in C++ where numpy merely wraps. Clamping is the one place this
//: deliberately does not reproduce the Python path: a face that far off screen
//: is clipped away by the scanline bounds either way, so the pixels are the
//: same and the undefined behaviour is not.
constexpr double kCoordLimit = 1.0e9;

inline std::int32_t to_screen_int(double v) {
    if (!(v > -kCoordLimit)) return static_cast<std::int32_t>(-kCoordLimit);
    if (!(v < kCoordLimit)) return static_cast<std::int32_t>(kCoordLimit);
    return static_cast<std::int32_t>(v);
}

inline std::uint8_t blend(double src, std::uint8_t dst, double opacity) {
    // cv2.addWeighted's saturate_cast: round half away from zero, then clamp.
    const double v = src * opacity + static_cast<double>(dst) * (1.0 - opacity);
    const double r = std::floor(v + 0.5);
    if (r <= 0.0) return 0;
    if (r >= 255.0) return 255;
    return static_cast<std::uint8_t>(r);
}

//: Bresenham, thickness 1, clipped per pixel. Only wireframe uses it, and that
//: is a debug view rather than a frame-budget path, so a test per pixel is
//: cheaper than getting a parametric clip subtly wrong.
void draw_line(std::uint8_t* frame, int height, int width, int x0, int y0,
               int x1, int y1, const std::int32_t* bgr, double opacity) {
    const int dx = std::abs(x1 - x0);
    const int dy = -std::abs(y1 - y0);
    const int sx = x0 < x1 ? 1 : -1;
    const int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    // A line to a vertex millions of pixels off screen would otherwise step one
    // pixel at a time to get there. Nothing on screen changes once the span has
    // left it, so refuse an absurd extent rather than walk it.
    if (static_cast<long long>(dx) - dy
        > 32LL * (static_cast<long long>(width) + height)) return;
    for (;;) {
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            std::uint8_t* px = frame + (static_cast<std::size_t>(y0) * width + x0) * 3;
            if (opacity < 0.99) {
                for (int c = 0; c < 3; ++c)
                    px[c] = blend(static_cast<double>(bgr[c]), px[c], opacity);
            } else {
                for (int c = 0; c < 3; ++c)
                    px[c] = static_cast<std::uint8_t>(bgr[c]);
            }
        }
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

//: Even-odd scanline fill. The faces this rasteriser is given are convex
//: triangles and quads, where even-odd and non-zero agree, so the simpler rule
//: is the right one.
//
// The crossing test is half-open in y - an edge counts on the scanline it
// starts on and not on the one it ends on - which is what stops a shared vertex
// being counted twice and tearing a hole across the middle of a polygon. A
// perfectly horizontal edge contributes nothing, which is correct: its two
// neighbours already bracket that scanline.
//
// A span runs ceil(x_enter) .. floor(x_leave) inclusive, so two faces meeting
// on a column both paint it. That overlap is invisible - they are drawn back to
// front and the nearer one wins - where the other rounding would leave a
// one-pixel crack of background between every pair of faces.
//
// There is no antialiasing here and that is deliberate. The Python path fills
// with cv2.LINE_AA, which cost 5.6 ms of a 14 ms paint on a carved N=80 lump
// and spent 89% of it on boundaries INSIDE the lump - seams between abutting
// faces of one continuous surface, where the blend buys nothing. See the
// renderer's own note.
void fill_poly(std::uint8_t* frame, int height, int width,
               const std::int32_t* pts, int n, const std::int32_t* bgr,
               double opacity, std::vector<double>& crossings) {
    if (n < 3) return;

    std::int32_t y_lo = pts[1], y_hi = pts[1];
    for (int i = 1; i < n; ++i) {
        y_lo = std::min(y_lo, pts[2 * i + 1]);
        y_hi = std::max(y_hi, pts[2 * i + 1]);
    }
    const int y_start = std::max(0, static_cast<int>(y_lo));
    const int y_end = std::min(height - 1, static_cast<int>(y_hi));

    const bool translucent = opacity < 0.99;
    const std::uint8_t solid[3] = {static_cast<std::uint8_t>(bgr[0]),
                                   static_cast<std::uint8_t>(bgr[1]),
                                   static_cast<std::uint8_t>(bgr[2])};

    for (int y = y_start; y <= y_end; ++y) {
        crossings.clear();
        for (int i = 0; i < n; ++i) {
            const int j = (i + 1 == n) ? 0 : i + 1;
            const double x0 = pts[2 * i], y0 = pts[2 * i + 1];
            const double x1 = pts[2 * j], y1 = pts[2 * j + 1];
            if (y0 == y1) continue;
            const double lo = std::min(y0, y1), hi = std::max(y0, y1);
            if (y < lo || y >= hi) continue;
            crossings.push_back(x0 + (y - y0) * (x1 - x0) / (y1 - y0));
        }
        if (crossings.size() < 2) continue;
        std::sort(crossings.begin(), crossings.end());
        for (std::size_t k = 0; k + 1 < crossings.size(); k += 2) {
            int xa = static_cast<int>(std::ceil(crossings[k]));
            int xb = static_cast<int>(std::floor(crossings[k + 1]));
            xa = std::max(xa, 0);
            xb = std::min(xb, width - 1);
            if (xb < xa) continue;
            std::uint8_t* row = frame + (static_cast<std::size_t>(y) * width + xa) * 3;
            if (translucent) {
                for (int x = xa; x <= xb; ++x, row += 3)
                    for (int c = 0; c < 3; ++c)
                        row[c] = blend(static_cast<double>(bgr[c]), row[c], opacity);
            } else {
                for (int x = xa; x <= xb; ++x, row += 3) {
                    row[0] = solid[0];
                    row[1] = solid[1];
                    row[2] = solid[2];
                }
            }
        }
    }
}

}  // namespace

void rotation_matrix(double rx_deg, double ry_deg, double rz_deg, double out[9]) {
    constexpr double kDeg = 3.14159265358979323846 / 180.0;
    const double cx = std::cos(rx_deg * kDeg), sx = std::sin(rx_deg * kDeg);
    const double cy = std::cos(ry_deg * kDeg), sy = std::sin(ry_deg * kDeg);
    const double cz = std::cos(rz_deg * kDeg), sz = std::sin(rz_deg * kDeg);

    // Rz * Ry * Rx, multiplied out here rather than written as an expanded
    // product, so it cannot drift from Transform3D.get_rotation_matrix.
    const double Rx[9] = {1, 0, 0, 0, cx, -sx, 0, sx, cx};
    const double Ry[9] = {cy, 0, sy, 0, 1, 0, -sy, 0, cy};
    const double Rz[9] = {cz, -sz, 0, sz, cz, 0, 0, 0, 1};
    double zy[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            zy[r * 3 + c] = Rz[r * 3] * Ry[c] + Rz[r * 3 + 1] * Ry[3 + c]
                            + Rz[r * 3 + 2] * Ry[6 + c];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out[r * 3 + c] = zy[r * 3] * Rx[c] + zy[r * 3 + 1] * Rx[3 + c]
                             + zy[r * 3 + 2] * Rx[6 + c];
}

FaceList collect(const std::vector<ItemGeom>& items, const Camera& cam,
                 const Light& light) {
    FaceList out;
    out.pts_off.push_back(0);

    std::vector<double> world;         // v_count * 3, reused between items
    std::vector<double> depth;         // v_count
    std::vector<std::int32_t> screen;  // v_count * 2
    std::vector<std::uint8_t> valid;   // v_count

    const double half_w = cam.screen_w / 2.0;
    const double half_h = cam.screen_h / 2.0;

    for (std::size_t index = 0; index < items.size(); ++index) {
        const ItemGeom& it = items[index];
        if (it.v_count <= 0 || it.face_count <= 0) continue;

        double R[9];
        rotation_matrix(it.xf[3], it.xf[4], it.xf[5], R);

        world.resize(static_cast<std::size_t>(it.v_count) * 3);
        depth.resize(static_cast<std::size_t>(it.v_count));
        screen.resize(static_cast<std::size_t>(it.v_count) * 2);
        valid.assign(static_cast<std::size_t>(it.v_count), 0);

        for (int v = 0; v < it.v_count; ++v) {
            // Scale, rotate, translate - Transform3D.transform_points.
            const double lx = it.verts[3 * v] * it.xf[6];
            const double ly = it.verts[3 * v + 1] * it.xf[7];
            const double lz = it.verts[3 * v + 2] * it.xf[8];
            const double wx = R[0] * lx + R[1] * ly + R[2] * lz + it.xf[0];
            const double wy = R[3] * lx + R[4] * ly + R[5] * lz + it.xf[1];
            const double wz = R[6] * lx + R[7] * ly + R[8] * lz + it.xf[2];
            world[3 * v] = wx;
            world[3 * v + 1] = wy;
            world[3 * v + 2] = wz;

            // Camera3D.project_points: z is measured back from the camera, so a
            // point in front of it has rel_z > near.
            const double rel_x = wx - cam.px;
            const double rel_y = wy - cam.py;
            const double rel_z = cam.pz - wz;
            depth[v] = rel_z;
            if (rel_z > cam.near_) {
                valid[v] = 1;
                screen[2 * v] = to_screen_int(rel_x * cam.focal / rel_z + half_w);
                screen[2 * v + 1] = to_screen_int(-rel_y * cam.focal / rel_z + half_h);
            } else {
                // Python leaves these at 0 and masks the face out; so do we.
                screen[2 * v] = 0;
                screen[2 * v + 1] = 0;
            }
        }

        for (int f = 0; f < it.face_count; ++f) {
            const std::int32_t begin = it.face_off[f];
            const std::int32_t end = it.face_off[f + 1];
            const int k = static_cast<int>(end - begin);
            if (k < 1) continue;

            bool on_screen = true;
            for (int c = 0; c < k; ++c) {
                const std::int32_t vi = it.face_idx[begin + c];
                if (vi < 0 || vi >= it.v_count || !valid[vi]) {
                    on_screen = false;
                    break;
                }
            }
            if (!on_screen) continue;

            const std::int32_t v0 = it.face_idx[begin];
            double shade = 1.0;
            if (k >= 3) {
                const std::int32_t v1 = it.face_idx[begin + 1];
                const std::int32_t v2 = it.face_idx[begin + 2];
                // Backface cull in screen space, in 64-bit: two screen
                // coordinates a few million apart square past 32 bits.
                const long long ax = screen[2 * v0], ay = screen[2 * v0 + 1];
                const long long bx = screen[2 * v1], by = screen[2 * v1 + 1];
                const long long cx = screen[2 * v2], cy = screen[2 * v2 + 1];
                if ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax) <= 0) continue;

                if (!it.wireframe) {
                    const double e1x = world[3 * v1] - world[3 * v0];
                    const double e1y = world[3 * v1 + 1] - world[3 * v0 + 1];
                    const double e1z = world[3 * v1 + 2] - world[3 * v0 + 2];
                    const double e2x = world[3 * v2] - world[3 * v0];
                    const double e2y = world[3 * v2 + 1] - world[3 * v0 + 1];
                    const double e2z = world[3 * v2 + 2] - world[3 * v0 + 2];
                    double nx = e1y * e2z - e1z * e2y;
                    double ny = e1z * e2x - e1x * e2z;
                    double nz = e1x * e2y - e1y * e2x;
                    const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
                    if (len <= 1e-6) {
                        nx = 0.0; ny = 0.0; nz = 1.0;
                    } else {
                        nx /= len; ny /= len; nz /= len;
                    }
                    const double dot = std::fabs(nx * light.x + ny * light.y + nz * light.z);
                    shade = std::max(light.ambient,
                                     std::min(1.0, light.ambient
                                         + (1.0 - light.ambient) * dot * light.intensity));
                }
            }

            double mean_depth = 0.0;
            for (int c = 0; c < k; ++c) mean_depth += depth[it.face_idx[begin + c]];
            mean_depth /= static_cast<double>(k);

            for (int c = 0; c < k; ++c) {
                const std::int32_t vi = it.face_idx[begin + c];
                out.pts.push_back(screen[2 * vi]);
                out.pts.push_back(screen[2 * vi + 1]);
            }
            out.pts_off.push_back(static_cast<std::int32_t>(out.pts.size() / 2));
            out.depth.push_back(mean_depth);
            out.item.push_back(static_cast<std::int32_t>(index));
            out.face_no.push_back(f);
            for (int c = 0; c < 3; ++c) {
                // A wireframe line is the material's colour undimmed; a filled
                // face is it scaled by the shade and truncated, which is what
                // numpy's astype(int64) does after the clip.
                const double v = it.wireframe ? it.bgr[c] : it.bgr[c] * shade;
                out.color.push_back(static_cast<std::int32_t>(std::min(255.0, v)));
            }
            out.opacity.push_back(it.opacity);
            out.wireframe.push_back(it.wireframe ? 1 : 0);
        }
    }
    return out;
}

std::vector<std::int32_t> paint_order(const FaceList& faces) {
    std::vector<std::int32_t> order(faces.size());
    for (std::size_t i = 0; i < order.size(); ++i)
        order[i] = static_cast<std::int32_t>(i);
    std::sort(order.begin(), order.end(),
              [&faces](std::int32_t a, std::int32_t b) {
                  if (faces.depth[a] != faces.depth[b])
                      return faces.depth[a] > faces.depth[b];
                  if (faces.item[a] != faces.item[b])
                      return faces.item[a] < faces.item[b];
                  return faces.face_no[a] < faces.face_no[b];
              });
    return order;
}

void paint(std::uint8_t* frame, int height, int width, const FaceList& faces,
           const std::vector<std::int32_t>& order) {
    std::vector<double> crossings;
    crossings.reserve(16);
    for (const std::int32_t i : order) {
        const std::int32_t begin = faces.pts_off[i];
        const int n = faces.pts_off[i + 1] - begin;
        const std::int32_t* pts = faces.pts.data() + 2 * begin;
        const std::int32_t* bgr = faces.color.data() + 3 * i;
        const double opacity = faces.opacity[i];
        if (faces.wireframe[i]) {
            for (int c = 0; c < n; ++c) {
                const int d = (c + 1 == n) ? 0 : c + 1;
                draw_line(frame, height, width, pts[2 * c], pts[2 * c + 1],
                          pts[2 * d], pts[2 * d + 1], bgr, opacity);
            }
        } else {
            fill_poly(frame, height, width, pts, n, bgr, opacity, crossings);
        }
    }
}

}  // namespace raster3d
