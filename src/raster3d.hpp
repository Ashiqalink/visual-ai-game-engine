// raster3d.hpp - the software rasteriser's C++ core.
//
// This is the compiled half of visual_ai.three_d.renderer. It does what
// Renderer3D._collect_faces and Renderer3D._paint did in Python: transform a
// mesh's vertices, project them, cull the faces that face away or fall off the
// screen, shade the survivors, sort them back to front, and fill them into an
// OpenCV frame buffer.
//
// The Python path stays and stays authoritative: it is what runs when the
// extension is not built, and tests/three_d/test_raster3d.py compares the two.
// Where the two deliberately differ - the fill rule, and antialiasing - the
// difference is documented at the point it happens, not here.
//
// Nothing in this header touches Python. The pybind11 layer is raster3d_bind.cpp,
// so this can be compiled and tested as plain C++.
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace raster3d {

//: A pinhole camera, matching visual_ai.three_d.camera.Camera3D exactly: no
//: look-at, no far clip. `screen_w`/`screen_h` are the camera's own screen
//: size, which is not necessarily the frame's - Camera3D projects against its
//: own, and a caller is free to hand a differently sized frame.
struct Camera {
    double px = 0.0, py = 0.0, pz = 500.0;
    double focal = 1.0;
    double near_ = 0.1;
    double screen_w = 800.0;
    double screen_h = 600.0;
};

//: The single directional light, already normalised, plus the two shading
//: constants Renderer3D holds.
struct Light {
    double x = 0.0, y = 0.0, z = 1.0;
    double ambient = 0.45;
    double intensity = 0.85;
};

//: One (mesh, transform, material) of a scene, as flat borrowed arrays.
//
// Faces are CSR - `face_idx` concatenated, `face_off` of length
// `face_count + 1` - rather than the Python side's buckets of equal vertex
// count. The buckets existed to keep numpy working on rectangular arrays; a
// C++ loop does not care, and iterating faces in their original order means a
// face's number IS its position, so the draw order needs no separate array.
struct ItemGeom {
    const double* verts = nullptr;      // v_count * 3, the mesh's LOCAL vertices
    int v_count = 0;
    const std::int32_t* face_idx = nullptr;
    const std::int32_t* face_off = nullptr;
    int face_count = 0;
    double xf[9] = {0, 0, 0, 0, 0, 0, 1, 1, 1};  // x,y,z, rx,ry,rz (deg), sx,sy,sz
    double bgr[3] = {255, 255, 255};             // material base colour, 0-255
    double opacity = 1.0;
    bool wireframe = false;
};

//: Every face that survived the cull, as parallel arrays - the C++ twin of
//: renderer._FaceBatch. `pts` is CSR over `pts_off`, two ints (x, y) a corner.
struct FaceList {
    std::vector<double> depth;          // F: mean depth, the sort key
    std::vector<std::int32_t> item;     // F: which scene item
    std::vector<std::int32_t> face_no;  // F: position in the mesh's face list
    std::vector<std::int32_t> pts;      // 2 * corners: screen x, y
    std::vector<std::int32_t> pts_off;  // F + 1
    std::vector<std::int32_t> color;    // 3 * F: shaded BGR
    std::vector<double> opacity;        // F
    std::vector<std::uint8_t> wireframe;  // F
    std::size_t size() const { return depth.size(); }
};

//: Rz * Ry * Rx from degrees, row-major - Transform3D.get_rotation_matrix.
void rotation_matrix(double rx_deg, double ry_deg, double rz_deg, double out[9]);

//: Transform, project, cull and shade every item into one face list.
FaceList collect(const std::vector<ItemGeom>& items, const Camera& cam,
                 const Light& light);

//: Painter's order: furthest first, ties by item then face number. This is
//: what np.lexsort((face_no, item, -depth)) gives, and the tie-breaks are why
//: it is not just a depth sort - see Renderer3D._paint.
std::vector<std::int32_t> paint_order(const FaceList& faces);

//: Fill the faces into a contiguous HxWx3 BGR frame, in the given order.
void paint(std::uint8_t* frame, int height, int width, const FaceList& faces,
           const std::vector<std::int32_t>& order);

}  // namespace raster3d
