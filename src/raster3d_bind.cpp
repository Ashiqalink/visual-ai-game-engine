// raster3d_bind.cpp - the pybind11 surface of the software rasteriser.
//
// Kept apart from raster3d.cpp so the rasteriser itself stays plain C++ with no
// Python in it. Everything here is marshalling: flat numpy arrays in, a painted
// frame or a dict of face arrays out.
//
// The scene arrives already flattened by visual_ai.three_d.renderer - one
// vertex array and one CSR face pair per item, plus parallel arrays of
// transform, colour, opacity and wireframe flag. Unpacking Mesh3D, Transform3D
// and Material in Python and handing over arrays is what keeps this file free
// of type casters, and it is cheap: those objects are read once per item per
// frame, not once per face.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

#include "raster3d.hpp"

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using FrameArray = py::array_t<std::uint8_t, py::array::c_style>;

//: The scene as the Python side hands it over. Held by value so the forcecast
//: temporaries outlive the pointers taken into them.
struct Scene {
    std::vector<DoubleArray> verts;
    std::vector<IntArray> face_idx;
    std::vector<IntArray> face_off;
    std::vector<raster3d::ItemGeom> items;
    raster3d::Camera cam;
    raster3d::Light light;
};

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("raster3d: ") + what);
}

Scene build_scene(std::vector<DoubleArray> verts, std::vector<IntArray> face_idx,
                  std::vector<IntArray> face_off, DoubleArray xforms,
                  DoubleArray colors, DoubleArray opacity,
                  py::array_t<bool, py::array::c_style | py::array::forcecast> wireframe,
                  DoubleArray cam, DoubleArray light) {
    Scene scene;
    const std::size_t n = verts.size();
    require(face_idx.size() == n && face_off.size() == n,
            "one vertex array, index array and offset array per item");
    require(xforms.ndim() == 2 && static_cast<std::size_t>(xforms.shape(0)) == n
                && xforms.shape(1) == 9,
            "xforms must be (items, 9)");
    require(colors.ndim() == 2 && static_cast<std::size_t>(colors.shape(0)) == n
                && colors.shape(1) == 3,
            "colors must be (items, 3)");
    require(opacity.ndim() == 1 && static_cast<std::size_t>(opacity.shape(0)) == n,
            "opacity must be (items,)");
    require(wireframe.ndim() == 1 && static_cast<std::size_t>(wireframe.shape(0)) == n,
            "wireframe must be (items,)");
    require(cam.ndim() == 1 && cam.shape(0) == 7,
            "cam must be (px, py, pz, focal, near, screen_w, screen_h)");
    require(light.ndim() == 1 && light.shape(0) == 5,
            "light must be (x, y, z, ambient, intensity)");

    const double* c = cam.data();
    scene.cam = {c[0], c[1], c[2], c[3], c[4], c[5], c[6]};
    const double* l = light.data();
    scene.light = {l[0], l[1], l[2], l[3], l[4]};

    scene.verts = std::move(verts);
    scene.face_idx = std::move(face_idx);
    scene.face_off = std::move(face_off);
    scene.items.resize(n);

    const double* xf = xforms.data();
    const double* col = colors.data();
    const double* op = opacity.data();
    const bool* wf = wireframe.data();

    for (std::size_t i = 0; i < n; ++i) {
        const auto& v = scene.verts[i];
        const auto& fi = scene.face_idx[i];
        const auto& fo = scene.face_off[i];
        require(v.ndim() == 2 && v.shape(1) == 3, "vertices must be (V, 3)");
        require(fi.ndim() == 1 && fo.ndim() == 1, "face arrays must be 1-D");
        require(fo.shape(0) >= 1, "face offsets must start with a 0");

        raster3d::ItemGeom& g = scene.items[i];
        g.verts = v.data();
        g.v_count = static_cast<int>(v.shape(0));
        g.face_idx = fi.data();
        g.face_off = fo.data();
        g.face_count = static_cast<int>(fo.shape(0)) - 1;
        // A trailing offset past the end of the index array would read off it.
        require(g.face_count < 0
                    || fo.data()[g.face_count] <= static_cast<std::int32_t>(fi.shape(0)),
                "face offsets run past the index array");
        for (int k = 0; k < 9; ++k) g.xf[k] = xf[i * 9 + k];
        for (int k = 0; k < 3; ++k) g.bgr[k] = col[i * 3 + k];
        g.opacity = op[i];
        g.wireframe = wf[i];
    }
    return scene;
}

template <typename T>
py::array_t<T> to_array(const std::vector<T>& v) {
    py::array_t<T> out(static_cast<py::ssize_t>(v.size()));
    if (!v.empty()) std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

}  // namespace

void bind_raster3d(py::module_& m) {
    m.def(
        "render_scene3d",
        [](FrameArray frame, std::vector<DoubleArray> verts,
           std::vector<IntArray> face_idx, std::vector<IntArray> face_off,
           DoubleArray xforms, DoubleArray colors, DoubleArray opacity,
           py::array_t<bool, py::array::c_style | py::array::forcecast> wireframe,
           DoubleArray cam, DoubleArray light) {
            require(frame.ndim() == 3 && frame.shape(2) == 3,
                    "frame must be (H, W, 3) uint8");
            require(frame.writeable(), "frame must be writeable");
            Scene scene = build_scene(std::move(verts), std::move(face_idx),
                                      std::move(face_off), xforms, colors,
                                      opacity, wireframe, cam, light);
            const int height = static_cast<int>(frame.shape(0));
            const int width = static_cast<int>(frame.shape(1));
            std::uint8_t* pixels = frame.mutable_data();
            std::size_t painted = 0;
            {
                // The GIL is only needed for the arrays' lifetimes, which the
                // Scene already holds; the rasteriser itself touches nothing
                // Python, so a second thread can render while this one runs.
                py::gil_scoped_release release;
                raster3d::FaceList faces = raster3d::collect(scene.items, scene.cam,
                                                             scene.light);
                painted = faces.size();
                raster3d::paint(pixels, height, width, faces,
                                raster3d::paint_order(faces));
            }
            return painted;
        },
        py::arg("frame"), py::arg("verts"), py::arg("face_idx"),
        py::arg("face_off"), py::arg("xforms"), py::arg("colors"),
        py::arg("opacity"), py::arg("wireframe"), py::arg("cam"),
        py::arg("light"),
        "Transform, cull, shade, sort and fill a scene into an HxWx3 BGR "
        "frame in place. Returns the number of faces painted.");

    m.def(
        "collect_faces3d",
        [](std::vector<DoubleArray> verts, std::vector<IntArray> face_idx,
           std::vector<IntArray> face_off, DoubleArray xforms,
           DoubleArray colors, DoubleArray opacity,
           py::array_t<bool, py::array::c_style | py::array::forcecast> wireframe,
           DoubleArray cam, DoubleArray light) {
            Scene scene = build_scene(std::move(verts), std::move(face_idx),
                                      std::move(face_off), xforms, colors,
                                      opacity, wireframe, cam, light);
            raster3d::FaceList faces = raster3d::collect(scene.items, scene.cam,
                                                         scene.light);
            const std::vector<std::int32_t> order = raster3d::paint_order(faces);

            py::dict out;
            out["depth"] = to_array(faces.depth);
            out["item"] = to_array(faces.item);
            out["face_no"] = to_array(faces.face_no);
            out["pts"] = to_array(faces.pts);
            out["pts_off"] = to_array(faces.pts_off);
            out["color"] = to_array(faces.color);
            out["opacity"] = to_array(faces.opacity);
            out["wireframe"] = to_array(faces.wireframe);
            out["order"] = to_array(order);
            return out;
        },
        py::arg("verts"), py::arg("face_idx"), py::arg("face_off"),
        py::arg("xforms"), py::arg("colors"), py::arg("opacity"),
        py::arg("wireframe"), py::arg("cam"), py::arg("light"),
        "The cull/shade/sort stage on its own, as flat arrays. This exists so "
        "the Python renderer and the compiled one can be compared face by face "
        "rather than only pixel by pixel - see tests/three_d/test_raster3d.py.");
}
