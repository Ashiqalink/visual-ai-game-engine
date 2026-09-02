// voxelmesh_bind.cpp - the pybind11 surface of the voxel mesher.
//
// Kept apart from voxelmesh.cpp so the mesher itself stays plain C++ with no
// Python in it. Everything here is marshalling: two boolean grids and a box
// in, a vertex array and a face array out.
//
// The faces come back as an (F, 4) int32 array rather than the list of lists
// Mesh3D holds, because the caller wants both: the list for the dataclass's
// declared type, and the flat array to prime Mesh3D.face_arrays with so the
// compiled rasteriser never has to rebuild it. Doing the split in Python
// keeps every Mesh3D detail out of this file.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "voxelmesh.hpp"

namespace py = pybind11;

namespace {

using BoolGrid = py::array_t<bool, py::array::c_style | py::array::forcecast>;

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("voxelmesh: ") + what);
}

}  // namespace

void bind_voxelmesh(py::module_& m) {
    m.def(
        "region_mesh",
        [](BoolGrid grid, BoolGrid padded, std::array<int, 3> lo,
           std::array<int, 3> hi, double cell) -> py::object {
            require(grid.ndim() == 3, "grid must be 3-D");
            require(padded.ndim() == 3, "padded grid must be 3-D");
            int shape[3];
            for (int a = 0; a < 3; ++a) {
                shape[a] = static_cast<int>(grid.shape(a));
                require(padded.shape(a) == shape[a] + 2,
                        "padded grid must be the grid plus one cell of margin");
                require(0 <= lo[a] && lo[a] <= hi[a] && hi[a] <= shape[a],
                        "box must lie inside the grid");
            }

            voxelmesh::Mesh mesh;
            {
                py::gil_scoped_release unlock;
                mesh = voxelmesh::region_mesh(grid.data(), padded.data(), shape,
                                              lo.data(), hi.data(), cell);
            }
            if (mesh.face_count == 0) return py::none();

            py::array_t<double> verts(
                {static_cast<py::ssize_t>(mesh.vertices.size() / 3),
                 static_cast<py::ssize_t>(3)});
            std::copy(mesh.vertices.begin(), mesh.vertices.end(),
                      verts.mutable_data());
            py::array_t<std::int32_t> faces(
                {static_cast<py::ssize_t>(mesh.face_count),
                 static_cast<py::ssize_t>(4)});
            std::copy(mesh.faces.begin(), mesh.faces.end(),
                      faces.mutable_data());
            return py::make_tuple(verts, faces);
        },
        py::arg("grid"), py::arg("padded"), py::arg("lo"), py::arg("hi"),
        py::arg("cell"),
        "Mesh grid[lo:hi] into exposed quads, judging exposure against the "
        "whole padded grid. Returns (vertices (V, 3) float64, faces (F, 4) "
        "int32), or None when the box raises no face.");

    m.def(
        "voxel_sides",
        []() {
            std::vector<py::tuple> out;
            for (const voxelmesh::Side& s : voxelmesh::SIDES) {
                std::vector<py::tuple> corners;
                for (const auto& c : s.corner) {
                    corners.push_back(py::make_tuple(c[0], c[1], c[2]));
                }
                out.push_back(py::make_tuple(
                    py::make_tuple(s.d[0], s.d[1], s.d[2]),
                    py::cast(corners)));
            }
            return out;
        },
        "The compiled side table, so a test can compare it against "
        "voxel._SIDES instead of trusting the transcription.");
}
