// gridops_bind.cpp - the pybind11 surface of the grid morphology.
//
// Kept apart from gridops.cpp so that stays plain C++. Everything here is
// marshalling: a boolean grid in, one or two boolean grids out, allocated as
// numpy arrays so the caller gets them with no further copy.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

#include "gridops.hpp"

namespace py = pybind11;

namespace {

using BoolGrid = py::array_t<bool, py::array::c_style | py::array::forcecast>;

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("gridops: ") + what);
}

//: The grid's shape as three ints, checking it is a 3-D grid on the way.
std::vector<int> shape_of(const BoolGrid& grid) {
    require(grid.ndim() == 3, "grid must be 3-D");
    return {static_cast<int>(grid.shape(0)), static_cast<int>(grid.shape(1)),
            static_cast<int>(grid.shape(2))};
}

BoolGrid make_like(const std::vector<int>& shape) {
    return BoolGrid({static_cast<py::ssize_t>(shape[0]),
                     static_cast<py::ssize_t>(shape[1]),
                     static_cast<py::ssize_t>(shape[2])});
}

}  // namespace

void bind_gridops(py::module_& m) {
    m.def(
        "grid_erode",
        [](BoolGrid grid) {
            const std::vector<int> shape = shape_of(grid);
            BoolGrid out = make_like(shape);
            {
                py::gil_scoped_release unlock;
                gridops::erode(grid.data(), out.mutable_data(), shape.data());
            }
            return out;
        },
        py::arg("grid"),
        "Occupied cells all six of whose neighbours are occupied. The grid's "
        "rim always peels: its neighbours off the grid count as empty.");

    m.def(
        "grid_wall_layers",
        [](BoolGrid grid, int depth, int skin_depth) {
            require(depth >= 0 && skin_depth >= 0, "depths must not be negative");
            const std::vector<int> shape = shape_of(grid);
            BoolGrid core = make_like(shape), skin = make_like(shape);
            {
                py::gil_scoped_release unlock;
                gridops::wall_layers(grid.data(), depth, skin_depth,
                                     shape.data(), core.mutable_data(),
                                     skin.mutable_data());
            }
            return py::make_tuple(core, skin);
        },
        py::arg("grid"), py::arg("depth"), py::arg("skin_depth"),
        "(core, skin): what a hollow takes out, and the outer shell it must "
        "not. Depth is counted from any surface, a cavity's included.");

    m.def(
        "grid_downsample",
        [](BoolGrid grid, int factor) {
            require(factor >= 1, "factor must be at least 1");
            const std::vector<int> shape = shape_of(grid);
            const std::vector<int> out_shape = {shape[0] / factor,
                                                shape[1] / factor,
                                                shape[2] / factor};
            BoolGrid out = make_like(out_shape);
            if (out_shape[0] && out_shape[1] && out_shape[2]) {
                py::gil_scoped_release unlock;
                gridops::downsample(grid.data(), out.mutable_data(),
                                    shape.data(), factor);
            }
            return out;
        },
        py::arg("grid"), py::arg("factor"),
        "The grid in blocks of `factor`, a block filled when at least half of "
        "it is. Trailing cells that do not fill a block are dropped.");
}
