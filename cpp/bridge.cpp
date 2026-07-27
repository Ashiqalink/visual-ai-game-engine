#include <pybind11/pybind11.h>
#include "engine.hpp"

namespace py = pybind11;
using namespace vision_engine;

PYBIND11_MODULE(engine_core, m) {
    m.doc() = "High-performance C++ Game Engine core with pybind11 bindings";

    py::class_<GameEngine>(m, "GameEngine")
        .def(py::init<float, float>(),
             py::arg("width") = 800.0f,
             py::arg("height") = 600.0f)
        .def("update", &GameEngine::update, py::arg("dt"), "Update physics loop for elapsed time dt")
        .def("set_target_position", &GameEngine::set_target_position, py::arg("x"), py::arg("y"), "Set vision target coordinates")
        .def("get_x", &GameEngine::get_x)
        .def("get_y", &GameEngine::get_y)
        .def("get_target_x", &GameEngine::get_target_x)
        .def("get_target_y", &GameEngine::get_target_y)
        .def("get_width", &GameEngine::get_width)
        .def("get_height", &GameEngine::get_height);
}
