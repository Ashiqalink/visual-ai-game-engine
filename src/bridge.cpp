#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.hpp"

namespace py = pybind11;
using namespace vision_engine;

PYBIND11_MODULE(engine_core, m) {
    m.doc() = "High-performance C++ Game Engine core with pybind11 bindings";

    py::class_<Block>(m, "Block")
        .def_readonly("x", &Block::x)
        .def_readonly("y", &Block::y)
        .def_readonly("width", &Block::width)
        .def_readonly("height", &Block::height)
        .def_readonly("health", &Block::health)
        .def_readonly("max_health", &Block::max_health)
        .def_readonly("active", &Block::active);

    py::class_<Debris>(m, "Debris")
        .def_readonly("x", &Debris::x)
        .def_readonly("y", &Debris::y)
        .def_readonly("vx", &Debris::vx)
        .def_readonly("vy", &Debris::vy)
        .def_readonly("width", &Debris::width)
        .def_readonly("height", &Debris::height)
        .def_readonly("lifespan", &Debris::lifespan)
        .def_readonly("active", &Debris::active);

    py::class_<GameEngine>(m, "GameEngine")
        .def(py::init<float, float>(),
             py::arg("width") = 800.0f,
             py::arg("height") = 600.0f)
        .def("update", &GameEngine::update, py::arg("dt"), "Update physics loop for elapsed time dt")
        .def("set_target_position", &GameEngine::set_target_position, py::arg("x"), py::arg("y"), "Set vision target coordinates")
        .def("add_block", &GameEngine::add_block, py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"), py::arg("health"))
        .def("get_blocks", &GameEngine::get_blocks)
        .def("get_debris", &GameEngine::get_debris)
        .def("clear_blocks", &GameEngine::clear_blocks)
        .def("get_x", &GameEngine::get_x)
        .def("get_y", &GameEngine::get_y)
        .def("get_target_x", &GameEngine::get_target_x)
        .def("get_target_y", &GameEngine::get_target_y)
        .def("get_width", &GameEngine::get_width)
        .def("get_height", &GameEngine::get_height);
}
