#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.hpp"

namespace py = pybind11;
using namespace vision_engine;

PYBIND11_MODULE(engine_core, m) {
    m.doc() = "High-performance C++ Game Engine core with pybind11 bindings";

    py::enum_<ShaderType>(m, "ShaderType")
        .value("PBR_Standard", ShaderType::PBR_Standard)
        .value("Unlit", ShaderType::Unlit)
        .value("Transparent", ShaderType::Transparent)
        .value("Phong", ShaderType::Phong)
        .value("Custom", ShaderType::Custom)
        .export_values();

    py::class_<Material>(m, "Material")
        .def(py::init<>())
        .def_readwrite("name", &Material::name)
        .def_readwrite("shader_type", &Material::shader_type)
        .def_readwrite("normal_map", &Material::normal_map)
        .def_readwrite("roughness", &Material::roughness)
        .def_readwrite("metallic", &Material::metallic)
        .def_readwrite("opacity", &Material::opacity)
        .def_property("base_color",
            [](const Material& mat) { return std::vector<float>(mat.base_color, mat.base_color + 4); },
            [](Material& mat, const std::vector<float>& color) {
                for (size_t i = 0; i < 4 && i < color.size(); ++i) mat.base_color[i] = color[i];
            })
        .def_property("emission",
            [](const Material& mat) { return std::vector<float>(mat.emission, mat.emission + 3); },
            [](Material& mat, const std::vector<float>& color) {
                for (size_t i = 0; i < 3 && i < color.size(); ++i) mat.emission[i] = color[i];
            });

    py::class_<Entity>(m, "Entity")
        .def_readonly("id", &Entity::id)
        .def_readwrite("name", &Entity::name)
        .def_readwrite("x", &Entity::x)
        .def_readwrite("y", &Entity::y)
        .def_readwrite("z", &Entity::z)
        .def_readwrite("rx", &Entity::rx)
        .def_readwrite("ry", &Entity::ry)
        .def_readwrite("rz", &Entity::rz)
        .def_readwrite("vx", &Entity::vx)
        .def_readwrite("vy", &Entity::vy)
        .def_readwrite("vz", &Entity::vz)
        .def_readwrite("vrx", &Entity::vrx)
        .def_readwrite("vry", &Entity::vry)
        .def_readwrite("vrz", &Entity::vrz)
        .def_readwrite("width", &Entity::width)
        .def_readwrite("height", &Entity::height)
        .def_readwrite("depth", &Entity::depth)
        .def_readwrite("active", &Entity::active)
        .def_readwrite("material", &Entity::material);

    py::class_<Block>(m, "Block")
        .def_readonly("x", &Block::x)
        .def_readonly("y", &Block::y)
        .def_readonly("width", &Block::width)
        .def_readonly("height", &Block::height)
        .def_readonly("health", &Block::health)
        .def_readonly("max_health", &Block::max_health)
        .def_readonly("active", &Block::active)
        .def_readonly("material", &Block::material);

    py::class_<Debris>(m, "Debris")
        .def_readonly("x", &Debris::x)
        .def_readonly("y", &Debris::y)
        .def_readonly("vx", &Debris::vx)
        .def_readonly("vy", &Debris::vy)
        .def_readonly("width", &Debris::width)
        .def_readonly("height", &Debris::height)
        .def_readonly("lifespan", &Debris::lifespan)
        .def_readonly("active", &Debris::active)
        .def_readonly("material", &Debris::material);

    py::class_<GameEngine>(m, "GameEngine")
        .def(py::init<float, float>(),
             py::arg("width") = 800.0f,
             py::arg("height") = 600.0f)
        .def("update", &GameEngine::update, py::arg("dt"), "Update physics loop for elapsed time dt")
        .def("set_target_position", &GameEngine::set_target_position, py::arg("x"), py::arg("y"), "Set vision target coordinates")
        .def("add_entity", &GameEngine::add_entity,
             py::arg("name") = "Entity",
             py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f,
             py::arg("vx") = 0.0f, py::arg("vy") = 0.0f, py::arg("vz") = 0.0f,
             py::arg("w") = 1.0f, py::arg("h") = 1.0f, py::arg("d") = 1.0f,
             py::arg("material") = Material())
        .def("add_3d_element", &GameEngine::add_3d_element,
             py::arg("name") = "3DElement",
             py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f,
             py::arg("rx") = 0.0f, py::arg("ry") = 0.0f, py::arg("rz") = 0.0f,
             py::arg("vx") = 0.0f, py::arg("vy") = 0.0f, py::arg("vz") = 0.0f,
             py::arg("vrx") = 0.0f, py::arg("vry") = 0.0f, py::arg("vrz") = 0.0f,
             py::arg("scale") = 1.0f,
             py::arg("material") = Material())
        .def("get_entities", &GameEngine::get_entities)
        .def("clear_entities", &GameEngine::clear_entities)
        .def("add_block", &GameEngine::add_block,
             py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"), py::arg("health"),
             py::arg("material") = Material())
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
