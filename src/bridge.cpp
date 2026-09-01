#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "engine.hpp"

namespace py = pybind11;
using namespace vision_engine;

// The block and debris vectors are the engine's own storage, not values to be
// converted. Without these two lines pybind11/stl.h copies each one into a
// fresh Python list on every get_blocks()/get_debris() call, which is what made
// the compiled core hand out a snapshot where the fallback hands out the list
// it iterates. See the bindings for BlockList/DebrisList below.
PYBIND11_MAKE_OPAQUE(std::vector<vision_engine::Block>)
PYBIND11_MAKE_OPAQUE(std::vector<vision_engine::Debris>)

namespace {

// The SDK exposes exactly one Material type: visual_ai.material.Material.
//
// This module used to publish a second, unrelated `engine_core.Material`, so a
// game that built a `visual_ai.Material` (which is what every example and every
// game here does) could not pass it to the compiled engine at all, and
// `isinstance(entity.material, visual_ai.Material)` was False on machines where
// the C++ core happened to be built. The type caster below translates the
// Python dataclass to and from the C++ struct instead, so callers only ever see
// the one type.
//
// The lookup is deliberately lazy — resolving it at module-import time would
// pull `visual_ai` in while `visual_ai/__init__.py` is still importing us — and
// the handle is intentionally leaked, so no Python object is released after
// interpreter shutdown.
py::handle material_type() {
    static py::handle cls = []() {
        py::object material = py::module_::import("visual_ai.material").attr("Material");
        return material.release();
    }();
    return cls;
}

py::object new_material() {
    return py::reinterpret_borrow<py::object>(material_type())();
}

// visual_ai.ShaderType is a str-valued Enum whose values match these names, and
// Material.__post_init__ turns a plain string back into the enum member, so the
// value string is the common currency between the two representations.
const char* shader_to_str(ShaderType type) {
    switch (type) {
        case ShaderType::Unlit:       return "Unlit";
        case ShaderType::Transparent: return "Transparent";
        case ShaderType::Phong:       return "Phong";
        case ShaderType::Custom:      return "Custom";
        case ShaderType::PBR_Standard:
        default:                      return "PBR_Standard";
    }
}

ShaderType shader_from_py(py::handle type) {
    std::string value = py::hasattr(type, "value")
                            ? py::str(type.attr("value")).cast<std::string>()
                            : py::str(type).cast<std::string>();
    if (value == "Unlit") return ShaderType::Unlit;
    if (value == "Transparent") return ShaderType::Transparent;
    if (value == "Phong") return ShaderType::Phong;
    if (value == "Custom") return ShaderType::Custom;
    return ShaderType::PBR_Standard;
}

void load_floats(py::handle src, float* dst, size_t count) {
    auto values = src.cast<std::vector<float>>();
    for (size_t i = 0; i < count && i < values.size(); ++i) {
        dst[i] = values[i];
    }
}

}  // namespace

namespace pybind11 {
namespace detail {

template <>
struct type_caster<vision_engine::Material> {
public:
    PYBIND11_TYPE_CASTER(vision_engine::Material, const_name("visual_ai.material.Material"));

    bool load(handle src, bool) {
        if (!src || src.is_none()) return false;
        try {
            if (!isinstance(src, material_type())) return false;
            value.name = src.attr("name").cast<std::string>();
            value.shader_type = shader_from_py(src.attr("shader_type"));
            load_floats(src.attr("base_color"), value.base_color, 4);
            value.normal_map = src.attr("normal_map").cast<std::string>();
            value.roughness = src.attr("roughness").cast<float>();
            value.metallic = src.attr("metallic").cast<float>();
            load_floats(src.attr("emission"), value.emission, 3);
            value.opacity = src.attr("opacity").cast<float>();
        } catch (const error_already_set&) {
            return false;
        }
        return true;
    }

    static handle cast(const vision_engine::Material& src, return_value_policy, handle) {
        object cls = reinterpret_borrow<object>(material_type());
        object out = cls(
            py::arg("name") = src.name,
            py::arg("shader_type") = shader_to_str(src.shader_type),
            py::arg("base_color") = py::make_tuple(src.base_color[0], src.base_color[1],
                                                   src.base_color[2], src.base_color[3]),
            py::arg("normal_map") = src.normal_map,
            py::arg("roughness") = src.roughness,
            py::arg("metallic") = src.metallic,
            py::arg("emission") = py::make_tuple(src.emission[0], src.emission[1], src.emission[2]),
            py::arg("opacity") = src.opacity);
        return out.release();
    }
};

}  // namespace detail
}  // namespace pybind11

namespace {

// Resolve the `material=` keyword: absent means a default Material, and
// anything that is not a visual_ai.Material is a TypeError rather than the
// caster's generic "incompatible function arguments".
Material coerce_material(py::object& material) {
    if (material.is_none()) material = new_material();
    if (!py::isinstance(material, material_type())) {
        throw py::type_error("material must be a visual_ai.Material, got " +
                             py::str(py::type::of(material).attr("__name__")).cast<std::string>());
    }
    return material.cast<Material>();
}

// Bindings-only subclass: it pins one Python wrapper per entity.
//
// `mesh` and the material object live on the wrapper rather than in the C++
// struct (a mesh is a pure Python concept from visual_ai/render3d.py, and
// engine.hpp has no renderer). Keeping the wrapper alive for as long as the
// engine holds the entity is what makes those attributes survive a round trip
// through `get_entities()` — pybind11 hands back the *same* wrapper for a
// pointer it already knows, but only while that wrapper is still alive.
//
// The wrappers own the entities through their shared_ptr holder and hold no
// reference back to the engine, so this pins memory without creating a cycle.
class PyGameEngine : public GameEngine {
public:
    using GameEngine::GameEngine;

    py::object track(const EntityPtr& entity, py::object material, py::object mesh) {
        py::object wrapper = py::cast(entity);
        wrapper.attr("_material_py") = std::move(material);
        wrapper.attr("mesh") = std::move(mesh);
        m_wrappers.push_back(wrapper);
        return wrapper;
    }

    void forget_all() { m_wrappers.clear(); }

private:
    std::vector<py::object> m_wrappers;
};

// A read-only, live view onto one of the engine's vectors.
//
// Only `__len__` and `__getitem__` are bound, and `__getitem__` returns a copy
// of the element. That is deliberate on both counts:
//
//   * update() erases destroyed blocks and expired debris from the middle of
//     these vectors, so a reference to an element would dangle the moment
//     anything ahead of it died. A copy is exactly what the read-only Block
//     and Debris bindings already handed out, and it cannot outlive its slot.
//   * iteration therefore runs through Python's sequence protocol, which asks
//     for index 0, 1, 2 ... until IndexError. Every step re-checks the bound
//     against the vector's current size, so a loop that shrinks the engine's
//     storage mid-pass stops short instead of walking off the end - which a
//     cached C++ iterator would do.
//
// The view is not a `list`, which the fallback's return value is; `len()`,
// indexing and iteration are the whole shared surface.
template <typename T>
void bind_live_view(py::module_& m, const char* name) {
    using Vec = std::vector<T>;
    py::class_<Vec>(m, name)
        .def("__len__", [](const Vec& v) { return v.size(); })
        .def("__getitem__", [](const Vec& v, py::ssize_t i) -> T {
            const auto size = static_cast<py::ssize_t>(v.size());
            if (i < 0) i += size;
            if (i < 0 || i >= size) throw py::index_error();
            return v[static_cast<size_t>(i)];
        });
}

}  // namespace

PYBIND11_MODULE(engine_core, m) {
    m.doc() = "High-performance C++ Game Engine core with pybind11 bindings";

    py::class_<Entity, std::shared_ptr<Entity>>(m, "Entity", py::dynamic_attr())
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
        // Reads return the very object that was handed to add_entity(), the way
        // the Python fallback's dataclass field does, instead of a fresh copy
        // per access — so `ent.material is gold_mat` and in-place edits behave
        // the same on both engines.
        .def_property(
            "material",
            [](py::object self) -> py::object {
                if (py::hasattr(self, "_material_py")) return self.attr("_material_py");
                py::object material = py::cast(self.cast<Entity&>().material);
                self.attr("_material_py") = material;
                return material;
            },
            [](py::object self, py::object material) {
                self.cast<Entity&>().material = material.cast<Material>();
                self.attr("_material_py") = std::move(material);
            });

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

    bind_live_view<Block>(m, "BlockList");
    bind_live_view<Debris>(m, "DebrisList");

    py::class_<PyGameEngine>(m, "GameEngine")
        .def(py::init<float, float>(),
             py::arg("width") = 800.0f,
             py::arg("height") = 600.0f)
        .def("update", &PyGameEngine::update, py::arg("dt"), "Update physics loop for elapsed time dt")
        .def("set_target_position", &PyGameEngine::set_target_position, py::arg("x"), py::arg("y"), "Set vision target coordinates")
        // add_entity / add_3d_element return the Entity itself, not its id: the
        // fallback engine does, and callers mutate what comes back
        // (`ent.x = ...`). Read `.id` for the identifier.
        .def("add_entity",
             [](PyGameEngine& self, std::string name, float x, float y, float z,
                float vx, float vy, float vz, float w, float h, float d,
                py::object material) {
                 auto entity = self.GameEngine::add_entity(std::move(name), x, y, z, vx, vy, vz, w, h, d,
                                                           coerce_material(material));
                 return self.track(entity, std::move(material), py::none());
             },
             py::arg("name") = "Entity",
             py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f,
             py::arg("vx") = 0.0f, py::arg("vy") = 0.0f, py::arg("vz") = 0.0f,
             py::arg("w") = 1.0f, py::arg("h") = 1.0f, py::arg("d") = 1.0f,
             py::arg("material") = py::none())
        .def("add_3d_element",
             [](PyGameEngine& self, std::string name, float x, float y, float z,
                float rx, float ry, float rz, float vx, float vy, float vz,
                float vrx, float vry, float vrz, float scale,
                py::object material, py::object mesh) {
                 auto entity = self.GameEngine::add_3d_element(std::move(name), x, y, z, rx, ry, rz,
                                                               vx, vy, vz, vrx, vry, vrz, scale,
                                                               coerce_material(material));
                 return self.track(entity, std::move(material), std::move(mesh));
             },
             py::arg("name") = "3DElement",
             py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f,
             py::arg("rx") = 0.0f, py::arg("ry") = 0.0f, py::arg("rz") = 0.0f,
             py::arg("vx") = 0.0f, py::arg("vy") = 0.0f, py::arg("vz") = 0.0f,
             py::arg("vrx") = 0.0f, py::arg("vry") = 0.0f, py::arg("vrz") = 0.0f,
             py::arg("scale") = 1.0f,
             py::arg("material") = py::none(),
             py::arg("mesh") = py::none())
        // Live entities, not copies: writing to what this returns used to be
        // silently discarded by the C++ engine and honoured by the fallback.
        .def("get_entities", &PyGameEngine::get_entities)
        .def("clear_entities",
             [](PyGameEngine& self) {
                 self.GameEngine::clear_entities();
                 self.forget_all();
             })
        .def("add_block",
             [](PyGameEngine& self, float x, float y, float w, float h, float health,
                py::object material) {
                 self.GameEngine::add_block(x, y, w, h, health, coerce_material(material));
             },
             py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"), py::arg("health"),
             py::arg("material") = py::none())
        // Live views, not snapshots. The fallback returns the very list it
        // iterates, so `blocks = engine.get_blocks()` there keeps tracking the
        // engine across update()'s compaction; this used to return a fresh
        // list of copies per call, so the same line held a stale frame on any
        // machine where the compiled core loaded. reference_internal hands
        // back the vector itself - the same Python object each call, kept
        // alive by the engine - which is what closes that.
        .def("get_blocks", &PyGameEngine::get_blocks,
             py::return_value_policy::reference_internal)
        .def("get_debris", &PyGameEngine::get_debris,
             py::return_value_policy::reference_internal)
        .def("clear_blocks", &PyGameEngine::clear_blocks)
        .def("get_x", &PyGameEngine::get_x)
        .def("get_y", &PyGameEngine::get_y)
        .def("get_target_x", &PyGameEngine::get_target_x)
        .def("get_target_y", &PyGameEngine::get_target_y)
        .def("get_width", &PyGameEngine::get_width)
        .def("get_height", &PyGameEngine::get_height)
        // The same state again, as assignable attributes.
        //
        // PythonFallbackEngine holds all ten as plain instance attributes, so
        // `engine.gravity = 500` and `engine.x` are ordinary things to write
        // against it. Only the six getters above were ever bound, so that code
        // raised AttributeError on any machine where the compiled core loaded
        // instead - and which one loads depends on nothing but whether a .pyd
        // happens to be next to the package. The getters stay: they are the
        // existing API, and dropping them would break the other direction.
        .def_property("x", &PyGameEngine::get_x, &PyGameEngine::set_x)
        .def_property("y", &PyGameEngine::get_y, &PyGameEngine::set_y)
        .def_property("vx", &PyGameEngine::get_vx, &PyGameEngine::set_vx)
        .def_property("vy", &PyGameEngine::get_vy, &PyGameEngine::set_vy)
        .def_property("gravity", &PyGameEngine::get_gravity, &PyGameEngine::set_gravity)
        .def_property("radius", &PyGameEngine::get_radius, &PyGameEngine::set_radius)
        .def_property("target_x", &PyGameEngine::get_target_x, &PyGameEngine::set_target_x)
        .def_property("target_y", &PyGameEngine::get_target_y, &PyGameEngine::set_target_y)
        .def_property("width", &PyGameEngine::get_width, &PyGameEngine::set_width)
        .def_property("height", &PyGameEngine::get_height, &PyGameEngine::set_height);
}
