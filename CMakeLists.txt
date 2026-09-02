cmake_minimum_required(VERSION 3.14)
project(engine_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(engine_core
    src/engine.hpp
    src/engine.cpp
    src/bridge.cpp
    src/raster3d.hpp
    src/raster3d.cpp
    src/raster3d_bind.cpp
    src/voxelmesh.hpp
    src/voxelmesh.cpp
    src/voxelmesh_bind.cpp
    src/gridops.hpp
    src/gridops.cpp
    src/gridops_bind.cpp
)

target_include_directories(engine_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

