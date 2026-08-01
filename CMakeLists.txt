cmake_minimum_required(VERSION 3.14)
project(engine_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(engine_core
    src/engine.hpp
    src/engine.cpp
    src/bridge.cpp
)

target_include_directories(engine_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

