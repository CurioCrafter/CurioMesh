cmake_minimum_required(VERSION 3.20)
project(curio_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(curio_core MODULE
  pybind_module.cpp
  curio_core.cpp
)

target_include_directories(curio_core PRIVATE ${Python3_INCLUDE_DIRS})

if (MSVC)
  target_compile_options(curio_core PRIVATE /W4)
else()
  target_compile_options(curio_core PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(curio_core PRIVATE m)
endif()


