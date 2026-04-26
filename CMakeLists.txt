cmake_minimum_required(VERSION 3.20)

project(metallic LANGUAGES CXX)

include(FetchContent)

find_package(Python COMPONENTS Development Interpreter REQUIRED)
set(PYBIND11_FINDPYTHON ON)

set(METALLIC_METAL_CPP_GIT_REPOSITORY
    "https://github.com/bkaradzic/metal-cpp.git"
    CACHE STRING "Repository used for the Metal C++ headers")
set(METALLIC_METAL_CPP_GIT_TAG
    "2948dd1ea03b11a41a079abe7d46e329d13af08e"
    CACHE STRING "Pinned metal-cpp commit")

FetchContent_Declare(
  metal_cpp
  GIT_REPOSITORY "${METALLIC_METAL_CPP_GIT_REPOSITORY}"
  GIT_TAG "${METALLIC_METAL_CPP_GIT_TAG}"
  GIT_SHALLOW FALSE)
FetchContent_MakeAvailable(metal_cpp)

add_library(metallic INTERFACE)
add_library(metallic::metallic ALIAS metallic)

target_compile_features(metallic INTERFACE cxx_std_17)
target_include_directories(metallic
                           INTERFACE "$<BUILD_INTERFACE:${metal_cpp_SOURCE_DIR}>")

if(APPLE)
  target_link_libraries(metallic INTERFACE "-framework Metal" "-framework Foundation"
                                           "-framework QuartzCore")
endif()

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
  GIT_TAG "v2.13.6")
FetchContent_MakeAvailable(pybind11)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.metal"
     METALLIC_KERNEL_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                              "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.metal")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/metal_kernels.hpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/generated/metal_kernels.hpp" @ONLY)

pybind11_add_module(ndarray_backend src/ndarray_backend.cpp)
target_include_directories(ndarray_backend
                           PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(ndarray_backend PRIVATE metallic::metallic)
