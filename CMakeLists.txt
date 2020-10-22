cmake_minimum_required(VERSION 3.16)
project(corsika CXX)

add_subdirectory(extern/pybind11)
pybind11_add_module(corsika main.cpp)
