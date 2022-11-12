cmake_minimum_required(VERSION 3.20)
project(paralleli CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

add_executable(paralleli main.cu)

include_directories(include "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

set_target_properties(paralleli PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CMAKE_CUDA_ARCHITECTURES "52;61;86")
