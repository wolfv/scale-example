cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(example_basic LANGUAGES CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(example_basic example.cu)

set_target_properties(example_basic PROPERTIES
    CUDA_STANDARD 17
)

install(TARGETS example_basic DESTINATION bin)
