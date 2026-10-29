cmake_minimum_required(VERSION 3.16)

project(FDTD_Benchmark LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(OpenMP REQUIRED)
add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)

set(Kokkos_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kokkos/build" CACHE PATH "Kokkos package directory")
find_package(Kokkos REQUIRED)

add_subdirectory(fdtd-openmp)
add_subdirectory(fdtd-kokkos)
