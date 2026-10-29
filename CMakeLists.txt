cmake_minimum_required(VERSION 3.18)

project(FDTD_Benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FDTD_BACKEND "OPENMP_KOKKOS" CACHE STRING
    "Backend to build: OPENMP_KOKKOS, SYCL, or CUDA")
set_property(CACHE FDTD_BACKEND PROPERTY STRINGS OPENMP_KOKKOS SYCL CUDA)

if(FDTD_BACKEND STREQUAL "OPENMP_KOKKOS")
    enable_testing()

    find_package(OpenMP REQUIRED)
    add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)

    set(Kokkos_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/kokkos/build"
        CACHE PATH "Kokkos package directory")
    find_package(Kokkos REQUIRED)

    add_subdirectory(fdtd-openmp)
    add_subdirectory(fdtd-kokkos)

elseif(FDTD_BACKEND STREQUAL "SYCL")
    add_subdirectory(fdtd-sycl)

elseif(FDTD_BACKEND STREQUAL "CUDA")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA GPU architectures")
    endif()
    enable_language(CUDA)
    add_subdirectory(fdtd-cuda)

else()
    message(FATAL_ERROR
        "Unknown FDTD_BACKEND='${FDTD_BACKEND}'. "
        "Choose OPENMP_KOKKOS, SYCL, or CUDA.")
endif()
