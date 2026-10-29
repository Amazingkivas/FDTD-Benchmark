# FDTD Benchmark

This repository contains independent implementations of the finite-difference time-domain (FDTD) method for Maxwell's equations.

## Repository layout

- `fdtd-openmp/` — OpenMP implementation, unit tests, and sample.
- `fdtd-kokkos/` — Kokkos implementation, unit tests, and sample.
- `fdtd-sycl/` — SYCL implementation and executable.
- `fdtd-cuda/` — CUDA implementation and executable.
- `fdtd-coarray/` — Coarray Fortran implementation.
- `scripts/` — Python visualization scripts.
- `3rdparty/` — GoogleTest and Kokkos git submodules.
- `animations/` — visualization assets.

## Clone

Clone the repository together with its submodules:

```bash
git clone --recurse-submodules https://github.com/Amazingkivas/FDTD-Benchmark.git
cd FDTD-Benchmark
```

If the repository was cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

## Select a backend

The root CMake project builds exactly one backend configuration at a time. The available values of `FDTD_BACKEND` are:

| Value | Implementations built | Toolchain |
|---|---|---|
| `OPENMP_KOKKOS` | OpenMP and Kokkos | C++ compiler, OpenMP, Kokkos |
| `SYCL` | SYCL only | Intel oneAPI DPC++ (`icpx`) |
| `CUDA` | CUDA only | C++ compiler and NVIDIA CUDA Toolkit |

Each backend should use its own build directory. This avoids reusing a CMake cache configured for a different compiler or GPU toolchain.

### OpenMP and Kokkos

Build Kokkos with the OpenMP execution space:

```bash
cmake -S 3rdparty/kokkos -B 3rdparty/kokkos/build \
    -DKokkos_ENABLE_OPENMP=ON
cmake --build 3rdparty/kokkos/build --parallel
```

Configure and build the OpenMP and Kokkos implementations:

```bash
cmake -S . -B build-openmp-kokkos -DFDTD_BACKEND=OPENMP_KOKKOS
cmake --build build-openmp-kokkos --parallel
```

Run unit tests and samples:

```bash
ctest --test-dir build-openmp-kokkos --output-on-failure
./build-openmp-kokkos/bin/sample 512 25
./build-openmp-kokkos/bin/kokkos_sample 512 25
```

### SYCL

Configure with the Intel oneAPI DPC++ compiler:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl \
    -DFDTD_BACKEND=SYCL \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-sycl --parallel
```

Run:

```bash
./build-sycl/bin/run_sycl [grid_size] [iterations]
```

The implementation selects a GPU when available. Set `FDTD_SYCL_DEVICE=cpu`, `gpu`, or `default` to choose a device explicitly.

### CUDA

Configure and build with the NVIDIA CUDA Toolkit:

```bash
cmake -S . -B build-cuda \
    -DFDTD_BACKEND=CUDA \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-cuda --parallel
```

Architecture `80` targets NVIDIA A100. Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU.

Run:

```bash
./build-cuda/bin/run_cuda [grid_size] [iterations]
```

### Coarray Fortran

The Coarray implementation is built separately from the CMake backend selection. For the Intel oneAPI compiler:

```bash
source /opt/intel/oneapi/setvars.sh
ifx -coarray fdtd-coarray/fdtd.F90 -o fdtd
export FOR_COARRAY_NUM_IMAGES=4
./fdtd
```

## Visualization

![FDTD animation](animations/animation_Ez.gif)
