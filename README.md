# FDTD Benchmark

This repository contains independent implementations of the finite-difference time-domain (FDTD) method using OpenMP and Kokkos, as well as a separate Coarray Fortran implementation.

## Repository layout

- `fdtd-openmp/` — OpenMP implementation, its headers, tests, and sample.
- `fdtd-kokkos/` — Kokkos implementation, its headers, tests, and sample.
- `coarray/` — Coarray Fortran implementation.
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

## Build

Build Kokkos with the OpenMP execution space:

```bash
cmake -S 3rdparty/kokkos -B 3rdparty/kokkos/build \
    -DKokkos_ENABLE_OPENMP=ON
cmake --build 3rdparty/kokkos/build --parallel
```

Configure and build both implementations from the repository root:

```bash
cmake -S . -B build
cmake --build build --parallel
```

The build produces separate test executables and samples:

- `build/fdtd-openmp/test_FDTD_method`
- `build/fdtd-kokkos/test_FDTD_method_kokkos`
- `build/bin/sample`
- `build/bin/kokkos_sample`

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

## Run samples

```bash
./build/bin/sample 512 25
./build/bin/kokkos_sample 512 25
```

## Visualization

![FDTD animation](animations/animation_Ez.gif)
