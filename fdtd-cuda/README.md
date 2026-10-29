# FDTD CUDA

CUDA implementation of the FDTD method for Maxwell's equations using double precision.

## Requirements

- CUDA Toolkit 13.0 or compatible
- CMake 3.18+
- NVIDIA GPU

## Build from the repository root

```bash
cmake -S . -B build-cuda \
    -DFDTD_BACKEND=CUDA \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-cuda --parallel
```

Architecture `80` targets NVIDIA A100. Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU.

## Run

```bash
./build-cuda/bin/run_cuda [grid_size] [iterations]
```

For a quick sanity check:

```bash
./build-cuda/bin/run_cuda 128 25
```
