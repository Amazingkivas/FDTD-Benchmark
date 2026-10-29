#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

#include "FDTD_cuda.h"
#include "Constants.h"

using namespace FDTD_cuda;
using namespace FDTD_struct;

__global__ void add_spherical_currents(
    FP* __restrict__ Jx, FP* __restrict__ Jy, FP* __restrict__ Jz,
    int Ni, int Nj, int Nk,
    int start_i, int start_j, int start_k,
    int max_i, int max_j, int max_k,
    FP time, FP T, FP Tx, FP Ty, FP Tz,
    FP dx, FP dy, FP dz
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + start_i;
    const int j = blockIdx.y * blockDim.y + threadIdx.y + start_j;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + start_k;

    if (i >= max_i || j >= max_j || k >= max_k) {
        return;
    }

    const size_t id = static_cast<size_t>(k) * Nj * Ni + static_cast<size_t>(j) * Ni + i;

    const FP x = static_cast<FP>(i) * dx;
    const FP y = static_cast<FP>(j) * dy;
    const FP z = static_cast<FP>(k) * dz;

    const FP cx = cos(2.0 * FDTD_const::PI * x / Tx);
    const FP cy = cos(2.0 * FDTD_const::PI * y / Ty);
    const FP cz = cos(2.0 * FDTD_const::PI * z / Tz);

    const FP value = sin(2.0 * FDTD_const::PI * time / T)
                   * (cx * cx) * (cy * cy) * (cz * cz);

    Jx[id] = value;
    Jy[id] = value;
    Jz[id] = value;
}

int main(int argc, char* argv[]) {
    int grid_size = 1024;
    int iterations = 25;

    if (argc >= 3) {
        grid_size = std::atoi(argv[1]);
        iterations = std::atoi(argv[2]);
    } else if (argc != 1) {
        std::cout << "ERROR: Incorrect number of parameters" << std::endl;
        return 1;
    }

    std::cout << "Running CUDA FDTD (double) with N=" << grid_size
              << ", iterations=" << iterations << std::endl;

    CurrentParameters current_parameters;
    current_parameters.period = 8;
    current_parameters.m = 4;
    current_parameters.dt = 0.2;
    current_parameters.iterations =
        static_cast<int>(static_cast<double>(current_parameters.period) / current_parameters.dt);
    current_parameters.period_x = current_parameters.period_y =
        current_parameters.period_z = current_parameters.m * FDTD_const::C;

    const FP spacing = FDTD_const::C;
    const FP boundary = static_cast<FP>(grid_size) / 2.0 * spacing;

    Parameters parameters{
        grid_size, grid_size, grid_size,
        -boundary, boundary,
        -boundary, boundary,
        -boundary, boundary,
        spacing, spacing, spacing
    };

    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::cout << "Using GPU: " << properties.name << std::endl;
    std::cout << "Compute Capability: " << properties.major << "." << properties.minor << std::endl;
    std::cout << "Global Memory: "
              << properties.totalGlobalMem / (1024 * 1024 * 1024.0) << " GB" << std::endl;

    FDTD solver(parameters, current_parameters.dt);

    const int current_time = std::min(current_parameters.iterations, iterations);

    const int start_i = static_cast<int>(std::floor(
        (-current_parameters.period_x / 4.0 - parameters.ax) / parameters.dx));
    const int start_j = static_cast<int>(std::floor(
        (-current_parameters.period_y / 4.0 - parameters.ay) / parameters.dy));
    const int start_k = static_cast<int>(std::floor(
        (-current_parameters.period_z / 4.0 - parameters.az) / parameters.dz));

    const int max_i = static_cast<int>(std::floor(
        (current_parameters.period_x / 4.0 - parameters.ax) / parameters.dx));
    const int max_j = static_cast<int>(std::floor(
        (current_parameters.period_y / 4.0 - parameters.ay) / parameters.dy));
    const int max_k = static_cast<int>(std::floor(
        (current_parameters.period_z / 4.0 - parameters.az) / parameters.dz));

    const int size_i = max_i - start_i;
    const int size_j = max_j - start_j;
    const int size_k = max_k - start_k;

    const dim3 threads(8, 8, 8);
    const dim3 blocks(
        (size_i + threads.x - 1) / threads.x,
        (size_j + threads.y - 1) / threads.y,
        (size_k + threads.z - 1) / threads.z
    );

    const auto start_time = std::chrono::high_resolution_clock::now();

    for (int time_index = 0; time_index < current_time; ++time_index) {
        const FP time = static_cast<FP>(time_index + 1) * current_parameters.dt;

        add_spherical_currents<<<blocks, threads>>>(
            solver.Jx, solver.Jy, solver.Jz,
            grid_size, grid_size, grid_size,
            start_i, start_j, start_k,
            max_i, max_j, max_k,
            time,
            static_cast<FP>(current_parameters.period),
            current_parameters.period_x, current_parameters.period_y, current_parameters.period_z,
            spacing, spacing, spacing
        );
        CUDA_CHECK(cudaGetLastError());

        solver.update_fields();
    }

    solver.zeroed_currents();
    for (int time_index = current_time; time_index < iterations; ++time_index) {
        solver.update_fields();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    const auto end_time = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Execution time: " << elapsed.count() << " s" << std::endl;
    std::cout << "Time per iteration: " << elapsed.count() / iterations << " s" << std::endl;

    std::cout << "CUDA results: \n" << std::endl;

    const int middle = parameters.Ni / 2;
    for (int j = parameters.Nj / 2 - 5; j < parameters.Nj / 2 + 5; ++j) {
        for (int k = parameters.Nk / 2 - 5; k < parameters.Nk / 2 + 5; ++k) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(5)
                      << solver.get_Ex(k, j, middle);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
