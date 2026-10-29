#include "FDTD_cuda.h"
#include "Constants.h"

#include <climits>
#include <cmath>
#include <cstdio>

namespace FDTD_cuda {

constexpr int BLOCK_SIZE_X = 32;
constexpr int BLOCK_SIZE_Y = 4;
constexpr int BLOCK_SIZE_Z = 4;

__device__ __forceinline__ int periodic(int i, int N) {
    const int is_minus_one = (i < 0);
    const int is_N = (i == N);
    return (N - 1) * is_minus_one + i * !(is_minus_one || is_N);
}

__global__ void update_B_opt(
    const FP* __restrict__ Ex, const FP* __restrict__ Ey, const FP* __restrict__ Ez,
    FP* __restrict__ Bx, FP* __restrict__ By, FP* __restrict__ Bz,
    int Ni, int Nj, int Nk,
    FP cdx, FP cdy, FP cdz
) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= Ni || j >= Nj || k >= Nk) return;

    const int NiNj = Ni * Nj;
    const int id = k * NiNj + j * Ni + i;

    const int ip = periodic(i + 1, Ni);
    const int jp = periodic(j + 1, Nj);
    const int kp = periodic(k + 1, Nk);

    const int id_ip = k * NiNj + j * Ni + ip;
    const int id_jp = k * NiNj + jp * Ni + i;
    const int id_kp = kp * NiNj + j * Ni + i;

    const FP Ex_val = Ex[id];
    const FP Ey_val = Ey[id];
    const FP Ez_val = Ez[id];

    const FP Ey_kp = Ey[id_kp];
    const FP Ez_jp = Ez[id_jp];
    const FP Ez_ip = Ez[id_ip];
    const FP Ex_kp = Ex[id_kp];
    const FP Ex_jp = Ex[id_jp];
    const FP Ey_ip = Ey[id_ip];

    Bx[id] = fma(cdz, Ey_kp - Ey_val, fma(-cdy, Ez_jp - Ez_val, Bx[id]));
    By[id] = fma(cdx, Ez_ip - Ez_val, fma(-cdz, Ex_kp - Ex_val, By[id]));
    Bz[id] = fma(cdy, Ex_jp - Ex_val, fma(-cdx, Ey_ip - Ey_val, Bz[id]));
}

__global__ void update_E_opt(
    FP* __restrict__ Ex, FP* __restrict__ Ey, FP* __restrict__ Ez,
    const FP* __restrict__ Bx, const FP* __restrict__ By, const FP* __restrict__ Bz,
    const FP* __restrict__ Jx, const FP* __restrict__ Jy, const FP* __restrict__ Jz,
    int Ni, int Nj, int Nk,
    FP current, FP cdx, FP cdy, FP cdz
) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= Ni || j >= Nj || k >= Nk) return;

    const int NiNj = Ni * Nj;
    const int id = k * NiNj + j * Ni + i;

    const int im = periodic(i - 1, Ni);
    const int jm = periodic(j - 1, Nj);
    const int km = periodic(k - 1, Nk);

    const int id_im = k * NiNj + j * Ni + im;
    const int id_jm = k * NiNj + jm * Ni + i;
    const int id_km = km * NiNj + j * Ni + i;

    const FP Bx_val = Bx[id];
    const FP By_val = By[id];
    const FP Bz_val = Bz[id];

    const FP Bz_jm = Bz[id_jm];
    const FP By_km = By[id_km];
    const FP Bx_km = Bx[id_km];
    const FP Bz_im = Bz[id_im];
    const FP By_im = By[id_im];
    const FP Bx_jm = Bx[id_jm];

    const FP Jx_val = Jx[id];
    const FP Jy_val = Jy[id];
    const FP Jz_val = Jz[id];

    Ex[id] = fma(current, Jx_val,
                 fma(cdy, Bz_val - Bz_jm,
                     fma(-cdz, By_val - By_km, Ex[id])));

    Ey[id] = fma(current, Jy_val,
                 fma(cdz, Bx_val - Bx_km,
                     fma(-cdx, Bz_val - Bz_im, Ey[id])));

    Ez[id] = fma(current, Jz_val,
                 fma(cdx, By_val - By_im,
                     fma(-cdy, Bx_val - Bx_jm, Ez[id])));
}

FDTD::FDTD(FDTD_struct::Parameters parameters, FP time_step)
    : params(parameters), dt(time_step) {
    Ni = parameters.Ni;
    Nj = parameters.Nj;
    Nk = parameters.Nk;
    dx = parameters.dx;
    dy = parameters.dy;
    dz = parameters.dz;

    const size_t cells = static_cast<size_t>(Ni) * static_cast<size_t>(Nj) * static_cast<size_t>(Nk);
    if (cells >= static_cast<size_t>(INT_MAX)) {
        std::fprintf(stderr, "Grid too large for 32-bit indexing\n");
        std::exit(1);
    }

    const size_t bytes = cells * sizeof(FP);
    size_t free_memory = 0;
    size_t total_memory = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    if (bytes * 9 > free_memory) {
        std::fprintf(stderr,
            "ERROR: Not enough GPU memory. Required: %.2f GiB, available: %.2f GiB\n",
            bytes * 9.0 / (1024.0 * 1024 * 1024),
            free_memory / (1024.0 * 1024 * 1024));
        std::exit(1);
    }

    CUDA_CHECK(cudaMalloc(&Ex, bytes));
    CUDA_CHECK(cudaMalloc(&Ey, bytes));
    CUDA_CHECK(cudaMalloc(&Ez, bytes));
    CUDA_CHECK(cudaMalloc(&Bx, bytes));
    CUDA_CHECK(cudaMalloc(&By, bytes));
    CUDA_CHECK(cudaMalloc(&Bz, bytes));
    CUDA_CHECK(cudaMalloc(&Jx, bytes));
    CUDA_CHECK(cudaMalloc(&Jy, bytes));
    CUDA_CHECK(cudaMalloc(&Jz, bytes));

    const FP cdt = FDTD_const::C * dt;
    coef_E_dx = cdt / dx;
    coef_E_dy = cdt / dy;
    coef_E_dz = cdt / dz;

    coef_B_dx = cdt / (2.0 * dx);
    coef_B_dy = cdt / (2.0 * dy);
    coef_B_dz = cdt / (2.0 * dz);

    cur_coef = -4.0 * FDTD_const::PI * dt;

    zero_fields();
    zeroed_currents();
    CUDA_CHECK(cudaDeviceSynchronize());
}

FDTD::~FDTD() {
    cudaFree(Ex);
    cudaFree(Ey);
    cudaFree(Ez);
    cudaFree(Bx);
    cudaFree(By);
    cudaFree(Bz);
    cudaFree(Jx);
    cudaFree(Jy);
    cudaFree(Jz);
}

void FDTD::zero_fields() {
    const size_t bytes = static_cast<size_t>(Ni) * Nj * Nk * sizeof(FP);
    CUDA_CHECK(cudaMemset(Ex, 0, bytes));
    CUDA_CHECK(cudaMemset(Ey, 0, bytes));
    CUDA_CHECK(cudaMemset(Ez, 0, bytes));
    CUDA_CHECK(cudaMemset(Bx, 0, bytes));
    CUDA_CHECK(cudaMemset(By, 0, bytes));
    CUDA_CHECK(cudaMemset(Bz, 0, bytes));
}

void FDTD::zeroed_currents() {
    const size_t bytes = static_cast<size_t>(Ni) * Nj * Nk * sizeof(FP);
    CUDA_CHECK(cudaMemset(Jx, 0, bytes));
    CUDA_CHECK(cudaMemset(Jy, 0, bytes));
    CUDA_CHECK(cudaMemset(Jz, 0, bytes));
}

void FDTD::update_fields() {
    const dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    const dim3 grid_size(
        (Ni + block_size.x - 1) / block_size.x,
        (Nj + block_size.y - 1) / block_size.y,
        (Nk + block_size.z - 1) / block_size.z
    );

    update_B_opt<<<grid_size, block_size>>>(
        Ex, Ey, Ez, Bx, By, Bz,
        Ni, Nj, Nk,
        coef_B_dx, coef_B_dy, coef_B_dz
    );

    update_E_opt<<<grid_size, block_size>>>(
        Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz,
        Ni, Nj, Nk,
        cur_coef, coef_E_dx, coef_E_dy, coef_E_dz
    );

    update_B_opt<<<grid_size, block_size>>>(
        Ex, Ey, Ez, Bx, By, Bz,
        Ni, Nj, Nk,
        coef_B_dx, coef_B_dy, coef_B_dz
    );

    CUDA_CHECK(cudaGetLastError());
}

FP FDTD::get_Ex(int i, int j, int k) {
    FP value;
    const size_t id = static_cast<size_t>(k) * Nj * Ni + static_cast<size_t>(j) * Ni + i;
    CUDA_CHECK(cudaMemcpy(&value, Ex + id, sizeof(FP), cudaMemcpyDeviceToHost));
    return value;
}

}
