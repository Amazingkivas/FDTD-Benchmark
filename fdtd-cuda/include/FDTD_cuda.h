#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "FP.h"
#include "Enums.h"
#include "Structures.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\\n",              \
                         cudaGetErrorName(err__), __FILE__, __LINE__,           \
                         cudaGetErrorString(err__));                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

namespace FDTD_cuda {

class FDTD {
private:
    FDTD_struct::Parameters params;

    int Ni, Nj, Nk;
    FP dx, dy, dz, dt;

    FP coef_E_dx, coef_E_dy, coef_E_dz;
    FP coef_B_dx, coef_B_dy, coef_B_dz;
    FP cur_coef;

    FP *Ex, *Ey, *Ez;
    FP *Bx, *By, *Bz;

public:
    FP *Jx, *Jy, *Jz;

    FDTD(FDTD_struct::Parameters parameters, FP time_step);
    ~FDTD();

    FDTD(const FDTD&) = delete;
    FDTD& operator=(const FDTD&) = delete;

    void zero_fields();
    void zeroed_currents();
    void update_fields();

    FP get_Ex(int i, int j, int k);

    int size() const { return Ni * Nj * Nk; }
};

}
