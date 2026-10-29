#pragma once

#include "FP.h"

namespace FDTD_struct {

struct Parameters {
    int Ni, Nj, Nk;
    FP ax, bx;
    FP ay, by;
    FP az, bz;
    FP dx, dy, dz;
};

struct CurrentParameters {
    int period;
    int m;
    FP dt;
    int iterations;
    FP period_x;
    FP period_y;
    FP period_z;
};

}
