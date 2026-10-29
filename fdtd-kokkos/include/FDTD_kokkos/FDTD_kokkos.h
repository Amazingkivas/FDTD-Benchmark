#pragma once

#include "kokkos_functors.h"


namespace FDTD_kokkos {
class FDTD
{
protected:
    Field Jx, Jy, Jz;
    Field Ex, Ey, Ez;
    Field Bx, By, Bz;

    int size_i_main[2];
    int size_j_main[2];
    int size_k_main[2];

    int Ni, Nj, Nk;
    FP dx, dy, dz, dt;
    FP current_coef;
    FP coef_Ex, coef_Ey, coef_Ez;
    FP coef_Bx, coef_By, coef_Bz;
    int begin_main_i, begin_main_j, begin_main_k;
    int end_main_i, end_main_j, end_main_k;

    Parameters parameters;

public:
    FDTD(Parameters _parameters, FP _dt);

    Field& get_field(Component);

    virtual void update_fields();

    void zeroed_currents();
};

}
