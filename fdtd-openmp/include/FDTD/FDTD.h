#pragma once

#include "shared.h"


namespace FDTD_openmp {

class FDTD {
protected:
    Parameters parameters;

    int Ni, Nj, Nk;
    double dx, dy, dz, dt;
    double cur_coef;
    double coef_E_dx, coef_E_dy, coef_E_dz;
    double coef_B_dx, coef_B_dy, coef_B_dz;
    int begin_main_i, begin_main_j, begin_main_k;
    int end_main_i, end_main_j, end_main_k;

    Field Jx, Jy, Jz;
    Field Ex, Ey, Ez;
    Field Bx, By, Bz;

    inline void applyPeriodicBoundary(int& i, const int& N) {
        int i_isMinusOne = (i < 0);
    
        int i_isNi = (i == N);
    
        i = (N - 1) * i_isMinusOne + i *
            !(i_isMinusOne || i_isNi);
    }

    void update_E();
    void update_B();
public:
    FDTD(Parameters _parameters, double _dt);

    Field& get_field(Component this_field);
    virtual void update_fields();
    void zeroed_currents();
};

}
