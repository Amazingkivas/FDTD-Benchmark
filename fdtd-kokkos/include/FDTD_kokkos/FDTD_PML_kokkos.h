#pragma once

#include "FDTD_kokkos.h"


namespace FDTD_kokkos {

using Boundaries = std::pair<int, int>;

class FDTD_PML : public FDTD {
private:
    Field Exy, Exz, Eyx, Eyz, Ezx, Ezy;
    Field Bxy, Bxz, Byx, Byz, Bzx, Bzy;
    Field EsigmaX, EsigmaY, EsigmaZ;
    Field BsigmaX, BsigmaY, BsigmaZ;

    Boundaries size_i_main, size_j_main, size_k_main;
    Boundaries size_i_solid, size_j_solid, size_k_solid;
    Boundaries size_i_part_from_start, size_i_part_from_end,
        size_k_part_from_start, size_k_part_from_end,
        size_xy_lower_k_pml, size_xy_upper_k_pml,
        size_yz_lower_i_pml, size_yz_upper_i_pml,
        size_zx_lower_j_pml, size_zx_upper_j_pml;

    int pml_size_i, pml_size_j, pml_size_k;

    void update_B_PML(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k);
    void update_E_PML(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k);

public:
    FDTD_PML(Parameters _parameters, FP _dt, FP pml_percent);

    void update_fields() override;
};

}
