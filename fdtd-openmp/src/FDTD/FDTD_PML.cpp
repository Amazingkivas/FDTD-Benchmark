#include "FDTD_PML.h"

inline void FDTD_openmp::FDTD_PML::set_sigma_x(Boundaries bounds_i,
    Boundaries bounds_j, Boundaries bounds_k,
    FP SGm, Function dist) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = bounds_k.first; k < bounds_k.second; k++) {
        for (int j = bounds_j.first; j < bounds_j.second; j++) {
            #pragma omp simd
            for (int i = bounds_i.first; i < bounds_i.second; i++) {
                int index = i + j * Ni + k * Ni * Nj;

                EsigmaX[index] = SGm *
                    std::pow((static_cast<FP>(dist(i, j, k))) /
                    static_cast<FP>(pml_size_i), FDTD_const::N);
                BsigmaX[index] = SGm *
                    std::pow((static_cast<FP>(dist(i, j, k))) /
                    static_cast<FP>(pml_size_i), FDTD_const::N);
            }
        }
    }
}
inline void FDTD_openmp::FDTD_PML::set_sigma_y(Boundaries bounds_i,
    Boundaries bounds_j, Boundaries bounds_k,
    FP SGm, Function dist) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = bounds_k.first; k < bounds_k.second; k++) {
        for (int j = bounds_j.first; j < bounds_j.second; j++) {
            #pragma omp simd
            for (int i = bounds_i.first; i < bounds_i.second; i++) {
                int index = i + j * Ni + k * Ni * Nj;

                EsigmaY[index] = SGm *
                    std::pow((static_cast<FP>(dist(i, j, k))) /
                    static_cast<FP>(pml_size_j), FDTD_const::N);
                BsigmaY[index] = SGm *
                    std::pow((static_cast<FP>(dist(i, j, k))) /
                    static_cast<FP>(pml_size_j), FDTD_const::N);
            }
        }
    }
}
inline void FDTD_openmp::FDTD_PML::set_sigma_z(Boundaries bounds_i, Boundaries bounds_j, Boundaries bounds_k,
    FP SGm, Function dist) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = bounds_k.first; k < bounds_k.second; k++) {
        for (int j = bounds_j.first; j < bounds_j.second; j++) {
            #pragma omp simd
            for (int i = bounds_i.first; i < bounds_i.second; i++) {
                int index = i + j * Ni + k * Ni * Nj;

                EsigmaZ[index] = SGm *
                    std::pow((static_cast<FP>(dist(i, j, k))) /
                    static_cast<FP>(pml_size_k), FDTD_const::N);
                BsigmaZ[index] = SGm *
                    std::pow((static_cast<FP>(dist(i, j, k))) /
                    static_cast<FP>(pml_size_k), FDTD_const::N);
            }
        }
    }
}

inline FP FDTD_openmp::FDTD_PML::PMLcoef(FP sigma) const {
    return std::exp(-sigma * this->dt * FDTD_const::C);
}

inline void FDTD_openmp::FDTD_PML::update_E_PML(Boundaries bounds_i,
    Boundaries bounds_j, Boundaries bounds_k) {
    FP PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

    int k_start = bounds_k.first;
    int k_end = bounds_k.second;
    int j_start = bounds_j.first;
    int j_end = bounds_j.second;
    int i_start = bounds_i.first;
    int i_end = bounds_i.second;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = k_start; k < k_end; k++)  {
        for (int j = j_start; j < j_end; j++) {
            int index_kj = j * Ni + k * Ni * Nj;
            int j_pred_kj = j - 1;
            int k_pred_k = k - 1;
            applyPeriodicBoundary(j_pred_kj, Nj);
            applyPeriodicBoundary(k_pred_k, Nk);
            k_pred_k = k_pred_k * Ni * Nj;
            j_pred_kj = j_pred_kj * Ni + k * Ni * Nj;
            int k_pred_kj = j * Ni + k_pred_k;
            #pragma omp simd
            for (int i = i_start; i < i_end; i++) {
                int index = i + index_kj;
                int i_pred = i - 1;
                applyPeriodicBoundary(i_pred, Ni);
                i_pred = i_pred + index_kj;
                int j_pred = i + j_pred_kj;
                int k_pred = i + k_pred_kj;

                if (EsigmaX[index] != 0.0)
                    PMLcoef2_x = (1.0 - PMLcoef(EsigmaX[index])) / (EsigmaX[index] * dx);
                else
                    PMLcoef2_x = this->coef_E_dx;

                if (EsigmaY[index] != 0.0)
                    PMLcoef2_y = (1.0 - PMLcoef(EsigmaY[index])) / (EsigmaY[index] * dy);
                else
                    PMLcoef2_y = this->coef_E_dy;

                if (EsigmaZ[index] != 0.0)
                    PMLcoef2_z = (1.0 - PMLcoef(EsigmaZ[index])) / (EsigmaZ[index] * dz);
                else
                    PMLcoef2_z = this->coef_E_dz;

                Eyx[index] = Eyx[index] * PMLcoef(EsigmaX[index]) -
                    PMLcoef2_x * (this->Bz[index] - this->Bz[i_pred]);
                Ezx[index] = Ezx[index] * PMLcoef(EsigmaX[index]) +
                    PMLcoef2_x * (this->By[index] - this->By[i_pred]);

                Exy[index] = Exy[index] * PMLcoef(EsigmaY[index]) +
                    PMLcoef2_y * (this->Bz[index] - this->Bz[j_pred]);
                Ezy[index] = Ezy[index] * PMLcoef(EsigmaY[index]) -
                    PMLcoef2_y * (this->Bx[index] - this->Bx[j_pred]);

                Exz[index] = Exz[index] * PMLcoef(EsigmaZ[index]) -
                    PMLcoef2_z * (this->By[index] - this->By[k_pred]);
                Eyz[index] = Eyz[index] * PMLcoef(EsigmaZ[index]) +
                    PMLcoef2_z * (this->Bx[index] - this->Bx[k_pred]);

                this->Ex[index] = Exz[index] + Exy[index];
                this->Ey[index] = Eyx[index] + Eyz[index];
                this->Ez[index] = Ezy[index] + Ezx[index];
            }
        }
    }
}

inline void FDTD_openmp::FDTD_PML::update_B_PML(Boundaries bounds_i,
    Boundaries bounds_j, Boundaries bounds_k) {
    FP PMLcoef2_x, PMLcoef2_y, PMLcoef2_z;

    int k_start = bounds_k.first;
    int k_end = bounds_k.second;
    int j_start = bounds_j.first;
    int j_end = bounds_j.second;
    int i_start = bounds_i.first;
    int i_end = bounds_i.second;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = k_start; k < k_end; k++) {
        for (int j = j_start; j < j_end; j++) {
            int index_kj = j * Ni + k * Ni * Nj;
            int j_next_kj = j + 1;
            int k_next_k = k + 1;
            applyPeriodicBoundary(j_next_kj, Nj);
            applyPeriodicBoundary(k_next_k, Nk);
            k_next_k = k_next_k * Ni * Nj;
            j_next_kj = j_next_kj * Ni + k * Ni * Nj;
            int k_next_kj = j * Ni + k_next_k;
            #pragma omp simd
            for (int i = i_start; i < i_end; i++) {
                int index = i + index_kj;
                int i_next = i + 1;
                applyPeriodicBoundary(i_next, Ni);
                i_next = i_next + index_kj;
                int j_next = i + j_next_kj;
                int k_next = i + k_next_kj;

                if (BsigmaX[index] != 0.0)
                    PMLcoef2_x = (1.0 - PMLcoef(BsigmaX[index])) / (BsigmaX[index] * dx);
                else
                    PMLcoef2_x = this->coef_E_dx;

                if (BsigmaY[index] != 0.0)
                    PMLcoef2_y = (1.0 - PMLcoef(BsigmaY[index])) / (BsigmaY[index] * dy);
                else
                    PMLcoef2_y = this->coef_E_dy;

                if (BsigmaZ[index] != 0.0)
                    PMLcoef2_z = (1.0 - PMLcoef(BsigmaZ[index])) / (BsigmaZ[index] * dz);
                else
                    PMLcoef2_z = this->coef_E_dz;

                Byx[index] = Byx[index] * PMLcoef(BsigmaX[index]) +
                    PMLcoef2_x * (this->Ez[i_next] - this->Ez[index]);
                Bzx[index] = Bzx[index] * PMLcoef(BsigmaX[index]) -
                    PMLcoef2_x * (this->Ey[i_next] - this->Ey[index]);

                Bxy[index] = Bxy[index] * PMLcoef(BsigmaY[index]) -
                    PMLcoef2_y * (this->Ez[j_next] - this->Ez[index]);
                Bzy[index] = Bzy[index] * PMLcoef(BsigmaY[index]) +
                    PMLcoef2_y * (this->Ex[j_next] - this->Ex[index]);

                Bxz[index] = Bxz[index] * PMLcoef(BsigmaZ[index]) +
                    PMLcoef2_z * (this->Ey[k_next] - this->Ey[index]);
                Byz[index] = Byz[index] * PMLcoef(BsigmaZ[index]) -
                    PMLcoef2_z * (this->Ex[k_next] - this->Ex[index]);

                this->Bx[index] = Bxy[index] + Bxz[index];
                this->By[index] = Byz[index] + Byx[index];
                this->Bz[index] = Bzx[index] + Bzy[index];
            }
        }
    }
}

FDTD_openmp::FDTD_PML::FDTD_PML(Parameters _parameters, FP _dt, FP pml_percent) :
    FDTD(_parameters, _dt) {
    const int size = _parameters.Ni * _parameters.Nj * _parameters.Nk;

    Exy = Field(size);
    Exz = Field(size);
    Eyx = Field(size);
    Eyz = Field(size);
    Ezx = Field(size);
    Ezy = Field(size);

    Bxy = Field(size);
    Bxz = Field(size);
    Byx = Field(size);
    Byz = Field(size);
    Bzx = Field(size);
    Bzy = Field(size);

    EsigmaX = Field(size);
    EsigmaY = Field(size);
    EsigmaZ = Field(size);
    BsigmaX = Field(size);
    BsigmaY = Field(size);
    BsigmaZ = Field(size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        Exy[i] = 0.0;
        Exz[i] = 0.0;
        Eyx[i] = 0.0;
        Eyz[i] = 0.0;
        Ezx[i] = 0.0;
        Ezy[i] = 0.0;
    
        Bxy[i] = 0.0;
        Bxz[i] = 0.0;
        Byx[i] = 0.0;
        Byz[i] = 0.0;
        Bzx[i] = 0.0;
        Bzy[i] = 0.0;
        
        EsigmaX[i] = 0.0;
        EsigmaY[i] = 0.0;
        EsigmaZ[i] = 0.0;
        BsigmaX[i] = 0.0;
        BsigmaY[i] = 0.0;
        BsigmaZ[i] = 0.0;
    }

    pml_size_i = static_cast<int>(static_cast<FP>(_parameters.Ni) * pml_percent);
    pml_size_j = static_cast<int>(static_cast<FP>(_parameters.Nj) * pml_percent);
    pml_size_k = static_cast<int>(static_cast<FP>(_parameters.Nk) * pml_percent);

    // Defining areas of computation
    // ======================================================================
    size_i_main = { pml_size_i, _parameters.Ni - pml_size_i };
    size_j_main = { pml_size_j, _parameters.Nj - pml_size_j };
    size_k_main = { pml_size_k, _parameters.Nk - pml_size_k };

    this->begin_main_i = size_i_main.first;
    this->begin_main_j = size_j_main.first; 
    this->begin_main_k = size_k_main.first;
    this->end_main_i = size_i_main.second;
    this->end_main_j = size_j_main.second;
    this->end_main_k = size_k_main.second;

    size_i_solid = { 0, _parameters.Ni };
    size_j_solid = { 0, _parameters.Nj };
    size_k_solid = { 0, _parameters.Nk };

    size_i_part_from_start = { 0, _parameters.Ni - pml_size_i };
    size_i_part_from_end = { pml_size_i, _parameters.Ni };

    size_k_part_from_start = { 0, _parameters.Nk - pml_size_k };
    size_k_part_from_end = { pml_size_k, _parameters.Nk };

    size_xy_lower_k_pml = { 0, pml_size_k };
    size_xy_upper_k_pml = { _parameters.Nk - pml_size_k, _parameters.Nk };

    size_yz_lower_i_pml = { 0, pml_size_i };
    size_yz_upper_i_pml = { _parameters.Ni - pml_size_i, _parameters.Ni };

    size_zx_lower_j_pml = { 0, pml_size_j };
    size_zx_upper_j_pml = { _parameters.Nj - pml_size_j, _parameters.Nj };
    // ======================================================================

    // Definition of functions for calculating the distance to the interface
    // ======================================================================
    Function calc_distant_i_up = [=](int i, int j, int k) {
        return i + 1 + pml_size_i - _parameters.Ni;
    };
    Function calc_distant_j_up = [=](int i, int j, int k) {
        return j + 1 + pml_size_j - _parameters.Nj;
    };
    Function calc_distant_k_up = [=](int i, int j, int k) {
        return k + 1 + pml_size_k - _parameters.Nk;
    };

    Function calc_distant_i_low = [=](int i, int j, int k) {
        return pml_size_i - i;
    };
    Function calc_distant_j_low = [=](int i, int j, int k) {
        return pml_size_j - j;
    };
    Function calc_distant_k_low = [=](int i, int j, int k) {
        return pml_size_k - k;
    };
    // ======================================================================

    // Calculation of maximum permittivity and permeability
    // ======================================================================
    FP SGm_x = -(FDTD_const::N + 1.0) / 2.0 * std::log(FDTD_const::R)
        / (static_cast<FP>(pml_size_i) * _parameters.dx);
    FP SGm_y = -(FDTD_const::N + 1.0) / 2.0 * std::log(FDTD_const::R)
        / (static_cast<FP>(pml_size_j) * _parameters.dy);
    FP SGm_z = -(FDTD_const::N + 1.0) / 2.0 * std::log(FDTD_const::R)
        / (static_cast<FP>(pml_size_k) * _parameters.dz);
    // ======================================================================

    // Calculation of permittivity and permeability in the cells
    // ======================================================================
    set_sigma_z(size_i_solid, size_j_solid, size_xy_lower_k_pml,
        SGm_z, calc_distant_k_low);
    set_sigma_y(size_i_solid, size_zx_lower_j_pml, size_k_solid,
        SGm_y, calc_distant_j_low);
    set_sigma_x(size_yz_lower_i_pml, size_j_solid, size_k_solid,
        SGm_x, calc_distant_i_low);

    set_sigma_z(size_i_solid, size_j_solid, size_xy_upper_k_pml,
        SGm_z, calc_distant_k_up);
    set_sigma_y(size_i_solid, size_zx_upper_j_pml, size_k_solid,
        SGm_y, calc_distant_j_up);
    set_sigma_x(size_yz_upper_i_pml, size_j_solid, size_k_solid,
        SGm_x, calc_distant_i_up);

    // ======================================================================
}

void FDTD_openmp::FDTD_PML::update_fields() {
    this->update_B();

    update_B_PML(size_i_part_from_start, size_j_solid, size_xy_lower_k_pml);
    update_B_PML(size_i_main, size_zx_lower_j_pml, size_k_main);
    update_B_PML(size_yz_lower_i_pml, size_j_solid, size_k_part_from_end);

    update_B_PML(size_i_part_from_end, size_j_solid, size_xy_upper_k_pml);
    update_B_PML(size_i_main, size_zx_upper_j_pml, size_k_main);
    update_B_PML(size_yz_upper_i_pml, size_j_solid, size_k_part_from_start);

    this->update_E();

    update_E_PML(size_i_part_from_start, size_j_solid, size_xy_lower_k_pml);
    update_E_PML(size_i_main, size_zx_lower_j_pml, size_k_main);
    update_E_PML(size_yz_lower_i_pml, size_j_solid, size_k_part_from_end);

    update_E_PML(size_i_part_from_end, size_j_solid, size_xy_upper_k_pml);
    update_E_PML(size_i_main, size_zx_upper_j_pml, size_k_main);
    update_E_PML(size_yz_upper_i_pml, size_j_solid, size_k_part_from_start);

    this->update_B();
}
