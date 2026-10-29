#include "FDTD.h"

FDTD_openmp::FDTD::FDTD(Parameters _parameters, FP _dt)
    : parameters(_parameters), dt(_dt) {
    if (parameters.Ni <= 0 || parameters.Nj <= 0 || parameters.Nk <= 0 || dt <= 0) {
        throw std::invalid_argument("ERROR: invalid parameters");
    }

    const int size = parameters.Nk * parameters.Nj * parameters.Ni;

    Jx = Field(size);
    Jy = Field(size);
    Jz = Field(size);
    Ex = Field(size);
    Ey = Field(size);
    Ez = Field(size);
    Bx = Field(size);
    By = Field(size);
    Bz = Field(size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
       Jx[i] = 0.0;
       Jy[i] = 0.0;
       Jz[i] = 0.0;
       Ex[i] = 0.0;
       Ey[i] = 0.0;
       Ez[i] = 0.0;
       Bx[i] = 0.0;
       By[i] = 0.0;
       Bz[i] = 0.0;
    }

    Ni = parameters.Ni;
    Nj = parameters.Nj;
    Nk = parameters.Nk;

    dx = parameters.dx;
    dy = parameters.dy;
    dz = parameters.dz;
    dt = _dt;

    const FP cdt = FDTD_const::C * dt;

    coef_E_dx = cdt / dx;
    coef_E_dy = cdt / dy;
    coef_E_dz = cdt / dz;

    coef_B_dx = cdt / (2.0 * dx);
    coef_B_dy = cdt / (2.0 * dy);
    coef_B_dz = cdt / (2.0 * dz);

    cur_coef = -4.0 * FDTD_const::PI * dt;

    begin_main_i = 0;
    begin_main_j = 0; 
    begin_main_k = 0;
    end_main_i = parameters.Ni;
    end_main_j = parameters.Nj;
    end_main_k = parameters.Nk;
}

void FDTD_openmp::FDTD::update_E() {
    #pragma omp parallel for schedule(static)
    for (int k = begin_main_k; k < end_main_k; k++) {
        int index_k = k * Ni * Nj;
        int k_pred_k = k - 1;
        applyPeriodicBoundary(k_pred_k, Nk);
        k_pred_k = k_pred_k * Ni * Nj;
        for (int j = begin_main_j; j < end_main_j; j++) {
            int index_kj = j * Ni + index_k;
            int j_pred_kj = j - 1;
            applyPeriodicBoundary(j_pred_kj, Nj);
            j_pred_kj = j_pred_kj * Ni + index_k;
            int k_pred_kj = j * Ni + k_pred_k;
            #pragma omp simd
            for (int i = begin_main_i; i < end_main_i; i++) {
                int index = i + index_kj;
                int i_pred = i - 1;
                applyPeriodicBoundary(i_pred, Ni);
                i_pred = i_pred + index_kj;
                int j_pred = i + j_pred_kj;
                int k_pred = i + k_pred_kj;

                Ex[index] += cur_coef * Jx[index] + 
                                coef_E_dy * (Bz[index] - Bz[j_pred]) - 
                                coef_E_dz * (By[index] - By[k_pred]);
                Ey[index] += cur_coef * Jx[index] + 
                                coef_E_dz * (Bx[index] - Bx[k_pred]) - 
                                coef_E_dx * (Bz[index] - Bz[i_pred]);
                Ez[index] += cur_coef * Jx[index] + 
                                coef_E_dx * (By[index] - By[i_pred]) - 
                                coef_E_dy * (Bx[index] - Bx[j_pred]);
            }
        }
    }
}

void FDTD_openmp::FDTD::update_B() {
    #pragma omp parallel for schedule(static)
    for (int k = begin_main_k; k < end_main_k; k++) {
        int index_k = k * Ni * Nj;
        int k_next_k = k + 1;
        applyPeriodicBoundary(k_next_k, Nk);
        k_next_k = k_next_k * Ni * Nj;
        for (int j = begin_main_j; j < end_main_j; j++) {
            int index_kj = j * Ni + index_k;
            int j_next_kj = j + 1;
            applyPeriodicBoundary(j_next_kj, Nj);
            j_next_kj = j_next_kj * Ni + index_k;
            int k_next_kj = j * Ni + k_next_k;
            #pragma omp simd
            for (int i = begin_main_i; i < end_main_i; i++) {
                int index = i + index_kj;
                int i_next = i + 1;
                applyPeriodicBoundary(i_next, Ni);
                i_next = i_next + index_kj;
                int j_next = i + j_next_kj;
                int k_next = i + k_next_kj;

                Bx[index] += coef_B_dz * (Ey[k_next] - Ey[index]) - 
                                   coef_B_dy * (Ez[j_next] - Ez[index]);
                By[index] += coef_B_dx * (Ez[i_next] - Ez[index]) - 
                                   coef_B_dz * (Ex[k_next] - Ex[index]);
                Bz[index] += coef_B_dy * (Ex[j_next] - Ex[index]) - 
                                   coef_B_dx * (Ey[i_next] - Ey[index]);
            }
        }
    }
}

void FDTD_openmp::FDTD::zeroed_currents() {
    std::fill(Jx.begin(), Jx.end(), 0.0);
    std::fill(Jy.begin(), Jy.end(), 0.0);
    std::fill(Jz.begin(), Jz.end(), 0.0);
}

FDTD_openmp::Field& FDTD_openmp::FDTD::get_field(Component this_field) {
    switch (this_field) {
        case Component::JX: return Jx;
        case Component::JY: return Jy;
        case Component::JZ: return Jz;
        case Component::EX: return Ex;
        case Component::EY: return Ey;
        case Component::EZ: return Ez;
        case Component::BX: return Bx;
        case Component::BY: return By;
        case Component::BZ: return Bz;
        default: throw std::logic_error("ERROR: Invalid field component");
    }
}

void FDTD_openmp::FDTD::update_fields() {
    update_B();
    update_E();
    update_B();
}
