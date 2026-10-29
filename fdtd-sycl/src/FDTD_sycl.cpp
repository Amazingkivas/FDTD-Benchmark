#include "FDTD_sycl.h"

#include <iostream>
#include <stdexcept>

#include <sycl/CL/sycl.hpp>

namespace FDTD_sycl {

AllDevices node;

FDTD::FDTD(Parameters input_parameters, FP time_step)
    : parameters(input_parameters),
      dt(time_step),
      q(node.active_device),
      Ni(input_parameters.Ni),
      Nj(input_parameters.Nj),
      Nk(input_parameters.Nk),
      grid_range(input_parameters.Nk, input_parameters.Nj, input_parameters.Ni),
      Jx(Ni * Nj * Nk),
      Jy(Ni * Nj * Nk),
      Jz(Ni * Nj * Nk),
      Ex(Ni * Nj * Nk),
      Ey(Ni * Nj * Nk),
      Ez(Ni * Nj * Nk),
      Bx(Ni * Nj * Nk),
      By(Ni * Nj * Nk),
      Bz(Ni * Nj * Nk),
      bufJx(Jx.data(), grid_range),
      bufJy(Jy.data(), grid_range),
      bufJz(Jz.data(), grid_range),
      bufEx(Ex.data(), grid_range),
      bufEy(Ey.data(), grid_range),
      bufEz(Ez.data(), grid_range),
      bufBx(Bx.data(), grid_range),
      bufBy(By.data(), grid_range),
      bufBz(Bz.data(), grid_range) {
    if (Ni <= 0 || Nj <= 0 || Nk <= 0 || dt <= 0) {
        throw std::invalid_argument("ERROR: Invalid grid size (must be > 0).");
    }

    std::cout << "SYCL FDTD running on device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    const FP cdt = FDTD_const::C * dt;

    coef_E_dx = cdt / parameters.dx;
    coef_E_dy = cdt / parameters.dy;
    coef_E_dz = cdt / parameters.dz;

    dx = parameters.dx;
    dy = parameters.dy;
    dz = parameters.dz;

    coef_B_dx = cdt / (2.0 * dx);
    coef_B_dy = cdt / (2.0 * dy);
    coef_B_dz = cdt / (2.0 * dz);

    cur_coef = -4.0 * FDTD_const::PI * dt;

    zero_fields();
    zeroed_currents();
}

void FDTD::zero_fields() {
    auto kernel = [&](sycl::handler& handler) {
        auto Ex_acc = bufEx.get_access<sycl::access::mode::write>(handler);
        auto Ey_acc = bufEy.get_access<sycl::access::mode::write>(handler);
        auto Ez_acc = bufEz.get_access<sycl::access::mode::write>(handler);
        auto Bx_acc = bufBx.get_access<sycl::access::mode::write>(handler);
        auto By_acc = bufBy.get_access<sycl::access::mode::write>(handler);
        auto Bz_acc = bufBz.get_access<sycl::access::mode::write>(handler);

        handler.parallel_for(grid_range, [=](sycl::item<3> item) {
            const int k = item[0];
            const int j = item[1];
            const int i = item[2];
            Ex_acc[k][j][i] = 0.0;
            Ey_acc[k][j][i] = 0.0;
            Ez_acc[k][j][i] = 0.0;
            Bx_acc[k][j][i] = 0.0;
            By_acc[k][j][i] = 0.0;
            Bz_acc[k][j][i] = 0.0;
        });
    };

    q.submit(kernel).wait_and_throw();
}

void FDTD::zeroed_currents() {
    auto kernel = [&](sycl::handler& handler) {
        auto Jx_acc = bufJx.get_access<sycl::access::mode::write>(handler);
        auto Jy_acc = bufJy.get_access<sycl::access::mode::write>(handler);
        auto Jz_acc = bufJz.get_access<sycl::access::mode::write>(handler);

        handler.parallel_for(grid_range, [=](sycl::item<3> item) {
            const int k = item[0];
            const int j = item[1];
            const int i = item[2];
            Jx_acc[k][j][i] = 0.0;
            Jy_acc[k][j][i] = 0.0;
            Jz_acc[k][j][i] = 0.0;
        });
    };

    q.submit(kernel).wait_and_throw();
}

void FDTD::update_B() {
    const sycl::id<3> start(1, 1, 1);
    const sycl::range<3> range(Nk - 2, Nj - 2, Ni - 2);

    const int local_Ni = Ni;
    const int local_Nj = Nj;
    const int local_Nk = Nk;
    const FP local_coef_B_dx = coef_B_dx;
    const FP local_coef_B_dy = coef_B_dy;
    const FP local_coef_B_dz = coef_B_dz;

    auto kernel = [&](sycl::handler& handler) {
        auto Ex_acc = bufEx.get_access<sycl::access::mode::read>(handler);
        auto Ey_acc = bufEy.get_access<sycl::access::mode::read>(handler);
        auto Ez_acc = bufEz.get_access<sycl::access::mode::read>(handler);

        auto Bx_acc = bufBx.get_access<sycl::access::mode::read_write>(handler);
        auto By_acc = bufBy.get_access<sycl::access::mode::read_write>(handler);
        auto Bz_acc = bufBz.get_access<sycl::access::mode::read_write>(handler);

        handler.parallel_for(range, start, [=](sycl::item<3> id) {
            const int k = id[0];
            const int j = id[1];
            const int i = id[2];

            const int k_next = applyPeriodic(k + 1, local_Nk);
            const int j_next = applyPeriodic(j + 1, local_Nj);
            const int i_next = applyPeriodic(i + 1, local_Ni);

            Bx_acc[k][j][i] += local_coef_B_dz * (Ey_acc[k_next][j][i] - Ey_acc[k][j][i])
                              - local_coef_B_dy * (Ez_acc[k][j_next][i] - Ez_acc[k][j][i]);
            By_acc[k][j][i] += local_coef_B_dx * (Ez_acc[k][j][i_next] - Ez_acc[k][j][i])
                              - local_coef_B_dz * (Ex_acc[k_next][j][i] - Ex_acc[k][j][i]);
            Bz_acc[k][j][i] += local_coef_B_dy * (Ex_acc[k][j_next][i] - Ex_acc[k][j][i])
                              - local_coef_B_dx * (Ey_acc[k][j][i_next] - Ey_acc[k][j][i]);
        });
    };

    q.submit(kernel).wait_and_throw();
}

void FDTD::update_E() {
    const sycl::id<3> start(1, 1, 1);
    const sycl::range<3> range(Nk - 2, Nj - 2, Ni - 2);

    const int local_Ni = Ni;
    const int local_Nj = Nj;
    const int local_Nk = Nk;
    const FP local_cur_coef = cur_coef;
    const FP local_coef_E_dx = coef_E_dx;
    const FP local_coef_E_dy = coef_E_dy;
    const FP local_coef_E_dz = coef_E_dz;

    auto kernel = [&](sycl::handler& handler) {
        auto Jx_acc = bufJx.get_access<sycl::access::mode::read>(handler);
        auto Jy_acc = bufJy.get_access<sycl::access::mode::read>(handler);
        auto Jz_acc = bufJz.get_access<sycl::access::mode::read>(handler);
        auto Bx_acc = bufBx.get_access<sycl::access::mode::read>(handler);
        auto By_acc = bufBy.get_access<sycl::access::mode::read>(handler);
        auto Bz_acc = bufBz.get_access<sycl::access::mode::read>(handler);
        auto Ex_acc = bufEx.get_access<sycl::access::mode::read_write>(handler);
        auto Ey_acc = bufEy.get_access<sycl::access::mode::read_write>(handler);
        auto Ez_acc = bufEz.get_access<sycl::access::mode::read_write>(handler);

        handler.parallel_for(range, start, [=](sycl::item<3> id) {
            const int k = id[0];
            const int j = id[1];
            const int i = id[2];

            const int k_prev = applyPeriodic(k - 1, local_Nk);
            const int j_prev = applyPeriodic(j - 1, local_Nj);
            const int i_prev = applyPeriodic(i - 1, local_Ni);

            Ex_acc[k][j][i] += local_cur_coef * Jx_acc[k][j][i]
                              + local_coef_E_dy * (Bz_acc[k][j][i] - Bz_acc[k][j_prev][i])
                              - local_coef_E_dz * (By_acc[k][j][i] - By_acc[k_prev][j][i]);

            Ey_acc[k][j][i] += local_cur_coef * Jy_acc[k][j][i]
                              + local_coef_E_dz * (Bx_acc[k][j][i] - Bx_acc[k_prev][j][i])
                              - local_coef_E_dx * (Bz_acc[k][j][i] - Bz_acc[k][j][i_prev]);

            Ez_acc[k][j][i] += local_cur_coef * Jz_acc[k][j][i]
                              + local_coef_E_dx * (By_acc[k][j][i] - By_acc[k][j][i_prev])
                              - local_coef_E_dy * (Bx_acc[k][j][i] - Bx_acc[k][j_prev][i]);
        });
    };

    q.submit(kernel).wait_and_throw();
}

void FDTD::update_fields() {
    update_B();
    update_E();
    update_B();
}

sycl::buffer<FP, 3>& FDTD::get_field_buffer(Component field) {
    switch (field) {
        case Component::JX: return bufJx;
        case Component::JY: return bufJy;
        case Component::JZ: return bufJz;
        case Component::EX: return bufEx;
        case Component::EY: return bufEy;
        case Component::EZ: return bufEz;
        case Component::BX: return bufBx;
        case Component::BY: return bufBy;
        case Component::BZ: return bufBz;
        default: throw std::logic_error("ERROR: Invalid field component");
    }
}

}
