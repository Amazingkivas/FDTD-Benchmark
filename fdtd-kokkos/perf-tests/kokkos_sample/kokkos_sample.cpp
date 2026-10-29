#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <Kokkos_Core.hpp>
#include <cstdlib>

#include "FDTD_kokkos.h"
#include "FDTD_PML_kokkos.h"

using namespace FDTD_kokkos;


void spherical_wave(int n, int it, const std::string base_path = "../../PlotScript/") {
    CurrentParameters cur_param {
        8,
        4,
        0.2
    };

    double T = cur_param.period;
    double Tx = cur_param.period_x;
    double Ty = cur_param.period_y;
    double Tz = cur_param.period_z;

    cur_param.iterations = static_cast<int>(static_cast<double>(cur_param.period) / cur_param.dt);

    std::function<double(double, double, double, double)> cur_func =
        [T, Tx, Ty, Tz](double x, double y, double z, double t) {
        return sin(2.0 * FDTD_const::PI * t / T)
            * pow(cos(2.0 * FDTD_const::PI * x / Tx), 2.0)
            * pow(cos(2.0 * FDTD_const::PI * y / Ty), 2.0)
            * pow(cos(2.0 * FDTD_const::PI * z / Tz), 2.0);
    };

    double d = FDTD_const::C;
    double boundary = static_cast<double>(n) / 2.0 * d;

    Parameters params {
        n,          // Ni
        n,          // Nj
        n,          // Nk
        -boundary,  // x_min
        boundary,   // x_max
        -boundary,  // y_min
        boundary,   // y_max
        -boundary,  // z_min
        boundary,   // z_max
        d,          // dx
        d,          // dy
        d           // dz
    };

    FDTD_kokkos::FDTD method(params, cur_param.dt);

    int cur_time = std::min(cur_param.iterations, it);

    int start_i = static_cast<int>(floor((-Tx / 4.0 - params.ax) / params.dx));
    int start_j = static_cast<int>(floor((-Ty / 4.0 - params.ay) / params.dy));
    int start_k = static_cast<int>(floor((-Tz / 4.0 - params.az) / params.dz));

    int max_i = static_cast<int>(floor((Tx / 4.0 - params.ax) / params.dx));
    int max_j = static_cast<int>(floor((Ty / 4.0 - params.ay) / params.dy));
    int max_k = static_cast<int>(floor((Tz / 4.0 - params.az) / params.dz));

    
    int Ni = params.Ni;
    int Nj = params.Nj;
    int Nk = params.Nk;
    double dx = params.dx;
    double dy = params.dy;
    double dz = params.dz;
    double dt_cur = cur_param.dt;

    double T_ = cur_param.period;
    double Tx_ = cur_param.period_x;
    double Ty_ = cur_param.period_y;
    double Tz_ = cur_param.period_z;

    auto Jx = method.get_field(Component::JX);
    auto Jy = method.get_field(Component::JY);
    auto Jz = method.get_field(Component::JZ);

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < cur_time; ++t) {
        double time = (t + 1) * dt_cur;
        
        Kokkos::parallel_for("SetCurrent", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({start_i, start_j, start_k}, 
                                                    {max_i, max_j, max_k}),
            KOKKOS_LAMBDA(int i, int j, int k) {
                int idx = i + j * Ni + k * Ni * Nj;
                double x = i * dx;
                double y = j * dy;
                double z = k * dz;
                
                double val = sin(2.0 * FDTD_const::PI * time / T_) *
                            pow(cos(2.0 * FDTD_const::PI * x / Tx_), 2.0) *
                            pow(cos(2.0 * FDTD_const::PI * y / Ty_), 2.0) *
                            pow(cos(2.0 * FDTD_const::PI * z / Tz_), 2.0);
                
                Jx(idx) = val;
                Jy(idx) = val;
                Jz(idx) = val;
            });
        
        Kokkos::fence();
        method.update_fields();
        Kokkos::fence();
    }
    method.zeroed_currents();
    Kokkos::fence();
    for (int t = cur_time; t < it; t++) {
        Kokkos::parallel_for("SetCurrent", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({start_i, start_j, start_k}, 
                                                    {max_i, max_j, max_k}),
            KOKKOS_LAMBDA(int i, int j, int k) {
                int idx = i + j * Ni + k * Ni * Nj;
                
                Jx(idx) = 0.0;
                Jy(idx) = 0.0;
                Jz(idx) = 0.0;
            });
        method.update_fields();
        Kokkos::fence();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " s" << std::endl;

    auto& Ex_device = method.get_field(Component::EX);

    auto Ex_host = Kokkos::create_mirror_view(Ex_device);

    Kokkos::deep_copy(Ex_host, Ex_device);

    int i = params.Nk / 2;
    for (int j = params.Nj/2 - 5; j < params.Nj/2 + 5; j++) {
        for (int k = params.Ni/2 - 5; k < params.Ni/2 + 5; k++) {
            int index = i + j * params.Ni + k * params.Ni * params.Nj;
            std::cout << std::setw(12) << std::fixed << std::setprecision(5) 
                    << Ex_host(index);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv); {
        if (Kokkos::hwloc::available())
            std::cout << "hwloc available" << std::endl;
        std::ostringstream msg;
        msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
          << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
          << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
        Kokkos::print_configuration(msg);
        std::cout << msg.str();


        std::ifstream source_fin;
        std::vector<char*> arguments(argv, argv + argc);

        if (argc == 1) {
            int N = 32;
            int Iterations = 100;
            spherical_wave(N, Iterations, "../../");
        }
        else if (argc == 3) {
            int N = std::atoi(arguments[1]);
            int Iterations = std::atoi(arguments[2]);
            spherical_wave(N, Iterations, "");
        }
        else {
            std::cout << "ERROR: Incorrect number of parameters" << std::endl;
            exit(1);
        }
    }
    Kokkos::finalize();
    return 0;
}
