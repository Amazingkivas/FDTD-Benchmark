#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <chrono>

#include "test_FDTD.h"
#include "FDTD_PML.h"

using namespace FDTD_openmp;


void spherical_wave(int n, int it, std::string base_path = "") {
    CurrentParameters cur_param {
        8,
        4,
        0.2,
    };
    double T = cur_param.period;
    double Tx = cur_param.period_x;
    double Ty = cur_param.period_y;
    double Tz = cur_param.period_z;
    cur_param.iterations = static_cast<int>(static_cast<double>(cur_param.period) / cur_param.dt);
    std::function<double(double, double, double, double)> cur_func 
        = [T, Tx, Ty, Tz](double x, double y, double z, double t) {
        return sin(2.0 * FDTD_const::PI * t / T) 
            * pow(cos(2.0 * FDTD_const::PI * x / Tx), 2.0) 
            * pow(cos(2.0 * FDTD_const::PI * y / Ty), 2.0) 
            * pow(cos(2.0 * FDTD_const::PI * z / Tz), 2.0);
    };
    
    // Initialization of the structures and method
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

    FDTD_openmp::FDTD method(params, cur_param.dt);

    int cur_time = std::min(cur_param.iterations, it);

    int start_i = static_cast<int>(floor((-Tx / 4.0 - params.ax) / params.dx));
    int start_j = static_cast<int>(floor((-Ty / 4.0 - params.ay) / params.dy));
    int start_k = static_cast<int>(floor((-Tz / 4.0 - params.az) / params.dz));

    int max_i = static_cast<int>(floor((Tx / 4.0 - params.ax) / params.dx));
    int max_j = static_cast<int>(floor((Ty / 4.0 - params.ay) / params.dy));
    int max_k = static_cast<int>(floor((Tz / 4.0 - params.az) / params.dz));

    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < cur_time; t++) {
        for (int k = start_k; k < max_k; ++k) {
            for (int j = start_j; j < max_j; ++j) {
                for (int i = start_i; i < max_i; ++i) {
                    int index = i + j * params.Ni + k * params.Ni * params.Nj;
                    double value = cur_func(static_cast<double>(i) * params.dx,
                                            static_cast<double>(j) * params.dy,
                                            static_cast<double>(k) * params.dz,
                                            static_cast<double>(t + 1) * cur_param.dt);

                    method.get_field(Component::JX)[index] = value;
                    method.get_field(Component::JY)[index] = value;
                    method.get_field(Component::JZ)[index] = value;
                }
            }
        }
        method.update_fields();
    }
    method.zeroed_currents();
    for (int t = cur_time; t < it; t++) {
        method.update_fields();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " s" << std::endl;

#ifdef __PML_TEST__
    FDTD_openmp::FDTD_PML pml_method(params, cur_param.dt, 0.2);

    auto start_pml = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < cur_time; t++) {
        for (int k = start_k; k < max_k; ++k) {
            for (int j = start_j; j < max_j; ++j) {
                for (int i = start_i; i < max_i; ++i) {
                    int index = i + j * params.Ni + k * params.Ni * params.Nj;
                    double value = cur_func(static_cast<double>(i) * params.dx,
                                            static_cast<double>(j) * params.dy,
                                            static_cast<double>(k) * params.dz,
                                            static_cast<double>(t + 1) * cur_param.dt);

                    pml_method.get_field(Component::JX)[index] = value;
                    pml_method.get_field(Component::JY)[index] = value;
                    pml_method.get_field(Component::JZ)[index] = value;
                }
            }
        }
        pml_method.update_fields();
    }
    pml_method.zeroed_currents();
    for (int t = cur_time; t < it; t++) {
        pml_method.update_fields();
    }
    auto end_pml = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_pml = end_pml - start_pml;
    std::cout << "Execution time (PML): " << elapsed_pml.count() << " s" << std::endl;
#endif //__PML_TEST__

    int k = params.Nk/2;
    for (int j = params.Nj/2 - 5; j < params.Nj/2 + 5; j++) {
        for (int i = params.Ni/2 - 5; i < params.Ni/2 + 5; i++) {
            int index = i + j * params.Ni + k * params.Ni * params.Nj;
            std::cout << std::setw(12) << std::fixed << std::setprecision(5) 
                  << method.get_field(Component::EX)[index];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

#ifdef __PML_TEST__
    std::cout << "PML: \n" << std::endl;
    for (int j = params.Nj/2 - 5; j < params.Nj/2 + 5; j++) {
        for (int i = params.Ni/2 - 5; i < params.Ni/2 + 5; i++) {
            int index = i + j * params.Ni + k * params.Ni * params.Nj;
            std::cout << std::setw(12) << std::fixed << std::setprecision(5) 
                  << pml_method.get_field(Component::EX)[index];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
#endif //__PML_TEST__
}

int main(int argc, char* argv[]) {
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
        spherical_wave(N, Iterations);
    }
    else {
        std::cout << "ERROR: Incorrect number of parameters" << std::endl;
        exit(1);
    }
    return 0;
}
