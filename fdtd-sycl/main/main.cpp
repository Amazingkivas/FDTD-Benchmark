#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "FDTD_sycl.h"

using namespace FDTD_sycl;

void spherical_wave(int n, int iterations, std::string base_path = "") {
    CurrentParameters current_parameters{
        8,
        4,
        0.2,
    };

    const double period = current_parameters.period;
    const double period_x = current_parameters.period_x;
    const double period_y = current_parameters.period_y;
    const double period_z = current_parameters.period_z;
    current_parameters.iterations =
        static_cast<int>(static_cast<double>(current_parameters.period) / current_parameters.dt);

    auto current_function = [=](double x, double y, double z, double time) {
        return std::sin(2.0 * FDTD_const::PI * time / period)
             * std::pow(std::cos(2.0 * FDTD_const::PI * x / period_x), 2.0)
             * std::pow(std::cos(2.0 * FDTD_const::PI * y / period_y), 2.0)
             * std::pow(std::cos(2.0 * FDTD_const::PI * z / period_z), 2.0);
    };

    const double spacing = FDTD_const::C;
    const double boundary = static_cast<double>(n) / 2.0 * spacing;

    Parameters parameters{
        n, n, n,
        -boundary, boundary,
        -boundary, boundary,
        -boundary, boundary,
        spacing, spacing, spacing
    };

    FDTD_sycl::FDTD method(parameters, current_parameters.dt);

    const int current_time = std::min(current_parameters.iterations, iterations);

    const int start_i = static_cast<int>(std::floor((-period_x / 4.0 - parameters.ax) / parameters.dx));
    const int start_j = static_cast<int>(std::floor((-period_y / 4.0 - parameters.ay) / parameters.dy));
    const int start_k = static_cast<int>(std::floor((-period_z / 4.0 - parameters.az) / parameters.dz));
    const int max_i = static_cast<int>(std::floor((period_x / 4.0 - parameters.ax) / parameters.dx));
    const int max_j = static_cast<int>(std::floor((period_y / 4.0 - parameters.ay) / parameters.dy));
    const int max_k = static_cast<int>(std::floor((period_z / 4.0 - parameters.az) / parameters.dz));

    auto& queue = method.get_queue();
    auto Jx_buffer = method.get_field_buffer(Component::JX);
    auto Jy_buffer = method.get_field_buffer(Component::JY);
    auto Jz_buffer = method.get_field_buffer(Component::JZ);

    const sycl::range<3> current_range(max_k - start_k, max_j - start_j, max_i - start_i);
    const sycl::id<3> current_offset(start_k, start_j, start_i);

    const auto start = std::chrono::high_resolution_clock::now();
    for (int time_index = 0; time_index < current_time; ++time_index) {
        const double time_value = static_cast<double>(time_index + 1) * current_parameters.dt;

        queue.submit([&](sycl::handler& handler) {
            auto Jx_acc = Jx_buffer.get_access<sycl::access::mode::write>(handler);
            auto Jy_acc = Jy_buffer.get_access<sycl::access::mode::write>(handler);
            auto Jz_acc = Jz_buffer.get_access<sycl::access::mode::write>(handler);

            handler.parallel_for(current_range, current_offset, [=](sycl::id<3> item) {
                const int k = item[0];
                const int j = item[1];
                const int i = item[2];

                const double value = current_function(
                    static_cast<double>(i) * parameters.dx,
                    static_cast<double>(j) * parameters.dy,
                    static_cast<double>(k) * parameters.dz,
                    time_value);

                Jx_acc[k][j][i] = value;
                Jy_acc[k][j][i] = value;
                Jz_acc[k][j][i] = value;
            });
        });

        method.update_fields();
    }

    method.zeroed_currents();
    for (int time_index = current_time; time_index < iterations; ++time_index) {
        method.update_fields();
    }

    queue.wait_and_throw();

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " s" << std::endl;

    std::cout << "SYCL results: \n" << std::endl;
    const int i = parameters.Ni / 2;
    sycl::host_accessor Ex_host_acc(method.get_field_buffer(Component::EX), sycl::read_only);

    for (int j = parameters.Nj / 2 - 5; j < parameters.Nj / 2 + 5; ++j) {
        for (int k = parameters.Nk / 2 - 5; k < parameters.Nk / 2 + 5; ++k) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(5)
                      << Ex_host_acc[i][j][k];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    int grid_size = 512;
    int iterations = 25;

    if (argc == 3) {
        grid_size = std::atoi(argv[1]);
        iterations = std::atoi(argv[2]);
    } else if (argc != 1) {
        std::cout << "ERROR: Incorrect number of parameters" << std::endl;
        return 1;
    }

    spherical_wave(grid_size, iterations, "../../");
    return 0;
}
