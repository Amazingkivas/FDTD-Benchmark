#pragma once

#include <CL/sycl.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

namespace FDTD_sycl {

template <typename T>
using sycl_vector = std::vector<T, sycl::usm_allocator<T, sycl::usm::alloc::shared>>;

class AllDevices {
public:
    AllDevices()
        : default_device{sycl::default_selector{}, sycl::async_handler{}},
          cpu_device{sycl::cpu_selector{}, sycl::async_handler{}},
          gpu_device{create_gpu_fallback_queue()},
          active_device{select_active_device()} {}

    sycl::queue default_device;
    sycl::queue cpu_device;
    sycl::queue gpu_device;
    sycl::queue active_device;

private:
    static std::string normalized_device_preference() {
        const char* value = std::getenv("FDTD_SYCL_DEVICE");
        if (value == nullptr) {
            return "auto";
        }

        std::string preference(value);
        std::transform(preference.begin(), preference.end(), preference.begin(),
            [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
        return preference;
    }

    static sycl::queue create_gpu_fallback_queue() {
        try {
            return sycl::queue{sycl::gpu_selector{}, sycl::async_handler{}};
        } catch (const sycl::exception&) {
            return sycl::queue{sycl::cpu_selector{}, sycl::async_handler{}};
        }
    }

    sycl::queue select_active_device() const {
        const std::string preference = normalized_device_preference();

        if (preference == "cpu") {
            return cpu_device;
        }

        if (preference == "default") {
            return default_device;
        }

        if (preference == "gpu" && gpu_device.get_device().is_gpu()) {
            return gpu_device;
        }

        if (gpu_device.get_device().is_gpu()) {
            return gpu_device;
        }

        return cpu_device;
    }
};

extern AllDevices node;

}
