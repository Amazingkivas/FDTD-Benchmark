#pragma once

#include <CL/sycl.hpp>

#include <vector>

#include "Structures.h"
#include "Constants.h"
#include "Enums.h"
#include "FP.h"
#include "allocate.h"
#include "sycl_devices.h"

namespace FDTD_sycl {

using namespace FDTD_enums;
using namespace FDTD_struct;

using Field = std::vector<FP, no_init_allocator<FP>>;

inline int applyPeriodic(const int& i, const int& N) {
    const int is_minus_one = (i < 0);
    const int is_N = (i == N);
    return (N - 1) * is_minus_one + i * !(is_minus_one || is_N);
}

}
