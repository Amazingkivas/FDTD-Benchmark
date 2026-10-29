#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <omp.h>
#include <vector>

#include "allocate.h"
#include "Structures.h"


namespace FDTD_openmp {
    using Field = std::vector<FP, no_init_allocator<FP>>;
    using Function = std::function<int(int, int, int)>;
    using namespace FDTD_enums;
    using namespace FDTD_struct;
}
