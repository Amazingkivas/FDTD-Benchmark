#pragma once

#include <cmath>
#include <iostream>
#include <functional>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

#include "Structures.h"


namespace FDTD_kokkos {
    using Device = Kokkos::DefaultExecutionSpace;
    using Field = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Aligned>>;
    using TimeField = std::vector<Field>;
    using Function = std::function<int(int, int, int)>;
    using InitFunction = std::function<double(double, double, double, double)>;
    using namespace FDTD_enums;
    using namespace FDTD_struct;
}
