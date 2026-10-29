#pragma once

#include "Constants.h"
#include "Enums.h"
#include "FP.h"

namespace FDTD_struct {
	struct SelectedFields {
		FDTD_enums::Component selected_E;
		FDTD_enums::Component selected_B;
	};

	struct CurrentParameters {
		int period;
		int m;
		FP dt;
		int iterations;
		FP period_x = static_cast<FP>(m) * FDTD_const::C;
		FP period_y = static_cast<FP>(m) * FDTD_const::C;
		FP period_z = static_cast<FP>(m) * FDTD_const::C;
	};

	struct Parameters {
		int Ni;
		int Nj;
		int Nk;

		FP ax;
		FP bx;
		FP ay;
		FP by;
		FP az;
		FP bz;

		FP dx;
		FP dy;
		FP dz;
	};
}
