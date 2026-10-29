#pragma once

#include "FDTD.h"


namespace FDTD_openmp {

class Test_FDTD
{
private:
	Parameters parameters;
	FP sign = 1.0;
	Axis axis = Axis::X;
	
	void set_sign(Component field_E, Component field_B);
	void set_axis(Component field_E, Component field_B);
	FP get_shift(Component _field, FP step);

public:
	Test_FDTD(Parameters);

	void initial_filling(FDTD& _test, SelectedFields, int iters,
		std::function<FP(FP, FP[2])>& init_function);

	FP get_max_abs_error(Field& this_field, Component field,
		std::function<FP(FP, FP, FP[2])>& true_function, FP time);
};

}
