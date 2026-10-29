#include "test_FDTD.h" 

FDTD_openmp::Test_FDTD::Test_FDTD(Parameters _parameters) : parameters(_parameters) {}

void FDTD_openmp::Test_FDTD::initial_filling(FDTD_openmp::FDTD& _test, SelectedFields fields, int iters,
	std::function<FP(FP, FP[2])>& init_function) {
	set_axis(fields.selected_E, fields.selected_B);
	set_sign(fields.selected_E, fields.selected_B);
	if (axis == Axis::X) {
		FP x_b = parameters.dx / 2.0;
		FP size_x[2] = { parameters.ax, parameters.bx };
		for (int i = 0; i < parameters.Ni; i++) {
			FP x = static_cast<FP>(i) * parameters.dx;
			for (int j = 0; j < parameters.Nj; j++) {
				for (int k = 0; k < parameters.Nk; k++) {
                    int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
					_test.get_field(fields.selected_E)[index] = sign * init_function(x, size_x);
					_test.get_field(fields.selected_B)[index] = init_function(x_b + x, size_x);
				}
			}
		}
	}
	else if (axis == Axis::Y) {
		FP y_b = parameters.dy / 2.0;
		FP size_y[2] = { parameters.ay, parameters.by };
		for (int j = 0; j < parameters.Nj; j++) {
			FP y = static_cast<FP>(j) * parameters.dy;
			for (int k = 0; k < parameters.Nk; k++) {
				for (int i = 0; i < parameters.Ni; i++) {
                    int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
					_test.get_field(fields.selected_E)[index] = sign * init_function(y, size_y);
					_test.get_field(fields.selected_B)[index] = init_function(y_b + y, size_y);
				}
			}
		}
	}
	else {
		FP z_b = parameters.dz / 2.0;
		FP size_z[2] = { parameters.az, parameters.bz };
		for (int k = 0; k < parameters.Nk; k++) {
			FP z = static_cast<FP>(k) * parameters.dz;
			for (int i = 0; i < parameters.Ni; i++) {
				for (int j = 0; j < parameters.Nj; j++) {
                    int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
					_test.get_field(fields.selected_E)[index] = sign * init_function(z, size_z);
					_test.get_field(fields.selected_B)[index] = init_function(z_b + z, size_z);
				}
			}
		}
	}
}

void FDTD_openmp::Test_FDTD::set_sign(Component field_E, Component field_B) {
	if (field_E == Component::EX && field_B == Component::BZ ||
		field_E == Component::EZ && field_B == Component::BY ||
		field_E == Component::EY && field_B == Component::BX) {
		sign = -1.0;
	}
	else if (field_E == Component::EY && field_B == Component::BZ ||
		     field_E == Component::EZ && field_B == Component::BX ||
		     field_E == Component::EX && field_B == Component::BY) {
		sign = 1.0;
	}
	else throw std::logic_error("ERROR: invalid selected fields");
}
void FDTD_openmp::Test_FDTD::set_axis(Component field_E, Component field_B) {
	if (field_E == Component::EY && field_B == Component::BZ ||
		field_E == Component::EZ && field_B == Component::BY) {
		axis = Axis::X;
	}
	else if (field_E == Component::EX && field_B == Component::BZ ||
		     field_E == Component::EZ && field_B == Component::BX) {
		axis = Axis::Y;
	}
	else if (field_E == Component::EX && field_B == Component::BY ||
		     field_E == Component::EY && field_B == Component::BX) {
		axis = Axis::Z;
	}
	else throw std::logic_error("ERROR: invalid selected fields");
}
FP FDTD_openmp::Test_FDTD::get_shift(Component _field, FP step) {
	if (static_cast<int>(_field) > static_cast<int>(Component::EZ)) {
		sign = 1.0;
		return step / 2.0;
	}
	return 0.0;
}

FP FDTD_openmp::Test_FDTD::get_max_abs_error(FDTD_openmp::Field& this_field, Component field,
	std::function<FP(FP, FP, FP[2])>& true_function, FP time) {
	FP this_error = 0.0;
	FP max_abs_error = 0.0;
	if (axis == Axis::X) {
		FP x = get_shift(field, parameters.dx);
		FP size_x[2] = { parameters.ax, parameters.bx };
		int j = 0;
		int k = 0;
		for (int i = 0; i < parameters.Ni; ++i, x += parameters.dx) {
            int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
			this_error = fabs(sign * this_field[index] - true_function(x, time, size_x));
			if (this_error > max_abs_error)
				max_abs_error = this_error;
		}
	}
	else if (axis == Axis::Y) {
		FP y = get_shift(field, parameters.dy);
		FP size_y[2] = { parameters.ay, parameters.by };
		int i = 0;
		int k = 0;
		for (int j = 0; j < parameters.Nj; ++j, y += parameters.dy) {
            int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
			this_error = fabs(sign * this_field[index] - true_function(y, time, size_y));
			if (this_error > max_abs_error)
				max_abs_error = this_error;
		}
	}
	else {
		FP z = get_shift(field, parameters.dz);
		FP size_z[2] = { parameters.az, parameters.bz };
		int i = 0;
		int j = 0;
		for (int k = 0; k < parameters.Nk; ++k, z += parameters.dz) {
            int index = i + j * parameters.Ni + k * parameters.Ni * parameters.Nj;
			this_error = fabs(sign * this_field[index] - true_function(z, time, size_z));
			if (this_error > max_abs_error)
				max_abs_error = this_error;
		}
	}
	return max_abs_error;
}
