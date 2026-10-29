#include "test_FDTD_kokkos.h"

#include <gtest/gtest.h>

using namespace FDTD_kokkos;

const FP default_time_k = 5e-13;

std::function<FP(FP, FP[2])> initial_func_k = [](FP x, FP size[2])
{
	return sin(2.0 * FDTD_const::PI * (x - size[0]) / (size[1] - size[0]));
};
std::function<FP(FP, FP, FP[2])> true_func_k = [](FP x, FP t, FP size[2])
{
	return sin(2.0 * FDTD_const::PI * (x - size[0] - FDTD_const::C * t) / (size[1] - size[0]));
};

FP run_test_kokkos(Component test_field, SelectedFields current_fields,
	int Ni, int Nj, int Nk) {
    //---------------------
    Parameters params_1 {
        Ni,
        Nj,
        Nk,
        0.0, 1.0, 0.0, 2.0, 0.0, 3.0,
        (1 - 0) / static_cast<FP>(Ni),
        (2 - 0) / static_cast<FP>(Nj),
        (3 - 0) / static_cast<FP>(Nk)
    };
	int iters_1 = 16;
    FP dt_1 = default_time_k / static_cast<FP>(iters_1);
    FDTD method_1(params_1, dt_1);

    Test_FDTD test_1(params_1);
    test_1.initial_filling(method_1, current_fields, iters_1, initial_func_k);
    for (int t = 0; t < iters_1; t++) {
		method_1.update_fields();
	}

    FP err_1 = test_1.get_max_abs_error(method_1.get_field(test_field),
		test_field, true_func_k, default_time_k);
    //---------------------

    //---------------------
    Parameters params_2 {
        Ni * 2,
        Nj * 2,
        Nk * 2,
        0.0, 1.0, 0.0, 2.0, 0.0, 3.0,
        (1 - 0) / static_cast<FP>(Ni * 2),
        (2 - 0) / static_cast<FP>(Nj * 2),
        (3 - 0) / static_cast<FP>(Nk * 2)
    };
	int iters_2 = iters_1 * 2;
    FP dt_2 = default_time_k / static_cast<FP>(iters_2);
    FDTD method_2(params_2, dt_2);

    Test_FDTD test_2(params_2);
    test_2.initial_filling(method_2, current_fields, iters_2, initial_func_k);
    for (int t = 0; t < iters_2; t++) {
		method_2.update_fields();
	}

    FP err_2 = test_2.get_max_abs_error(method_2.get_field(test_field),
		test_field, true_func_k, default_time_k);
    //---------------------

    return err_1 / err_2;
}

TEST(Convergence_kokkos, x_axis_EY) 
{
    SelectedFields current_fields
	{
		Component::EY,
		Component::BZ
	};
    Component test_field = Component::EY;
    ASSERT_NEAR(run_test_kokkos(test_field, current_fields, 16, 8, 4), 4.0, 0.1);
}

TEST(Convergence_kokkos, x_axis_BZ) 
{
    SelectedFields current_fields
	{
		Component::EY, 
		Component::BZ
	};
    Component test_field = Component::BZ;
    ASSERT_NEAR(run_test_kokkos(test_field, current_fields, 16, 8, 4), 4.0, 0.1);
}

TEST(Convergence_kokkos, x_axis_EZ) 
{
	SelectedFields current_fields
	{
		Component::EZ,
		Component::BY
	};
	Component test_field = Component::EZ;
	ASSERT_NEAR(run_test_kokkos(test_field, current_fields, 16, 8, 4), 4.0, 0.1);
}

TEST(Convergence_kokkos, x_axis_BY)
{
	SelectedFields current_fields
	{
		Component::EZ,
		Component::BY
	};
	Component test_field = Component::BY;
	ASSERT_NEAR(run_test_kokkos(test_field, current_fields, 16, 8, 4), 4.0, 0.1);
}

TEST(Convergence_kokkos, y_axis_EX)
{
	SelectedFields current_fields
	{
		Component::EX,
		Component::BZ
	};
	Component test_field = Component::EX;
	ASSERT_NEAR(run_test_kokkos(test_field, current_fields, 8, 16, 4), 4.0, 0.1);
}

TEST(Convergence_kokkos, y_axis_BZ)
{
	SelectedFields current_fields
	{
		Component::EX,
		Component::BZ
	};
	Component test_field = Component::BZ;
	ASSERT_NEAR(run_test_kokkos(test_field, current_fields, 8, 16, 4), 4.0, 0.1);
}

TEST(Convergence_kokkos, y_axis_EZ)
{
	SelectedFields current_fields
	{
		Component::EZ,
		Component::BX
	};
	Component test_field = Component::EZ;
	ASSERT_NEAR(run_test_kokkos(test_field, current_fields, 8, 16, 4), 4.0, 0.1);
}

TEST(Convergence_kokkos, y_axis_BX)
{
	SelectedFields current_fields
	{
		Component::EZ,
		Component::BX
	};
	Component test_field = Component::BX;
	ASSERT_NEAR(run_test_kokkos(test_field, current_fields, 8, 16, 4), 4.0, 0.1);
}

TEST(Convergence_kokkos, z_axis_EX)
{
	SelectedFields current_fields
	{
		Component::EX,
		Component::BY
	};
	Component test_field = Component::EX;
	ASSERT_NEAR(run_test_kokkos(test_field, current_fields, 4, 8, 16), 4.0, 0.1);
}

TEST(Convergence_kokkos, z_axis_BY)
{
	SelectedFields current_fields
	{
		Component::EX,
		Component::BY
	};
	Component test_field = Component::BY;
	ASSERT_NEAR(run_test_kokkos(test_field, current_fields, 4, 8, 16), 4.0, 0.1);
}

TEST(Convergence_kokkos, z_axis_EY)
{
	SelectedFields current_fields
	{
		Component::EY,
		Component::BX
	};
	Component test_field = Component::EY;
	ASSERT_NEAR(run_test_kokkos(test_field, current_fields, 4, 8, 16), 4.0, 0.1);
}

TEST(Convergence_kokkos, z_axis_BX)
{
	SelectedFields current_fields
	{
		Component::EY,
		Component::BX
	};
	Component test_field = Component::BX;
	ASSERT_NEAR(run_test_kokkos(test_field, current_fields, 4, 8, 16), 4.0, 0.1);
}
