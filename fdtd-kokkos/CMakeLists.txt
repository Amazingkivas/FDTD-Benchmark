add_library(fdtd_kokkos
    src/FDTD_kokkos/FDTD_kokkos.cpp
    src/FDTD_kokkos/FDTD_PML_kokkos.cpp
    src/FDTD_kokkos/test_FDTD_kokkos.cpp
)

target_include_directories(fdtd_kokkos PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/FDTD_kokkos
)

target_link_libraries(fdtd_kokkos PUBLIC Kokkos::kokkos)

add_executable(test_FDTD_method_kokkos
    unit-tests/test_FDTD_method_kokkos.cpp
    unit-tests/test_main.cpp
)
target_link_libraries(test_FDTD_method_kokkos PRIVATE fdtd_kokkos gtest)
add_test(NAME fdtd-kokkos-unit-tests COMMAND test_FDTD_method_kokkos)

add_executable(kokkos_sample perf-tests/kokkos_sample/kokkos_sample.cpp)
target_link_libraries(kokkos_sample PRIVATE fdtd_kokkos)
set_target_properties(kokkos_sample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
