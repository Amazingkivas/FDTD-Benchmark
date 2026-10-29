add_library(fdtd_openmp
    src/FDTD/FDTD.cpp
    src/FDTD/FDTD_PML.cpp
    src/FDTD/test_FDTD.cpp
)

target_include_directories(fdtd_openmp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/FDTD
)

target_link_libraries(fdtd_openmp PUBLIC OpenMP::OpenMP_CXX)

add_executable(test_FDTD_method unit-tests/test_FDTD_method.cpp)
target_link_libraries(test_FDTD_method PRIVATE fdtd_openmp gtest_main)
add_test(NAME fdtd-openmp-unit-tests COMMAND test_FDTD_method)

add_executable(sample perf-tests/sample/sample.cpp)
target_link_libraries(sample PRIVATE fdtd_openmp)
set_target_properties(sample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
