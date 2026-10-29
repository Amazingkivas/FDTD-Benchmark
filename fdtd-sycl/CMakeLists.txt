if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(FATAL_ERROR "The SYCL backend requires Intel oneAPI DPC++ (icpx). Configure with -DCMAKE_CXX_COMPILER=icpx.")
endif()

add_library(fdtd_sycl_lib STATIC src/FDTD_sycl.cpp)
target_include_directories(fdtd_sycl_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(fdtd_sycl_lib PUBLIC -fsycl -O2 -DNDEBUG)
target_link_options(fdtd_sycl_lib PUBLIC -fsycl)

add_executable(run_sycl main/main.cpp)
target_link_libraries(run_sycl PRIVATE fdtd_sycl_lib)
set_target_properties(run_sycl PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
