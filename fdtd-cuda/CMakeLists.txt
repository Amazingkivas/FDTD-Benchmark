if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA language support was not enabled.")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_executable(run_cuda
    src/FDTD_cuda.cu
    main/main.cu
)

target_include_directories(run_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(run_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)
target_link_libraries(run_cuda PRIVATE CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})

set_target_properties(run_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
