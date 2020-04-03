cmake_minimum_required(VERSION 3.5)
project(GGNN)
# Options.
set(CMAKE_CXX_STANDARD 11)
enable_language(CUDA)
FIND_PACKAGE(CUDA REQUIRED)
find_package(Threads REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
set(OBLIB /usr/local/blas/OpenBLAS/lib/libopenblas.a)
set(CBLIB /usr/local/blas/CBLAS/lib/cblas_LINUX.a)
set(CBINC /usr/local/blas/CBLAS/include)
include_directories(${CBINC})


add_subdirectory(Matrix)
add_subdirectory(GPU-Computation)

# Add the executable.
aux_source_directory(. SRC)
add_executable(ggnn ${SRC})

target_link_libraries(ggnn matrix CuMatrix ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
