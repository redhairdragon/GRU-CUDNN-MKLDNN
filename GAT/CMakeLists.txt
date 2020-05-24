cmake_minimum_required(VERSION 3.5)
enable_language(CUDA)
project(GAT)
# Options.
set(CMAKE_CXX_STANDARD 11)
FIND_PACKAGE(CUDA REQUIRED)
find_package(Threads REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
set(OBLIB /usr/local/blas/OpenBLAS/lib/libopenblas.a)
set(CBLIB /usr/local/blas/CBLAS/lib/cblas_LINUX.a)
set(CBINC /usr/local/blas/CBLAS/include)
include_directories(${CBINC})


add_subdirectory(Matrix)
add_subdirectory(GPU-Computation)

# Add the executable.
aux_source_directory(. SRC)
cuda_add_executable(gat ${SRC})
target_link_libraries(gat  matrix ComputingUnit CuMatrix ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})