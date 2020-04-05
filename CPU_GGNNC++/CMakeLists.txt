cmake_minimum_required(VERSION 3.5)
project(GGNN)
# Options.
set(CMAKE_CXX_STANDARD 11)
find_package(Threads REQUIRED)


find_library(DNNL_LIB libdnnl.so HINTS /usr/local/oneDNN/lib/)
set(DNNL_INC /usr/local/oneDNN/include/)
include_directories(${DNNL_INC})


set(OBLIB /usr/local/blas/OpenBLAS/lib/libopenblas.a)
set(CBLIB /usr/local/blas/CBLAS/lib/cblas_LINUX.a)
set(CBINC /usr/local/blas/CBLAS/include)
include_directories(${CBINC})


add_subdirectory(Matrix)

# Add the executable.
aux_source_directory(. SRC)
add_executable(ggnn ${SRC})

target_link_libraries(ggnn matrix ${DNNL_LIB})
