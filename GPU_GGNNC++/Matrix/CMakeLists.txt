cmake_minimum_required(VERSION 3.5)

aux_source_directory(. COMMON_SRC)
add_library(matrix SHARED ${COMMON_SRC})
target_link_libraries(matrix PUBLIC ${OBLIB} ${CBLIB} Threads::Threads)
set_property(TARGET matrix PROPERTY POSITION_INDEPENDENT_CODE ON)
