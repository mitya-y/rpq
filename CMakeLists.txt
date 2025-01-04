cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)

project(rpq)

add_executable(${CMAKE_PROJECT_NAME} "")

option(RPQ_RUN_ON_CPU "Run RPQ algorithm on CPU" OFF)

if(RPQ_RUN_ON_CPU)
    set(CUBOOL_WITH_CUDA OFF)
    set(CUBOOL_WITH_SEQUENTIAL ON)
endif()

add_subdirectory(cuBool)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC cuBool/cubool/include/cubool)

target_link_directories(${CMAKE_PROJECT_NAME} PUBLIC build/cuBool/cubool)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC cubool)

target_sources(${CMAKE_PROJECT_NAME} PUBLIC main.cpp)
target_sources(${CMAKE_PROJECT_NAME} PUBLIC regular_path_query.cpp)
target_sources(${CMAKE_PROJECT_NAME} PUBLIC benchmark.cpp)

# load .mtx format utility
add_subdirectory(fast_matrix_market)
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC fast_matrix_market/include)

file(COPY test_data DESTINATION .)
