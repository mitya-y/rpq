cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)

project(rpq)

option(RPQ_RUN_ON_CPU "Run RPQ algorithm on CPU" OFF)

if(RPQ_RUN_ON_CPU)
  set(CUBOOL_WITH_CUDA OFF)
  set(CUBOOL_WITH_SEQUENTIAL ON)
endif()

set(CUBOOL_USE_NSPARSE_MERGE_FUNCTOR OFF)

add_subdirectory(cuBool)

# load .mtx format utility
add_subdirectory(fast_matrix_market)

file(COPY test_data DESTINATION .)
file(COPY parse_mem.py DESTINATION .)

# ------------------------------------------------
# add tests
# ------------------------------------------------
set(TESTS_TARGET ${CMAKE_PROJECT_NAME}_tests)
add_executable(${TESTS_TARGET} "")
target_include_directories(${TESTS_TARGET} PUBLIC cuBool/cubool/include/cubool)

target_link_directories(${TESTS_TARGET} PUBLIC build/cuBool/cubool)
target_link_libraries(${TESTS_TARGET} PUBLIC cubool)

target_sources(${TESTS_TARGET} PUBLIC tests.cpp)
target_sources(${TESTS_TARGET} PUBLIC regular_path_query.cpp)

# load .mtx format utility
target_include_directories(${TESTS_TARGET} PUBLIC fast_matrix_market/include)

# ------------------------------------------------
# add benchmark
# ------------------------------------------------
set(BENCHMARK_TARGET ${CMAKE_PROJECT_NAME}_bench)
add_executable(${BENCHMARK_TARGET} "")
target_include_directories(${BENCHMARK_TARGET} PUBLIC cuBool/cubool/include/cubool)

target_link_directories(${BENCHMARK_TARGET} PUBLIC build/cuBool/cubool)
target_link_libraries(${BENCHMARK_TARGET} PUBLIC cubool)

target_sources(${BENCHMARK_TARGET} PUBLIC regular_path_query.cpp)
target_sources(${BENCHMARK_TARGET} PUBLIC par_regular_path_query.cpp)
target_sources(${BENCHMARK_TARGET} PUBLIC benchmark.cpp)

# load .mtx format utility
target_include_directories(${BENCHMARK_TARGET} PUBLIC fast_matrix_market/include)

# thread pool library
target_include_directories(${BENCHMARK_TARGET} PUBLIC thread-pool/include)

set(RPQ_BENCH_WIKIDATA_PATH "/home/mitya/Documents/wiki-dataset/" CACHE STRING "Path to wiki-dataset")
target_compile_definitions(${BENCHMARK_TARGET} PUBLIC WIKIDATA_DIR="${RPQ_BENCH_WIKIDATA_PATH}")
