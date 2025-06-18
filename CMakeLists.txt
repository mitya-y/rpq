cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 23)

project(rpq)

option(RPQ_RUN_ON_CPU "Run RPQ algorithm on CPU" OFF)

if(RPQ_RUN_ON_CPU)
  set(CUBOOL_WITH_CUDA OFF)
  set(CUBOOL_WITH_SEQUENTIAL ON)
endif()

set(CUBOOL_USE_NSPARSE_MERGE_FUNCTOR OFF)

add_subdirectory(cuBoolGraph)

# load .mtx format utility
add_subdirectory(fast_matrix_market)

file(COPY parse_mem.py DESTINATION .)

# ------------------------------------------------
# add benchmark
# ------------------------------------------------

set(BENCHMARK_TARGET ${CMAKE_PROJECT_NAME}_bench)
add_executable(${BENCHMARK_TARGET} "")

target_link_libraries(${BENCHMARK_TARGET} PUBLIC cuboolgraph)

target_sources(${BENCHMARK_TARGET} PUBLIC benchmark.cpp)

# load .mtx format utility
target_include_directories(${BENCHMARK_TARGET} PUBLIC fast_matrix_market/include)

# thread pool library
target_include_directories(${BENCHMARK_TARGET} PUBLIC thread-pool/include)

# benchmark dataset options
# set(RPQ_BENCH_DATASET_PATH "/home/mitya/Documents/datasets/wikidata" CACHE STRING "Path to dataset")
# set(RPQ_BENCH_QUERY_COUNT "660" CACHE STRING "NumberCACHE STRING of queries (can be extra)")
# set(RPQ_BENCH_LABEL_COUNT "1395" CACHE STRING "Number of labels (can be extra)")

set(RPQ_BENCH_DATASET_PATH "/home/mitya/Documents/datasets/rpqbench" CACHE STRING "Path to dataset")
set(RPQ_BENCH_QUERY_COUNT "20000" CACHE STRING "Number of queries (can be extra)")
set(RPQ_BENCH_LABEL_COUNT "9" CACHE STRING "Number of labels (can be extra)")

# set(RPQ_BENCH_DATASET_PATH "/home/mitya/Documents/datasets/rpqbench-100kk" CACHE STRING "Path to dataset")
# set(RPQ_BENCH_QUERY_COUNT "2000" CACHE STRING "Number of queries (can be extra)")
# set(RPQ_BENCH_LABEL_COUNT "9" CACHE STRING "Number of labels (can be extra)")

target_compile_definitions(${BENCHMARK_TARGET} PUBLIC BENCH_DATASET_DIR="${RPQ_BENCH_DATASET_PATH}")
target_compile_definitions(${BENCHMARK_TARGET} PUBLIC BENCH_LABEL_COUNT=${RPQ_BENCH_LABEL_COUNT})
target_compile_definitions(${BENCHMARK_TARGET} PUBLIC BENCH_QUERY_COUNT=${RPQ_BENCH_QUERY_COUNT})
