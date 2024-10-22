cmake_minimum_required(VERSION 3.10)
project(rpq)

add_executable(${CMAKE_PROJECT_NAME} "")

add_subdirectory(cuBool)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC cuBool/cubool/include/cubool)

target_link_directories(${CMAKE_PROJECT_NAME} PUBLIC build/cuBool/cubool)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC cubool)

# target_sources(${CMAKE_PROJECT_NAME} PUBLIC main.cpp cuBool/cubool/include/cubool.h)
target_sources(${CMAKE_PROJECT_NAME} PUBLIC main.cpp)
