cmake_minimum_required(VERSION 3.0)
project(fpfh_example)

find_package(PCL 1.10 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(fpfh_example src/test.cpp)
target_link_libraries(fpfh_example ${PCL_LIBRARIES})
