cmake_minimum_required(VERSION 3.20)

project(thinning VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(PCL 1.8 REQUIRED)

find_package(Boost REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)

find_package(Threads)

include_directories(${PROJECT_SOURCE_DIR}/include)

message(STATUS "Found PCL: ${PCL_INCLUDE_DIRS}")
message(STATUS "Found Boost: ${Boost_INCLUDE_DIRS}")
message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")

add_executable(thinning src/thinning.cpp)


target_link_libraries (thinning PRIVATE ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(thinning PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(thinning PRIVATE ${PCL_LIBRARIES})

target_include_directories(thinning PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(thinning PRIVATE ${BOOST_LIBRARIES})
target_link_libraries(thinning PRIVATE yaml-cpp)

if(OpenMP_CXX_FLAGS)
    target_compile_options(thinning PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(thinning PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()