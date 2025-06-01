cmake_minimum_required(VERSION 3.10)

# Project name and CUDA support
project(LieCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find CUDA package
find_package(CUDA REQUIRED)

# Find the OpenCV Library
find_package (OpenCV REQUIRED)


# Set CMake CUDA flags. This is required to include __device__ functions in different files.
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# Add executables
add_executable(dene 
    src/main.cpp
    tests/testExpSO3.cpp
    tests/testLogSO3.cpp
    tests/testJacobiansSO3.cpp
    tests/testSE3.cpp
)

# Add CUDA source files separately using target_sources
target_sources(dene PRIVATE 
    src/LiuUtils/utils.cu
    src/LiuUtils/SO3.cu
    src/LiuUtils/SE3.cu
    tests/tests.cu
)

# Link necesssary libraries
target_link_libraries(dene PRIVATE 
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)