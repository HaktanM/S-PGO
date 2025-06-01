cmake_minimum_required(VERSION 3.10)

# Project name and CUDA support
project(LieCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the OpenCV Eigen
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)


# =========================================
#             CUDA Settings
# =========================================

# Find CUDA package
cmake_policy(SET CMP0104 NEW)  # For CUDA_ARCHITECTURES

# Set CMake CUDA flags. This is required to include __device__ functions in different files.
set(CMAKE_CUDA_ARCHITECTURES "native")    # Use the architecture of the local machine
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


# =========================================
#            Cython Settings
# =========================================
find_program(CYTHON_EXECUTABLE cython REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(CYTHON_MODULE_NAME Manager)                                                 # Give a name to our Cython Module
set(CYTHON_SRC ${CMAKE_SOURCE_DIR}/src/CythonAPI/${CYTHON_MODULE_NAME}.pyx)     # Path to cython codes
set(CYTHON_GEN_CPP ${CMAKE_BINARY_DIR}/${CYTHON_MODULE_NAME}.cpp)               # Where to save the created cpp file


add_custom_command(
    OUTPUT ${CYTHON_GEN_CPP}
    COMMAND ${CYTHON_EXECUTABLE} --cplus -3 ${CYTHON_SRC} -o ${CYTHON_GEN_CPP}
    DEPENDS ${CYTHON_SRC}
    COMMENT "Running Cython on ${CYTHON_SRC}"
    VERBATIM
)
# DEPENDS ${CYTHON_SRC} -> If manager.pyx has been modified or updated, CMake will rerun the command to regenerate manager.cpp.
# VERBATIM ->  ensures that the command arguments are passed exactly as written. It's a way to make sure CMake doesn't modify or interpret any part of the command unintentionally.



add_library(${CYTHON_MODULE_NAME} MODULE
    ${CYTHON_GEN_CPP}
    ${CMAKE_SOURCE_DIR}/include
)

# Set output file to be a .so and properly named
set_target_properties(${CYTHON_MODULE_NAME} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${CYTHON_MODULE_NAME}"
    SUFFIX ".so"
)

target_include_directories(${CYTHON_MODULE_NAME} PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Add CUDA source files separately using target_sources
target_sources(${CYTHON_MODULE_NAME} PRIVATE 
    src/LieUtils/utils.cu
    src/LieUtils/SO3.cu
    src/LieUtils/SE3.cu
    src/Optimization/Manager.cpp
)

# Link necesssary libraries
target_link_libraries(${CYTHON_MODULE_NAME} PRIVATE 
${Python3_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
)