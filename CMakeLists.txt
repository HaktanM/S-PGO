cmake_minimum_required(VERSION 3.10)

# Project name and CUDA support
project(LieCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)



# Fİrst, we need to find python and path to torch so that we can include torch 
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Run python early and set as cache variable forcibly
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH} CACHE PATH "PyTorch CMake prefix path" FORCE)


# Find the OpenCV Eigen Torch
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)



# =========================================
#             CUDA Settings
# =========================================

# Find CUDA package
cmake_policy(SET CMP0104 NEW)  # For CUDA_ARCHITECTURES
find_package(CUDAToolkit REQUIRED)

# Set CMake CUDA flags. This is required to include __device__ functions in different files.
set(CMAKE_CUDA_ARCHITECTURES "native")    # Use the architecture of the local machine
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


# =========================================
#            Cython Settings
# =========================================
find_program(CYTHON_EXECUTABLE cython REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

if(NOT DEFINED NumPy_INCLUDE_DIR)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)  # Numpy expects a depticated version definition.

set(CYTHON_MODULE_NAME Solver)                                                 # Give a name to our Cython Module
set(CYTHON_SRC ${CMAKE_SOURCE_DIR}/src/CythonAPI/${CYTHON_MODULE_NAME}.pyx)     # Path to cython codes
set(CYTHON_GEN_CPP ${CMAKE_BINARY_DIR}/${CYTHON_MODULE_NAME}.cpp)               # Where to save the created cpp file


add_custom_command(
    OUTPUT ${CYTHON_GEN_CPP}
    COMMAND ${CYTHON_EXECUTABLE} --cplus -3 ${CYTHON_SRC} -o ${CYTHON_GEN_CPP}
    DEPENDS ${CYTHON_SRC}
    COMMENT "Running Cython on ${CYTHON_SRC}"
    VERBATIM
)
# DEPENDS ${CYTHON_SRC} -> If Solver.pyx has been modified or updated, CMake will rerun the command to regenerate Solver.cpp.
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
    ${PYTHON3_INCLUDE_DIRS}
    ${NumPy_INCLUDE_DIR}
)

# Add CUDA source files separately using target_sources
target_sources(${CYTHON_MODULE_NAME} PRIVATE 
    src/LieUtils/utils.cu
    src/LieUtils/SO3.cu
    src/LieUtils/SE3.cu
    src/Optimization/LMutils.cu  
    src/Optimization/Solver_cuda.cu 
    src/Optimization/Solver.cpp
)

# Link necesssary libraries
target_link_libraries(${CYTHON_MODULE_NAME} PRIVATE 
${Python3_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${TORCH_LIBRARIES} 
    Python3::Python
    CUDA::cublas
)