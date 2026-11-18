cmake_minimum_required(VERSION 3.18)
project(pgc_exp5_Neural_Network_Layers_In_Cuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add CPU validation executable
add_executable(cpu_validation src/cpu_validation.cpp)
if(MSVC)
    target_compile_options(cpu_validation PRIVATE /O2 /W4 /EHsc)
else()
    target_compile_options(cpu_validation PRIVATE -O3 -Wall -Wextra)
endif()

# Check for CUDA Toolkit support
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    add_executable(neural_network src/neural_network.cu)
    set_target_properties(neural_network PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
    )
    if(MSVC)
        target_compile_options(neural_network PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)
    else()
        target_compile_options(neural_network PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)
    endif()
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}. neural_network target enabled.")
else()
    message(STATUS "CUDA compiler not found. Only CPU validation target will be built.")
endif()
