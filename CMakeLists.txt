cmake_minimum_required(VERSION 3.2)
project(gpusat)

if (DEFINED ENV{AMDAPPSDKROOT})
    # add amd opencl libraries if AMDAPPSDKROOT environment variable is set
    include_directories("$ENV{AMDAPPSDKROOT}/include")
    link_directories("$ENV{AMDAPPSDKROOT}/lib/x86_64")
elseif (DEFINED ENV{CUDA_PATH})
    # add nvidia opencl libraries if CUDA_PATH environment variable is set
    include_directories("$ENV{CUDA_PATH}/include")
    link_directories("$ENV{CUDA_PATH}/lib/x64")
endif ()

include_directories(${CMAKE_SOURCE_DIR}/include)

if (DEBUG)
    # compiler flags
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
    # linker flags
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
        #
    add_definitions(-DDEBUG)
elseif (UNIX)
    # compiler flags
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
    # linker flags
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
endif ()

file(GLOB gpusat_SRC "src/*.cpp")
add_executable(gpusat ${gpusat_SRC})
target_link_libraries(gpusat OpenCL)
