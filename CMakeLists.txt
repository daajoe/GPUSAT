 cmake_minimum_required(VERSION 3.2)
project(gpusat)
include_directories(/home/roemar/Ba/htd/include)

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
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -O0 -Wno-ignored-attributes")
    # linker flags
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    add_definitions(-DDEBUG)
elseif (UNIX)
    # compiler flags
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wno-ignored-attributes")
    # linker flags
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
endif ()

find_library(HTD_LIB htd ../../htd/lib)
add_library(htd SHARED IMPORTED)
set_target_properties(htd PROPERTIES IMPORTED_LOCATION ${HTD_LIB})

find_library(HTD_IO_LIB htd_io ../../htd/lib)
add_library(htd_io SHARED IMPORTED)
set_target_properties(htd_io PROPERTIES IMPORTED_LOCATION ${HTD_IO_LIB})

file(GLOB gpusat_SRC "src/*.cpp" "src/*/*.cpp")
add_executable(gpusat ${gpusat_SRC})
target_link_libraries(gpusat OpenCL htd htd_io)
