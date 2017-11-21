cmake_minimum_required(VERSION 3.2)
project(gpusat)

include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/include" "/opt/AMDAPPSDK-3.0/include")
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/lib/x64" "/opt/AMDAPPSDK-3.0/lib/x86_64/sdk")
include_directories(${CMAKE_SOURCE_DIR}/include)

if (double)
    add_definitions(-DsType_Double)
endif (double)

if (debug)
    add_definitions(-DDEBUG)
endif (debug)

if (valgrind)
    # compiler flags
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O0")
    # linker flags
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
else (valgrind)
    # compiler flags
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
    # linker flags
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
endif (valgrind)

if (WIN32)
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif (WIN32)

file(GLOB gpusat_SRC "src/*.cpp")
add_executable(gpusat ${gpusat_SRC})
target_link_libraries(gpusat OpenCL)
