cmake_minimum_required(VERSION 3.2)
project(gpusat)

include_directories("C:/Program Files (x86)/AMD APP SDK/3.0/include" "/opt/AMDAPPSDK-3.0/include")
link_directories("C:/Program Files (x86)/AMD APP SDK/3.0/bin/x86_64" "/opt/AMDAPPSDK-3.0/lib/x86_64/sdk")
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")# -Wl,--stack,4194304")

file(GLOB gpusat_SRC
        "include/*.h"
        "src/*.cpp"
        )
add_executable(gpusat ${gpusat_SRC})
target_link_libraries(gpusat OpenCL)