cmake_minimum_required(VERSION 3.7)
project(LogicSem)

include_directories("C:/Program Files (x86)/AMD APP SDK/3.0/include" "C:/Users/Markus/AMD APP SDK/3.0/include")
link_directories("C:/Program Files (x86)/AMD APP SDK/3.0/bin/x86_64" "C:/Program Files (x86)/AMD APP SDK/3.0/lib/x86_64")
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")# -Wl,--stack,4194304")

file(GLOB LogicSem_SRC
        "include/*.h"
        "src/*.cpp"
        )
add_executable(LogicSem ${LogicSem_SRC})
target_link_libraries(LogicSem OpenCL)