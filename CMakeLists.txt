cmake_minimum_required(VERSION 3.8)
project(gpusati CXX CUDA)

include_directories("$ENV{CUDA_PATH}/include")
link_directories("$ENV{CUDA_PATH}/lib/x64")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(gpusat_kernel STATIC "src/kernel.cu")
target_compile_options(gpusat_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)

if (DEBUG)
    # compiler flags
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O3 -Wno-ignored-attributes")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --expt-relaxed-constexpr")
    # linker flags
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    add_definitions(-DDEBUG)
    message("Using debug mode!")
elseif (UNIX)
    # compiler flags
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wno-ignored-attributes")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --expt-relaxed-constexpr")
    # linker flags
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
endif ()

find_library(HTD_LIB htd)
add_library(htd STATIC IMPORTED)
set_target_properties(htd PROPERTIES IMPORTED_LOCATION ${HTD_LIB})

find_library(HTD_IO_LIB htd_io)
add_library(htd_io STATIC IMPORTED)
set_target_properties(htd_io PROPERTIES IMPORTED_LOCATION ${HTD_IO_LIB})

file(GLOB gpusat_SRC "src/*.cpp" "src/*/*.cpp")
add_executable(gpusat ${gpusat_SRC})
target_link_libraries(gpusat htd htd_io gpusat_kernel)
