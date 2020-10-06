cmake_minimum_required(VERSION 3.10)
project(gpusat CXX CUDA)

option(WITH_CLI "Build the GPUSAT cli interface" ON)

include_directories(
    "${PROJECT_SOURCE_DIR}/include"
    "$ENV{CUDA_PATH}/include"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
#link_directories("$ENV{CUDA_PATH}/lib/x64")

find_library(HTD_LIB htd)
add_library(htd STATIC IMPORTED)
set_target_properties(htd PROPERTIES IMPORTED_LOCATION ${HTD_LIB})

find_package(Boost REQUIRED)

file(GLOB gpusat_SRC "src/*.cpp")
add_library(gpusat STATIC ${gpusat_SRC})
target_compile_features(gpusat PUBLIC cxx_std_17)
target_link_libraries(gpusat htd gpusat_kernel)

file(GLOB_RECURSE gpusat_HEADERS "include/*.h")

# TODO:  -Wconversion -Wsign-conversion -pedantic-errors
set(COMMON_FLAGS -Wextra -Wall -Werror)

add_library(gpusat_kernel STATIC "src/kernel.cu")
target_compile_options(gpusat_kernel PRIVATE -gencode arch=compute_75,code=sm_75)
target_compile_options(gpusat_kernel PRIVATE -std=c++17 --expt-relaxed-constexpr)

foreach (flag IN LISTS COMMON_FLAGS)
    target_compile_options(gpusat_kernel PRIVATE --compiler-options=${flag})
endforeach()

target_compile_options(gpusat PRIVATE $<$<CONFIG:Debug>:-fsanitize=leak>)
target_include_directories(gpusat PUBLIC include/)

set_target_properties(gpusat PROPERTIES PUBLIC_HEADER "${gpusat_HEADERS}")

include(GNUInstallDirs)

install(
    TARGETS gpusat
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if (WITH_CLI)
    add_executable(gpusat_cli "src/cli/main.cpp")
    target_include_directories(gpusat_cli PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_features(gpusat_cli PUBLIC cxx_std_11)
    target_link_libraries(gpusat_cli gpusat)
endif()

