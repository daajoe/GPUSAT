cmake_minimum_required(VERSION 3.10)
project(gpusat CXX CUDA)

option(WITH_CLI "Build the GPUSAT cli interface" ON)

find_library(HTD_LIB htd)
find_library(HTD_IO_LIB htd_io)
add_library(htd STATIC IMPORTED)
add_library(htd_io STATIC IMPORTED)
set_target_properties(htd PROPERTIES IMPORTED_LOCATION ${HTD_LIB})
set_target_properties(htd_io PROPERTIES IMPORTED_LOCATION ${HTD_IO_LIB})

find_package(Boost REQUIRED)

file(GLOB gpusat_SRC "src/*.cpp")
add_library(gpusat STATIC ${gpusat_SRC})
target_compile_features(gpusat PUBLIC cxx_std_17)
target_link_libraries(gpusat htd htd_io gpusat_kernel -static-libgcc -static-libstdc++)
target_include_directories(gpusat PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

file(GLOB_RECURSE gpusat_HEADERS "include/*.h")

# TODO:  -Wconversion -Wsign-conversion -pedantic-errors
set(COMMON_FLAGS -Wextra -Wall -Werror)

add_library(gpusat_kernel STATIC "src/kernel.cu")
target_compile_options(gpusat_kernel PRIVATE -arch=sm_75)
target_compile_options(gpusat_kernel PRIVATE -std=c++17 --expt-relaxed-constexpr)
target_include_directories(gpusat_kernel PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(gpusat_kernel PRIVATE include/)
target_compile_options(gpusat_kernel PRIVATE $<$<CONFIG:Debug>:-lineinfo>)

foreach (flag IN LISTS COMMON_FLAGS)
    target_compile_options(gpusat_kernel PRIVATE --compiler-options=${flag})
endforeach()

#target_compile_options(gpusat PRIVATE $<$<CONFIG:Debug>:-fsanitize=leak>)
#target_compile_options(gpusat PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
target_include_directories(gpusat PUBLIC include/)

set_target_properties(gpusat PROPERTIES PUBLIC_HEADER "${gpusat_HEADERS}")

include(GNUInstallDirs)

install(
    TARGETS gpusat
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

target_compile_options(gpusat PRIVATE -fdiagnostics-color=always)

if (WITH_CLI)
    add_executable(gpusat_cli "src/cli/main.cpp")
    target_include_directories(gpusat_cli PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_features(gpusat_cli PUBLIC cxx_std_11)
    target_link_libraries(gpusat_cli gpusat  -static-libgcc -static-libstdc++)
    # FIXME: Remove the cuda dependency in the gpusat.h header
    target_include_directories(gpusat_cli PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

