# gpusat

A SAT solver based on dynamic programming running on the GPU.

## Dependencies

* OpenCL 1.2

## Build

### Dependencies

* cmake 3.2
* gcc 5.4
* AMD APP SDK 3.0/CUDA Toolkit 9.1 or newer
* Boost Multiprecision 1.66

### Compilation

To build the program with normal double precision use `cmake -Ddouble=ON` and the `make`, to build with double4 precision use `cmake` and the `make
The double4 Type is an adaption of [https://github.com/scibuilder/QD]

## Usage

First one needs the generate the primal, incidence or dual graph of the formula and generate the tree decomposition of the graph with a decomposer such as htd [https://github.com/mabseher/htd].

`gpusat [-f <treedecomp>] -s <formula> [-w <width>] [-m <size>] [-c <kerneldir>] [-h]`

The tree decomposition can either be provided via file or via input stream.

*    --decomposition,-f <treedecomp> : <treedecomp> path to the file containing the tree decomposition\n"
*    --formula,-s <formula>          : <formula> path to the file containing the sat formula\n"
*    --combineWidth,-w <width>       : <width> maximum width to combine bags of the decomposition\n"
*    --maxBagSize,-m <size>          : <size> maximum size of a bag before splitting it\n"
*    --kernelDir,-c <kerneldir>      : <kerneldir> path to the directory containing the kernel files\n"
*    --help,-h                       : prints this message\n";
*    --weighted                      : calculates the weighted model count of the formula