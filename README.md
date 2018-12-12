# gpusat

A #SAT solver based on dynamic programming running on the GPU.

## Dependencies

* OpenCL 1.2

## Build

### Tested with

* cmake 3.9.1
* gcc 7.2.0
* AMD APP SDK 3.0/CUDA Toolkit 9.1
* Boost Multiprecision 1.66

### Compilation

To build the program with normal double precision use `cmake` and the `make`.

## Usage

* -h,--help                   Print this help message and exit
* -s,--seed INT               path to the file containing the sat formula
* -f,--formula TEXT           path to the file containing the sat formula
* -d,--decomposition TEXT     path to the file containing the tree decomposition
* -n,--numDecomps UINT=30     
* --fitnessFunction TEXT 
    * fitness functions:  
    * **numJoin**: minimize the number of joins  
    * **joinSize**: minimize the numer of variables in a join node  
    * **width_cutSet**: minimize the width and then the cut set size  
    * **cutSet_width**: minimize the cut set size and then the width  
* --CPU                       run the solver on the cpu
* --NVIDIA                    run the solver on an NVIDIA device
* --AMD                       run the solver on an AMD device
* --weighted                  use weighted model count
* --noExp                     don't use extended exponents
* --dataStructure TEXT in {array,tree}=tree
                              choose a data structure for storing the solutions: