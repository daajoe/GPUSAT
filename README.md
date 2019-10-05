# gpusat

A #SAT solver based on dynamic programming running on the GPU.

## Dependencies

* htd 1.2.0 - https://github.com/mabseher/htd (build with cmake option -DBUILD_SHARED_LIBS=OFF)
* OpenCL 1.2
* cmake 3.2+
* gcc 7+
* Boost Multiprecision

### Tested with

* htd 1.2.0
* ocl-icd-opencl-dev 2.2.11-1ubuntu1
* AMD APP SDK 3.0/CUDA Toolkit 9.1
* cmake 3.9.1
* gcc 7.2.0
* Boost Multiprecision 1.66

## How to Build

The following commands can be used to build `GPUSAT`
```
git clone https://github.com/Budddy/GPUSAT
cd GPUSAT
mkdir build && cd build
cmake ..
make
```

## Usage

A program call is of the following form

./gpusat [-h] [-s \<seed\>] [-f \<formulaPath\>] [-d \<decompositionPath\>] [-n \<numberOfDecompositions\>] [--fitnessFunction \<fitness\>] [--CPU] [--NVIDIA] [--AMD] [--weighted] [--noExp] [--dataStructure \<dataStructure\>] [-m \<maxBagSize\>] [-w \<combineWidth\>] < $CNFFILE


Options:
*  -h,--help: Print this help message and exit.
*  -s,--seed \<seed\>: number used to initialize the pseudorandom number generator.
*  -f,--formula \<formulaPath\>: path to the file containing the sat formula.
*  -d,--decomposition \<decompositionPath\>: path to the file containing the tree decomposition.
*  -n,--numDecomps \<numberOfDecompositions\>: number of decompositions to generate for the fitness function.
*  --fitnessFunction \<function\>: use the fitnessfunction to optimize the tree decomposition.
    * Permitted Values:
        * numJoin: minimize the number of joins
        * joinSize: minimize the numer of variables in a join node
        * width_cutSet: minimize the width and then the cut set size (default)
        * cutSet_width: minimize the cut set size and then the width
*  --CPU: run the solver on a cpu
*  --NVIDIA: run the solver on an NVIDIA device
*  --AMD: run the solver on an AMD device
*  --weighted: use weighted model count
*  --noExp: don't use extended exponents
*  --dataStructure \<dataStructure\>: data structure for storing the solution.
    * Permitted Values:
        * array: use an array to store the solutions
        * tree: use a tree structure to store the solutions
        * combined: use a combination of tree and array structure (default)
*  -m,--maxBagSize \<maxBagSize\>: fixes the number of variables after which we need to split the bags
*  -w,--combineWidth \<combineWidth\>: fixes the maximum width to combine bags of the decomposition
