#gpusat
A SAT solver based on dynamic programming running on the GPU.

##Dependencies
* OpenCL 1.2
* AMD GPU

##Build
###Dependencies
* cmake 3.2
* gcc 5.4
* AMD APP SDK 3.0

###Compilation
For compilation use `cmake` and then `make`.

##Usage
`./gpusat [-f <treedecomp>] -s <formula>`

The treedecomposition can eather be provided via file or piped into the program.

* `-f <treedecomp>`: read the treedecomposition from the <treedecomp> file in td format
* `-s <formula>`: read the sat formula from the <formula> file in cnf format
