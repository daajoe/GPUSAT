#ifndef GPUSAT_MAIN_H
#define GPUSAT_MAIN_H

#define alloca __builtin_alloca

#include <CL/cl.hpp>

/// type for a bag in the tree decomposition
struct bagType {
    cl_long numv = 0;
    cl_long nume = 0;
    cl_long numSol = 0;
    cl_long *vertices;
    cl_long *edges;
    cl_long *solution;
};

/// type for saving a tree decomposition
struct treedecType {
    cl_long numb = 0;
    bagType *bags;
};

/// type for saving the sat formula
struct satformulaType {
    cl_long numclauses = 0;
    cl_long totalNumVar = 0;
    cl_long *numVarsC;
    cl_long *clauses;
};


void solveProblem(treedecType decomp, satformulaType formula, bagType node, cl::Context &context,
                  cl::Program &program, cl::CommandQueue &commandQueue);

#endif //GPUSAT_MAIN_H
