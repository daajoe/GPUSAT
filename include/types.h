#ifndef GPUSAT_TYPES_H_H
#define GPUSAT_TYPES_H_H

#define alloca __builtin_alloca

#include <CL/cl.hpp>

#define solType cl_float

namespace gpusat {

    /// type for a bag in the tree decomposition
    struct bagType {
        cl_long numVars = 0;
        cl_long numEdges = 0;
        cl_long numSol = 0;
        cl_long *variables = nullptr;
        cl_long *edges = nullptr;
        solType *solution = nullptr;
    };

    /// type for saving a tree decomposition
    struct treedecType {
        cl_long numb = 0;
        bagType *bags = nullptr;
    };

    /// type for saving the sat formula
    struct satformulaType {
        cl_long numclauses = 0;
        cl_long totalNumVar = 0;
        cl_long *numVarsC = nullptr;
        cl_long *clauses = nullptr;
    };

}

#endif //GPUSAT_TYPES_H_H
