#ifndef GPUSAT_TYPES_H_H
#define GPUSAT_TYPES_H_H

#define alloca __builtin_alloca

#include <CL/cl.hpp>
#include <CL/cl_platform.h>
#include <cmath>
#include <list>
#include <set>

typedef struct {
    cl_double x[4];
} d4_Type;

#ifdef sType_Double
#define solType double
#else
#define solType d4_Type
#endif

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

    /// type for preprocessing a tree decomposition
    struct preebagType {
        cl_long id = 0;
        cl_long numEdges = 0;
        cl_long numVariables = 0;
        cl_long *variables = nullptr;
        preebagType **edges = nullptr;
    };

    /// type for saving a tree decomposition
    struct preetreedecType {
        cl_long numb = 0;
        preebagType *bags = nullptr;
    };

    bool compTreedType(const preebagType *a, const preebagType *b);

    /// type for saving the sat formula
    struct satformulaType {
        cl_long numclauses = 0;
        cl_long totalNumVar = 0;
        cl_long *numVarsC = nullptr;
        cl_long *clauses = nullptr;
    };
}

#endif //GPUSAT_TYPES_H_H
