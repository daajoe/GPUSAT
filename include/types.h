#ifndef GPUSAT_TYPES_H_H
#define GPUSAT_TYPES_H_H

#if defined __CYGWIN__ || defined __MINGW32__
#define alloca __builtin_alloca
#endif

#include <CL/cl_platform.h>
#include <cmath>
#include <list>
#include <vector>
#include <set>

namespace dual {
    /**
     * parameters for the dual join kernel
     */
    typedef struct {
        cl_long numC;
        cl_long numCE1;
        cl_long numCE2;
        cl_long minId1;
        cl_long maxId1;
        cl_long minId2;
        cl_long maxId2;
        cl_long startIDNode;
        cl_long startIDEdge1;
        cl_long startIDEdge2;
        cl_long numV;
        cl_long numVE1;
        cl_long numVE2;
    } sJVars;

    /**
     * parameters for the dual introduce forget kernel
     */
    typedef struct {
        cl_long numCI;
        cl_long numCE;
        cl_long numVF;
        cl_long numVI;
        cl_long numVE;
        cl_long combinations;
        cl_long minIdE;
        cl_long maxIdE;
        cl_long startIDF;
        cl_long startIDE;
        cl_long numCF;
    } sIFVars;
}
namespace gpusat {

    struct treeType {
        cl_long *elements = nullptr;
        cl_int numSolutions = 0;
        cl_long size = 0;
        cl_long minId = 0;
        cl_long maxId = 0;
    };

    /// type for a bag in the tree decomposition
    struct bagType {
        cl_long id = 0;
        std::vector<cl_long> variables;
        std::vector<bagType *> edges;
        cl_long bags = 0;
        treeType *solution;
    };

    /// type for saving a tree decomposition
    struct treedecType {
        cl_long numb = 0;
        cl_long numVars = 0;
        std::vector<bagType> bags;
    };

    /**
     * Function that compares two tree decompostions by id.
     *
     * @param a     the first tree decompostion
     * @param b     the second tree decomposition
     * @return      a < b
     */
    inline bool compTreedType(const bagType *a, const bagType *b) {
        return a->id < b->id;
    }

    /// type for saving the sat formula
    struct satformulaType {
        cl_long numVars = 0;
        cl_long numWeights = 0;
        bool unsat = false;
        cl_double *variableWeights = nullptr;
        std::vector<std::vector<cl_long>> clauses;
        std::vector<cl_long> facts;
    };

    /// the graph type which was the base for the tree decomposition
    enum graphTypes {
        PRIMAL, INCIDENCE, DUAL, NONE
    };

    /**
     * Function that compares two variables.
     *
     * @param a     the first variable
     * @param b     the second variable
     * @return      true if abs a < b
     */
    inline bool compVars(const cl_long &a, const cl_long &b) {
        return std::abs(a) < std::abs(b);
    }
}

#endif //GPUSAT_TYPES_H_H
