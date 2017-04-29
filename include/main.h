#ifndef GPUSAT_MAIN_H
#define GPUSAT_MAIN_H
#define alloca __builtin_alloca

#include <CL/cl.hpp>

/// type for a bag in the tree decomposition
struct bagType {
    cl_int numVars = 0;
    cl_int numEdges = 0;
    cl_int numSol = 0;
    cl_int *variables;
    cl_int *edges;
    cl_int *solution;
};

/// type for saving a tree decomposition
struct treedecType {
    cl_int numb = 0;
    bagType *bags;
};

/// type for saving the sat formula
struct satformulaType {
    cl_int numVar = 0;
    cl_int numclauses = 0;
    cl_int totalNumVar = 0;
    cl_int *numVarsC;
    cl_int *clauses;
};

void solveProblemCPU(treedecType decomp, satformulaType formula, bagType node);

void solveProblem(treedecType decomp, satformulaType formula, bagType node);

void genSolution(treedecType decomp, cl_int *solution, bagType node);

void solveJoin(treedecType &decomp, bagType &node);

void solveForget(treedecType &decomp, bagType &node);

void solveLeaf(satformulaType &formula, bagType &node);

void solveIntroduce(treedecType &decomp, satformulaType &formula, bagType &node);

#endif //GPUSAT_MAIN_H
