#ifndef GPUSAT_MAIN_H
#define GPUSAT_MAIN_H
#define alloca __builtin_alloca

#include <CL/cl.hpp>

/// type for a bag in the tree decomposition
struct bagType {
    cl_long numVars = 0;
    cl_long numEdges = 0;
    cl_long numSol = 0;
    cl_long *variables;
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
    cl_long numVar = 0;
    cl_long numclauses = 0;
    cl_long totalNumVar = 0;
    cl_long *numVarsC;
    cl_long *clauses;
};

/**
 *
 * @param decomp
 * @param formula
 * @param node
 */
void solveProblemCPU(treedecType decomp, satformulaType formula, bagType node);

/**
 *
 * @param decomp
 * @param formula
 * @param node
 */
void solveProblem(treedecType decomp, satformulaType formula, bagType node);

/**
 *
 * @param decomp
 * @param solution
 * @param node
 */
void genSolution(treedecType decomp, cl_long *solution, bagType node);

/**
 *
 * @param node
 * @param edge1
 * @param edge2
 */
void solveJoin(bagType &node, bagType &edge1, bagType &edge2);

/**
 *
 * @param node
 * @param edge
 */
void solveForget(bagType &node, bagType &edge);

/**
 *
 * @param formula
 * @param node
 */
void solveLeaf(satformulaType &formula, bagType &node);

/**
 *
 * @param formula
 * @param node
 * @param edge
 */
void solveIntroduce(satformulaType &formula, bagType &node, bagType &edge);

/**
 *
 * @param decomp
 * @param solution
 * @param lastNode
 * @param lastId
 * @param edge
 */
void genSolEdge(treedecType decomp, cl_long *solution, bagType lastNode, cl_long lastId, int edge);

#endif //GPUSAT_MAIN_H
