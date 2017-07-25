#ifndef GPUSAT_MAIN_H
#define GPUSAT_MAIN_H
#define alloca __builtin_alloca

#include <CL/cl.hpp>

#define solType cl_double

/// type for a bag in the tree decomposition
struct bagType {
    cl_long numVars = 0;
    cl_long numEdges = 0;
    cl_long numSol = 0;
    cl_long *variables;
    cl_long *edges;
    solType *solution;
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
 * function to solve the sat problem
 *
 * @param decomp
 *      the tree decomposition
 * @param formula
 *      the sat formula
 * @param node
 *      the node to start from in the tree decompostion
 */
void solveProblem(treedecType &decomp, satformulaType &formula, bagType &node);

/**
 * function to solve a join node
 *
 * @param node
 *      the node to save the solutions in
 * @param edge1
 *      the first edge
 * @param edge2
 *      the second edge
 */
void solveJoin(bagType &node, bagType &edge1, bagType &edge2);

/**
 * function to solve a forget node
 *
 * @param node
 *      the node to save the solutions in
 * @param edge
 *      the next node
 */
void solveForget(bagType &node, bagType &edge);

/**
 * function to solve a leaf node
 *
 * @param formula
 *      the sat formula
 * @param node
 *      the node to save the solutions in
 */
void solveLeaf(satformulaType &formula, bagType &node);

/**
 * function to solve a introduce node
 *
 * @param formula
 *      the sat formula
 * @param node
 *      the node to save the solutions in
 * @param edge
 *      the next node
 */
void solveIntroduce(satformulaType &formula, bagType &node, bagType &edge);

#endif //GPUSAT_MAIN_H
