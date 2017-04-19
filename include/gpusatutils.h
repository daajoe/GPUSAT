#ifndef LOGICSEM_UTILS_H
#define LOGICSEM_UTILS_H

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

/**
 * function to print the tree decomposition in human readable form
 * @param decomp the tree decomposition to print
 */
void printTreeD(treedecType decomp);

/**
 * function to print the tree decomposition in human readable form
 * @param decomp the tree decomposition to print
 */
void printFormula(satformulaType formula);

void printSolutions(treedecType decomp);

/**
 * function to read a file into the string
 * @param path the path to the file
 * @return the contents of the file
 */
std::string readFile(std::string path);

#endif //LOGICSEM_UTILS_H
