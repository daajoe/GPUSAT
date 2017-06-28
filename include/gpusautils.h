#ifndef GPUSAT_GPUSAUTILS_H
#define GPUSAT_GPUSAUTILS_H

#include <queue>
#include <main.h>

/**
 * function to print the tree decomposition in human readable form
 *
 * @param decomp
 *      the tree decomposition to print
 */
void printTreeD(treedecType decomp);

/**
 * function to print the solution for each bag in tree decomposition in human readable form
 *
 * @param decomp
 *      the tree decomposition containing the solutions
 */
void printSolutions(treedecType decomp);

/**
 * function to print the sat formula in human readable form
 *
 * @param formula
 *      the sat formula to print
 */
void printFormula(satformulaType formula);

/**
 * function to read a file into a string
 * @param path
 *      the path of the file
 * @return
 *      the contents of the file
 */
std::string readFile(std::string path);

/**
 * function to print the solution of a single bag in tree decomposition in human readable form
 *
 * @param numS
 *      the number of assignments
 * @param numVariables
 *      the number of variables
 * @param vars
 *      arrray containing the variables
 * @param sol
 *      array ontaining the number of models for each assignment
 */
void printSol(cl_long numS, cl_long numVariables, const cl_long *vars, const cl_long *sol);

#endif //GPUSAT_GPUSAUTILS_H
