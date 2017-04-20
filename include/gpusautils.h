#ifndef GPUSAT_GPUSAUTILS_H
#define GPUSAT_GPUSAUTILS_H

#include <queue>
#include <main.h>

/**
 * function to print the tree decomposition in human readable form
 * @param decomp the tree decomposition to print
 */
void printTreeD(treedecType decomp);

void printSolutions(treedecType decomp);

/**
 * function to print the tree decomposition in human readable form
 * @param decomp the tree decomposition to print
 */
void printFormula(satformulaType formula);

/**
 * function to read a file into the string
 * @param path the path to the file
 * @return the contents of the file
 */
std::string readFile(std::string path);

#endif //GPUSAT_GPUSAUTILS_H
