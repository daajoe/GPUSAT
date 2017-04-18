#ifndef LOGICSEM_UTILS_H
#define LOGICSEM_UTILS_H

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <fstream>
#include <queue>

using namespace std;

/// id of the vertex in the primal graph
typedef long vertexIdType;
/// id of the bag in the tree decomposition
typedef long bagIdType;
/// type for variables in the sat clauses
typedef long varType;

/// type for a bag in the tree decomposition
struct bagType {
    long numv = 0;
    long nume = 0;
    vertexIdType *vertices;
    bagIdType *edges;
};

/// type for saving a tree decomposition
struct treedecType {
    long numb;
    bagType *bags;
};

/// type for a clause in the sat formula
struct clauseType {
    long numVars;
    varType *var;
};

/// type for saving the sat formula
struct satformulaType {
    long numVars;
    long numclauses;
    clauseType *clauses;
};

/**
 * counts the occurences of a single character in a string
 * @param str the string to search
 * @param c the character to find
 * @return the number of occurences
 */
int numCharOcc(string str, char c);

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

/**
 * function to read a file into the string
 * @param path the path to the file
 * @return the contents of the file
 */
string readFile(string path);

#endif //LOGICSEM_UTILS_H
