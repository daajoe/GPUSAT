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
typedef long varIdType;
/// type for variables in the sat clauses
typedef long clauseIdType;

enum nodeType {
    JOIN, INTRODUCE, FORGET, LEAF
};


struct solutionType {
    long n = 0;
    varIdType *vars;
};

/// type for a bag in the tree decomposition
struct bagType {
    bagIdType id = 0;
    vertexIdType numv = 0;
    bagIdType nume = 0;
    long numSol = 0;
    vertexIdType *vertices;
    bagIdType *edges;
    solutionType *solution;
};

/// type for saving a tree decomposition
struct treedecType {
    bagIdType numb = 0;
    bagType *bags;
};

/// type for a clause in the sat formula
struct clauseType {
    varIdType numVars = 0;
    varIdType *var;
};

/// type for saving the sat formula
struct satformulaType {
    varIdType numVars = 0;
    clauseIdType numclauses = 0;
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

void printSolutions(treedecType decomp);

/**
 * function to read a file into the string
 * @param path the path to the file
 * @return the contents of the file
 */
string readFile(string path);

#endif //LOGICSEM_UTILS_H
