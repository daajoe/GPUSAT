#ifndef LOGICSEM_UTILS_H
#define LOGICSEM_UTILS_H

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <fstream>
#include <utils.h>

using namespace std;

/// id of the vertex in the primal graph
typedef long int vertexId;
/// id of the bag in the tree decomposition
typedef long int bagId;
/// type for variables in the sat clauses
typedef long int varType;

/// type for a bag in the tree decomposition
struct bag {
    int numv = 0;
    unsigned long long int nume = 0;
    vertexId *vertices;
    bagId *edges;
};

/// type for saving a tree decomposition
struct treedec {
    int numb;
    bag *bags;
};

/// type for a clause in the sat formula
struct clause {
    long int numVars;
    varType *var;
};

/// type for saving the sat formula
struct satformula {
    long int numclauses;
    clause *clauses;
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
void printTreeD(treedec decomp);

/**
 * function to read a file into the string
 * @param path the path to the file
 * @return the contents of the file
 */
string readFile(string path);

#endif //LOGICSEM_UTILS_H
