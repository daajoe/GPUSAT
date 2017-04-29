#include <math.h>
#include <regex>
#include <fstream>
#include <iostream>
#include <main.h>
#include <gpusautils.h>

void printTreeD(treedecType decomp) {
    cl_int size = decomp.numb;
    for (int i = 0; i < size; i++) {
        std::cout << "\nbagnum: " << i + 1 << "\n";
        cl_int vsize = decomp.bags[i].numVars;
        std::cout << "vertices: ";
        for (int a = 0; a < vsize; a++) {
            std::cout << decomp.bags[i].variables[a] << " ";
        }
        std::cout << "\n";
        cl_int esize = decomp.bags[i].numEdges;
        std::cout << "edges: ";
        for (int a = 0; a < esize; a++) {
            std::cout << decomp.bags[i].edges[a] << " ";
        }
        std::cout << "\n";
    }
}

void printSolutions(treedecType decomp) {
    cl_int size = decomp.numb;
    for (int i = 0; i < size; i++) {
        std::cout << "\nbagnum: " << i + 1 << "\n";
        cl_int numS = decomp.bags[i].numSol;
        cl_int numVariables = decomp.bags[i].numVars;
        cl_int *vars = decomp.bags[i].variables;
        std::cout << "solutions: \n";
        for (int a = 0; a < numS; a++) {
            std::cout << a << ": ";
            for (int b = 0; b < numVariables; b++) {
                std::cout << vars[b] * ((a & (1 << (numVariables - b - 1))) >> (numVariables - b - 1) == 0 ? -1 : 1)
                          << " ";
            }
            std::cout << decomp.bags[i].solution[a] << "\n";
        }
        std::cout << "\n";
    }
}

void printFormula(satformulaType formula) {
    cl_int size = formula.numclauses;
    int numVar = 0;
    for (int i = 0; i < size; i++) {
        std::cout << "\nclause: " << i + 1 << "\n";
        cl_int vsize = formula.numVarsC[i];
        std::cout << "variables: ";
        for (int a = 0; a < vsize; a++) {
            std::cout << formula.clauses[numVar + a] << " ";
        }
        numVar += vsize;
        std::cout << "\n";
    }
}

std::string readFile(std::string path) {
    std::stringbuf treeD;
    std::string inputLine;
    std::ifstream fileIn(path);
    while (getline(fileIn, inputLine)) {
        treeD.sputn(inputLine.c_str(), inputLine.size());
        treeD.sputn("\n", 1);
    }
    return treeD.str();
}