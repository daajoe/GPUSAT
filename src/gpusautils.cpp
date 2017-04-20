#include <math.h>
#include <regex>
#include <fstream>
#include <iostream>
#include <main.h>
#include <gpusautils.h>

void printTreeD(treedecType decomp) {
    cl_long size = decomp.numb;
    for (int i = 0; i < size; i++) {
        std::cout << "\nbagnum: " << i + 1 << "\n";
        cl_long vsize = decomp.bags[i].numv;
        std::cout << "vertices: ";
        for (int a = 0; a < vsize; a++) {
            std::cout << decomp.bags[i].vertices[a] << " ";
        }
        std::cout << "\n";
        cl_long esize = decomp.bags[i].nume;
        std::cout << "edges: ";
        for (int a = 0; a < esize; a++) {
            std::cout << decomp.bags[i].edges[a] << " ";
        }
        std::cout << "\n";
    }
}

void printSolutions(treedecType decomp) {
    cl_long size = decomp.numb;
    for (int i = 0; i < size; i++) {
        std::cout << "\nbagnum: " << i + 1 << "\n";
        cl_long ns = decomp.bags[i].numSol;
        cl_long vsize = decomp.bags[i].numv;
        std::cout << "solutions: \n";
        for (int a = 0; a < ns; a++) {
            std::cout << a << ": ";
            for (int b = 0; b < vsize; b++) {
                std::cout << decomp.bags[i].solution[(decomp.bags[i].numv + 1) * a + b] << " ";
            }
            std::cout << decomp.bags[i].solution[(decomp.bags[i].numv + 1) * a + decomp.bags[i].numv] << "\n";
        }
        std::cout << "\n";
    }
}

void printFormula(satformulaType formula) {
    cl_long size = formula.numclauses;
    int numVar = 0;
    for (int i = 0; i < size; i++) {
        std::cout << "\nclause: " << i + 1 << "\n";
        cl_long vsize = formula.numVarsC[i];
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