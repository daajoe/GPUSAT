#include <math.h>
#include <regex>
#include <fstream>
#include <iostream>
#include <main.h>
#include <gpusautils.h>
#include <stdio.h>

namespace gpusat {

    void GPUSATUtils::printTreeD(treedecType decomp) {
        cl_long size = decomp.numb;
        for (int i = 0; i < size; i++) {
            std::cout << "\nbagnum: " << i + 1 << "\n";
            cl_long vsize = decomp.bags[i].numVars;
            std::cout << "vertices: ";
            for (int a = 0; a < vsize; a++) {
                std::cout << decomp.bags[i].variables[a] << " ";
            }
            std::cout << "\n";
            cl_long esize = decomp.bags[i].numEdges;
            std::cout << "edges: ";
            for (int a = 0; a < esize; a++) {
                std::cout << decomp.bags[i].edges[a] << " ";
            }
            std::cout << "\n";
        }
    }

    void GPUSATUtils::printSolutions(treedecType decomp) {
        cl_long size = decomp.numb;
        for (int i = 0; i < size; i++) {
            std::cout << "\nbagnum: " << i + 1 << "\n";
            cl_long numS = decomp.bags[i].numSol;
            cl_long numVariables = decomp.bags[i].numVars;
            cl_long *vars = decomp.bags[i].variables;
            solType *sol = decomp.bags[i].solution;
            printSol(numS, numVariables, vars, sol);
            std::cout << "\n";
        }
    }

    void GPUSATUtils::printSol(cl_long numS, cl_long numVariables, const cl_long *vars, const solType *sol) {
        std::cout << "solutions: \n";
        for (int a = 0; a < numS; a++) {
            std::cout << a << ": ";
            for (int b = 0; b < numVariables; b++) {
                std::cout << (((a >> b) & 1) == 0 ? "-" : " ") << vars[b]
                          << " ";
            }
            printf("%i\n", sol[a]);
        }
    }

    void GPUSATUtils::printFormula(satformulaType formula) {
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

    std::string GPUSATUtils::readFile(std::string path) {
        std::stringbuf treeD;
        std::string inputLine;
        std::ifstream fileIn(path);
        while (getline(fileIn, inputLine)) {
            treeD.sputn(inputLine.c_str(), inputLine.size());
            treeD.sputn("\n", 1);
        }
        return treeD.str();
    }

    std::string GPUSATUtils::readBinary(std::string path) {
        FILE *input = fopen(path.c_str(), "rb");
        fseek(input, 0L, SEEK_END);
        size_t size = ftell(input);
        char *binary = (char *) malloc(size);
        rewind(input);
        fread(binary, sizeof(char), size, input);
        fclose(input);
        std::string buffer;
        buffer.assign(binary, size);
        return buffer;
    }

    void GPUSATUtils::writeBinary(char *data, size_t size, std::string path) {
        std::ofstream fileOut(path, std::ios::binary);
        fileOut.write(data, size);
    }
}