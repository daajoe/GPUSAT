#include <math.h>
#include <regex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <gpusautils.h>

namespace gpusat {

    void GPUSATUtils::printTreeD(treedecType decomp) {
        cl_long size = decomp.numb;
        for (int i = 0; i < size; i++) {
            std::cout << "\nbagnum: " << i + 1 << "\n";
            cl_long vsize = decomp.bags[i].variables.size();
            std::cout << "variables: ";
            for (int a = 0; a < vsize; a++) {
                std::cout << decomp.bags[i].variables[a] << " ";
            }
            std::cout << "\n";
            cl_long esize = decomp.bags[i].edges.size();
            std::cout << "edgeschild nodes: ";
            for (int a = 0; a < esize; a++) {
                std::cout << decomp.bags[i].edges[a] << " ";
            }
            std::cout << "\n";
        }
    }

    void GPUSATUtils::printTreeD(preetreedecType decomp) {
        cl_long size = decomp.numb;
        for (int i = 0; i < size; i++) {
            std::cout << "\nbagnum: " << i + 1 << "\n";
            cl_long vsize = decomp.bags[i].variables.size();
            std::cout << "variables: ";
            for (int a = 0; a < vsize; a++) {
                std::cout << decomp.bags[i].variables[a] << " ";
            }
            std::cout << "\n";
            cl_long esize = decomp.bags[i].edges.size();
            std::cout << "edgeschild nodes: ";
            for (int a = 0; a < esize; a++) {
                std::cout << decomp.bags[i].edges[a]->id + 1 << " ";
            }
            std::cout << "\n";
        }
    }

    void GPUSATUtils::printSol(cl_long numS, std::vector<cl_long> vars, solType **sol, satformulaType &formula, cl_long bagSize) {
        std::cout << "variables: \n";
        for (int j = 0; j < vars.size(); ++j) {
            std::cout << vars[j] << " ";
        }
        std::cout << "\nsolutions: \n";
        for (int i = 0; i < numS / bagSize; i++) {
            if (sol[i] == nullptr) {
                for (int a = 0; a < bagSize; a++) {
                    std::cout << i * bagSize + a << ": 0\n";
                }
                continue;
            }
            for (int a = 0; a < bagSize; a++) {
                std::cout << i * bagSize + a << ": ";
#ifdef sType_Double
                std::cout << sol[i][a] << "\n";
#else
                std::cout << d4_to_string(sol[i][a]) << "\n";
#endif
            }
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
        free(binary);
        return buffer;
    }

    void GPUSATUtils::writeBinary(char *data, size_t size, std::string path) {
        std::ofstream fileOut(path, std::ios::binary);
        fileOut.write(data, size);
    }

}