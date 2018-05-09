#include <math.h>
#include <regex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <gpusautils.h>
#include <d4_utils.h>

namespace gpusat {

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

}