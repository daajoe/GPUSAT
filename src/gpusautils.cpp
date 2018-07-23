#include <math.h>
#include <regex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <gpusautils.h>

namespace gpusat {

    void GPUSATUtils::printSol(bagType &bag) {
        std::cout << "variables: \n";
        for (int j = 0; j < bag.variables.size(); ++j) {
            std::cout << bag.variables[j] << " ";
        }
        std::cout << "\nsolutions: \n";
        for (int i = 0; i < bag.solution.size(); i++) {
            for (int a = 0; a < bag.solution[i].elements.size(); a++) {
                std::cout << bag.solution[i].elements[a].id << ": ";
                std::cout << bag.solution[i].elements[a].count << "\n";
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