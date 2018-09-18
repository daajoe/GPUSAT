#include <math.h>
#include <regex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <gpusautils.h>

namespace gpusat {

    bool myfunction(myTableElement i, myTableElement j) { return (i.id < j.id); }

    void GPUSATUtils::printSol(bagType &bag) {
        std::cout << "variables: \n";
        for (int j = 0; j < bag.variables.size(); ++j) {
            std::cout << bag.variables[j] << " ";
        }
        std::cout << "\nbags: " << bag.bags << "\n";
        for (int i = 0; i < bag.bags; i++) {
            std::cout << "solutions: " << bag.solution[i].numSolutions << "\n";
            std::cout << "size: " << bag.solution[i].size << "\n";
            std::cout << "min: " << bag.solution[i].minId << "\n";
            std::cout << "max: " << bag.solution[i].maxId << "\n";
            std::vector<myTableElement> sols;
            for (int a = 0; a < bag.solution[i].size; a++) {
                sols.push_back(bag.solution[i].elements[a]);
            }
            std::sort(sols.begin(), sols.end(), myfunction);
            for (auto a:sols) {
                std::cout << a.id << ": ";
                std::cout << a.count << "\n";
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