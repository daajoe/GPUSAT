#include <math.h>
#include <regex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <gpusautils.h>

namespace gpusat {
    cl_double GPUSATUtils::getCount(cl_long id, cl_long *tree, cl_int numVars) {
        int nextId = 0;
        for (int i = 0; i < numVars; i++) {
            nextId = ((int *) &(tree[nextId]))[(id >> (numVars - i - 1)) & 1];
            if (nextId == 0) {
                return 0.0;
            }
        }
        return *reinterpret_cast <cl_double *>(&tree[nextId]);
    }

    void GPUSATUtils::printSol(bagType &bag) {
        std::cout << "variables: \n";
        for (int j = 0; j < bag.variables.size(); ++j) {
            std::cout << bag.variables[j] << " ";
        }
        std::cout << "\nsolutions: " << "\n";
        for (int i = 0; i < bag.bags; i++) {
            for (cl_long a = bag.solution[i].minId; a < bag.solution[i].maxId; a++) {
                if (bag.solution[i].elements != nullptr) {
                    std::cout << a << ": " << getCount(a, bag.solution[i].elements, bag.variables.size()) << "\n";
                } else {
                    std::cout << a << ": " << 0 << "\n";
                }
            }
        }
        std::cout << "-----\n";
        for (int i = 0; i < bag.bags; i++) {
            std::cout << "solutions: " << bag.solution[i].numSolutions << "\n";
            std::cout << "size: " << bag.solution[i].size << "\n";
            std::cout << "min: " << bag.solution[i].minId << "\n";
            std::cout << "max: " << bag.solution[i].maxId << "\n";
            std::vector<cl_double> sols;
            for (cl_long a = 0; a < bag.solution[i].size; a++) {
                if (bag.solution[i].elements != nullptr) {
                    std::cout << a << ": " << ((int *) &(bag.solution[i].elements[a]))[0] << "/" << ((int *) &(bag.solution[i].elements[a]))[1] << "\n";
                }
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