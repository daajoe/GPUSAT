#ifndef GPUSAT_GPUSAUTILS_H
#define GPUSAT_GPUSAUTILS_H

#include <queue>
#include <types.h>

namespace gpusat {
    class GPUSATUtils {
    public:

        /**
         * function to read a file into a string
         *
         * @param path      the path of the file
         * @return          the contents of the file
         */
        static std::string readFile(std::string path);

        /**
         * function to print the solution of a single bag in tree decomposition in human readable form
         *
         * @param numS              the number of assignments
         * @param numVariables      the number of variables
         * @param vars              arrray containing the variables
         * @param sol               array ontaining the number of models for each assignment
         */
        static void printSol(cl_long numS, std::vector<cl_long> vars, solType **sol, satformulaType &type, cl_long i);

        /**
         * reads a binary file from @path and retuns the contents in a string
         *
         * @param path      path to the file
         * @return          the file contents
         */
        static std::string readBinary(std::string path);
    };
}
#endif //GPUSAT_GPUSAUTILS_H
