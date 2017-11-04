#ifndef GPUSAT_GPUSAUTILS_H
#define GPUSAT_GPUSAUTILS_H
#define alloca __builtin_alloca

#include <queue>
#include <types.h>

namespace gpusat {
    class GPUSATUtils {
    public:
        /**
         * function to print the tree decomposition in human readable form
         *
         * @param decomp
         *      the tree decomposition to print
         */
        static void printTreeD(treedecType decomp);

        /**
         * function to print the solution for each bag in tree decomposition in human readable form
         *
         * @param decomp
         *      the tree decomposition containing the solutions
         */
        static void printSolutions(treedecType decomp);

        /**
         * function to print the sat formula in human readable form
         *
         * @param formula
         *      the sat formula to print
         */
        static void printFormula(satformulaType formula);

        /**
         * function to read a file into a string
         * @param path
         *      the path of the file
         * @return
         *      the contents of the file
         */
        static std::string readFile(std::string path);

        /**
         * function to print the solution of a single bag in tree decomposition in human readable form
         *
         * @param numS
         *      the number of assignments
         * @param numVariables
         *      the number of variables
         * @param vars
         *      arrray containing the variables
         * @param sol
         *      array ontaining the number of models for each assignment
         */
        static void printSol(cl_long numS, cl_long numVariables, cl_long *vars, solType **sol, satformulaType &type, cl_long i);

        /**
         * reads a binary file from @path and retuns the contents in a string
         *
         * @param path
         *      path to the file
         * @return
         *      the file contents
         */
        static std::string readBinary(std::string path);

        /**
         * writes data into a file in binary format
         *
         * @param path
         *      path to the fil
         * @param data
         *      the data to write
         * @param size
         *      the size of the data
         */
        static void writeBinary(char *data, size_t size, std::string path);

        static void printTreeD(preetreedecType decomp);
    };
}
#endif //GPUSAT_GPUSAUTILS_H
