#ifndef GPUSAT_MAIN_H
#define GPUSAT_MAIN_H
#define alloca __builtin_alloca

#include <CL/cl.hpp>

#define solType cl_float

namespace gpusat {

    /// type for a bag in the tree decomposition
    struct bagType {
        cl_long numVars = 0;
        cl_long numEdges = 0;
        cl_long numSol = 0;
        cl_long *variables = nullptr;
        cl_long *edges = nullptr;
        solType *solution = nullptr;
    };

    /// type for saving a tree decomposition
    struct treedecType {
        cl_long numb = 0;
        bagType *bags = nullptr;
    };

    /// type for saving the sat formula
    struct satformulaType {
        cl_long numclauses = 0;
        cl_long totalNumVar = 0;
        cl_long *numVarsC = nullptr;
        cl_long *clauses = nullptr;
    };

    class Solver {
    private:
        std::vector<cl::Platform> &platforms;
        cl::Context &context;
        std::vector<cl::Device> &devices;
        cl::CommandQueue &queue;
        cl::Program &program;
        cl::Kernel &kernel;

        void solveForgIntroduce(satformulaType &formula, bagType &node, bagType &next);

    public:

        Solver(std::vector<cl::Platform> &platforms_, cl::Context &context_, std::vector<cl::Device> &devices_,
               cl::CommandQueue &queue_, cl::Program &program_, cl::Kernel &kernel_) : platforms(platforms_),
                                                                                       context(context_),
                                                                                       devices(devices_), queue(queue_),
                                                                                       program(program_),
                                                                                       kernel(kernel_) {

        }

        /**
         * function to solve the sat problem
         *
         * @param decomp
         *      the tree decomposition
         * @param formula
         *      the sat formula
         * @param node
         *      the node to start from in the tree decompostion
         */
        void solveProblem(treedecType &decomp, satformulaType &formula, bagType &node);

    protected:
        /**
         * function to solve a join node
         *
         * @param node
         *      the node to save the solutions in
         * @param edge1
         *      the first edge
         * @param edge2
         *      the second edge
         */
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2);

        /**
         * function to solve a forget node
         *
         * @param node
         *      the node to save the solutions in
         * @param edge
         *      the next node
         */
        void solveForget(bagType &node, bagType &edge);

        /**
         * function to solve a leaf node
         *
         * @param formula
         *      the sat formula
         * @param node
         *      the node to save the solutions in
         */
        void solveLeaf(satformulaType &formula, bagType &node);

        /**
         * function to solve a introduce node
         *
         * @param formula
         *      the sat formula
         * @param node
         *      the node to save the solutions in
         * @param edge
         *      the next node
         */
        void solveIntroduce(satformulaType &formula, bagType &node, bagType &edge);
    };
}
#endif //GPUSAT_MAIN_H
