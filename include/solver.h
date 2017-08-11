#ifndef GPUSAT_SOLVER_H
#define GPUSAT_SOLVER_H
#define alloca __builtin_alloca

#include <CL/cl.hpp>
#include <types.h>

namespace gpusat {
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
        int isSat = 1;

        Solver(std::vector<cl::Platform> &platforms_, cl::Context &context_, std::vector<cl::Device> &devices_, cl::CommandQueue &queue_,
               cl::Program &program_, cl::Kernel &kernel_) : platforms(platforms_), context(context_), devices(devices_), queue(queue_),
                                                             program(program_), kernel(kernel_) {
            isSat = 1;
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

#endif //GPUSAT_SOLVER_H
