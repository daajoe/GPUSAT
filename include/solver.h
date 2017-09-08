#ifndef GPUSAT_SOLVER_H_H
#define GPUSAT_SOLVER_H_H

#include <types.h>

namespace gpusat {
    class Solver {
    protected:
        std::vector<cl::Platform> &platforms;
        cl::Context &context;
        std::vector<cl::Device> &devices;
        cl::CommandQueue &queue;
        cl::Program &program;
        cl::Kernel &kernel;
        //max with before splitting the bags
        cl_long maxWidth;
        cl_long inci;

    public:
        int isSat = 1;

        /**
         * TODO
         */
        Solver(std::vector<cl::Platform> &platforms_, cl::Context &context_, std::vector<cl::Device> &devices_, cl::CommandQueue &queue_, cl::Program &program_,
               cl::Kernel &kernel_, int width, bool incidence) : platforms(platforms_), context(context_), devices(devices_), queue(queue_), program(program_),
                                                                 kernel(kernel_), maxWidth(width), inci(incidence) {}

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
         * TODO
         * @param formula
         * @param node
         * @param next
         */
        virtual void solveForgIntroduce(satformulaType &formula, bagType &node, bagType &next);

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
        virtual void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &type)=0;

        /**
         * function to solve a forget node
         *
         * @param node
         *      the node to save the solutions in
         * @param edge
         *      the next node
         */
        virtual void solveForget(bagType &node, bagType &edge, satformulaType &formula)=0;

        /**
         * function to solve a leaf node
         *
         * @param formula
         *      the sat formula
         * @param node
         *      the node to save the solutions in
         */
        virtual void solveLeaf(satformulaType &formula, bagType &node)=0;

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
        virtual void solveIntroduce(satformulaType &formula, bagType &node, bagType &edge)=0;
    };

    class Solver_Primal : public Solver {
    public:
        Solver_Primal(std::vector<cl::Platform> &platforms_, cl::Context &context_, std::vector<cl::Device> &devices_, cl::CommandQueue &queue_, cl::Program &program_,
                      cl::Kernel &kernel_, int width, bool inzi) : Solver(platforms_, context_, devices_, queue_, program_, kernel_, width, inzi) {}

    protected:
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula);

        void solveForget(bagType &node, bagType &edge, satformulaType &formula);

        void solveLeaf(satformulaType &formula, bagType &node);

        void solveIntroduce(satformulaType &formula, bagType &node, bagType &edge);
    };

    class Solver_Incidence : public Solver {
    public:
        Solver_Incidence(std::vector<cl::Platform> &platforms_, cl::Context &context_, std::vector<cl::Device> &devices_, cl::CommandQueue &queue_, cl::Program &program_,
                         cl::Kernel &kernel_, int width, bool inzi) : Solver(platforms_, context_, devices_, queue_, program_, kernel_, width, inzi) {}

    protected:
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula);

        void solveForget(bagType &node, bagType &edge, satformulaType &formula);

        void solveLeaf(satformulaType &formula, bagType &node);

        void solveIntroduce(satformulaType &formula, bagType &node, bagType &edge);
    };
}
#endif //GPUSAT_SOLVER_H_H
