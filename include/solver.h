#ifndef GPUSAT_SOLVER_H_H
#define GPUSAT_SOLVER_H_H
#if defined __CYGWIN__ || defined __MINGW32__
#define alloca __builtin_alloca
#endif

#include <CL/cl.hpp>
#include <types.h>

namespace gpusat {
    class Solver {
    protected:
        cl::Context &context;
        cl::CommandQueue &queue;
        cl::Program &program;
        cl_long memorySize;

    public:
        cl_long isSat = 1;
        cl_long numJoin = 0;
        cl_long numIntroduceForget = 0;
        cl_long maxTableSize = 0;
        cl_long maxMemoryBuffer = 0;

        Solver(cl::Context &context_, cl::CommandQueue &queue_, cl::Program &program_, cl_long memorySize_, cl_long maxMemoryBuffer_) : context(context_), queue(queue_), program(program_), memorySize(memorySize_), maxMemoryBuffer(maxMemoryBuffer_) {}

        /**
         * function to solve the sat problem
         *
         * @param decomp    the tree decomposition
         * @param formula   the sat formula
         * @param node      the node to start from in the tree decompostion
         */
        void solveProblem(treedecType &decomp, satformulaType &formula, bagType &node, bagType &pnode, nodeTypes lastNode);

    protected:

        /**
         * function to solve an introduce forget node
         *
         * @param formula   the sat formula
         * @param pnode     the parent of the current node
         * @param node      the current node
         * @param cnode     the child of the current node
         * @param leaf      indicates that the current node is a leaf node
         */
        virtual void solveIntroduceForget(satformulaType &formula, bagType &pnode, bagType &node, bagType &cnode, bool leaf, nodeTypes nextNode) = 0;

        /**
         * function to solve a join node
         *
         * @param node      the node to save the solutions in
         * @param edge1     the first child node
         * @param edge2     the second child node
         * @param formula   the sat formula
         */
        virtual void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula, nodeTypes nextNode) = 0;

        void cleanTree(treeType &table, cl_long size, cl_long numVars);

        void combineTree(treeType &to, treeType &from, cl_long numVars);
    };

    class Solver_Primal : public Solver {
    public:
        Solver_Primal(cl::Context &context_, cl::CommandQueue &queue_, cl::Program &program_, cl_ulong memorySize_, cl_ulong maxMemoryBuffer_) : Solver(context_, queue_, program_, memorySize_, maxMemoryBuffer_) {}

    protected:
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula, nodeTypes nextNode) override;

        void solveIntroduceForget(satformulaType &, bagType &, bagType &, bagType &, bool leaf, nodeTypes nextNode) override;

    };
/*
    class Solver_Incidence : public Solver {
    public:
        Solver_Incidence(cl::Context &context_, cl::CommandQueue &queue_, cl::Program &program_, int width) : Solver(context_, queue_, program_, width) {}

    protected:
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula);

        void solveIntroduceForget(satformulaType &, bagType &, bagType &, bagType &, bool leaf);
    };

    class Solver_Dual : public Solver {
    public:
        Solver_Dual(cl::Context &context_, cl::CommandQueue &queue_, cl::Program &program_, int width) : Solver(context_, queue_, program_, width) {}

    protected:
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula);

        void solveIntroduceForget(satformulaType &, bagType &, bagType &, bagType &, bool leaf);
    };*/
}
#endif //GPUSAT_SOLVER_H_H
