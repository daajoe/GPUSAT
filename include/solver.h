#ifndef GPUSAT_SOLVER_H_H
#define GPUSAT_SOLVER_H_H
#if defined __CYGWIN__ || defined __MINGW32__
#define alloca __builtin_alloca
#endif

#include <CL/cl.hpp>
#include <types.h>

namespace gpusat {
    /**
     *
     */
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
        cl_long maxBag = 0;
        cl_long maxMemoryBuffer = 0;
        SolveMode solve_mode = DEFAULT;
        dataStructure solutionType = TREE;

        /**
         *
         * @param context_
         * @param queue_
         * @param program_
         * @param memorySize_
         * @param maxMemoryBuffer_
         */
        Solver(cl::Context &context_, cl::CommandQueue &queue_, cl::Program &program_, cl_long memorySize_, cl_long maxMemoryBuffer_, dataStructure solutionType_, cl_long maxBag_, SolveMode solve_mode_) : context(context_), queue(queue_), program(program_), memorySize(memorySize_), maxMemoryBuffer(maxMemoryBuffer_), solutionType(solutionType_), maxBag(maxBag_), solve_mode(solve_mode_) {}

        /**
         * function to solve the sat problem
         *
         * @param decomp    the tree decomposition
         * @param formula   the sat formula
         * @param node      the node to start from in the tree decompostion
         */
        void solveProblem(treedecType &decomp, satformulaType &formula, bagType &node, bagType &pnode, nodeType lastNode);

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
        void solveIntroduceForget(satformulaType &formula, bagType &pnode, bagType &node, bagType &cnode, bool leaf, nodeType nextNode);

        /**
         * function to solve a join node
         *
         * @param node      the node to save the solutions in
         * @param edge1     the first child node
         * @param edge2     the second child node
         * @param formula   the sat formula
         */
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula, nodeType nextNode);

        /**
         *
         * @param table
         * @param size
         * @param numVars
         * @param node
         */
        void cleanTree(treeType &table, cl_long size, cl_long numVars, bagType &node, cl_long nextSize);

        /**
         *
         * @param to
         * @param from
         * @param numVars
         */
        void combineTree(treeType &to, treeType &from, cl_long numVars);
    };
}
#endif //GPUSAT_SOLVER_H_H
