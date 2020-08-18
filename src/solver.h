#ifndef GPUSAT_SOLVER_H_H
#define GPUSAT_SOLVER_H_H
#if defined __CYGWIN__ || defined __MINGW32__
#define alloca __builtin_alloca
#endif

#include "types.h"

namespace gpusat {


    size_t bagTypeHash(const BagType&);
    /**
     *
     */
    class Solver {
    protected:
        int64_t memorySize;

    public:
        int64_t isSat = 1;
        int64_t numJoin = 0;
        int64_t numIntroduceForget = 0;
        int64_t maxTableSize = 0;
        int64_t maxBag = 0;
        int64_t maxMemoryBuffer = 0;
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
        Solver(int64_t memorySize_, int64_t maxMemoryBuffer_, dataStructure solutionType_, int64_t maxBag_, SolveMode solve_mode_) : memorySize(memorySize_), maxMemoryBuffer(maxMemoryBuffer_), solutionType(solutionType_), maxBag(maxBag_), solve_mode(solve_mode_) {}

        /**
         * function to solve the sat problem
         *
         * @param decomp    the tree decomposition
         * @param formula   the sat formula
         * @param node      the node to start from in the tree decompostion
         */
        void solveProblem(satformulaType &formula, BagType &node, BagType &pnode, nodeType lastNode);

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
        void solveIntroduceForget(satformulaType &formula, BagType &pnode, BagType &node, BagType &cnode, bool leaf, nodeType nextNode);

        /**
         * function to solve a join node
         *
         * @param node      the node to save the solutions in
         * @param edge1     the first child node
         * @param edge2     the second child node
         * @param formula   the sat formula
         */
        void solveJoin(BagType &node, BagType &edge1, BagType &edge2, satformulaType &formula, nodeType nextNode);

        /**
         *
         * @param table
         * @param size
         * @param numVars
         * @param node
         */
        TreeSolution arrayToTree(ArraySolution &table, int64_t size, int64_t numVars, BagType &node, int64_t nextSize);

        /**
         *
         * @param to
         * @param from
         * @param numVars
         */
        void combineTree(TreeSolution &to, TreeSolution &from, int64_t numVars);
    };
}
#endif //GPUSAT_SOLVER_H_H
