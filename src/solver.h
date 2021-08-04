#ifndef GPUSAT_SOLVER_H_H
#define GPUSAT_SOLVER_H_H
#if defined __CYGWIN__ || defined __MINGW32__
#define alloca __builtin_alloca
#endif

#include <gpusat_types.h>
#include <map>
#include <boost/multiprecision/cpp_bin_float.hpp>

namespace gpusat {


    /**
     *
     */
    class Solver {
    protected:
        size_t memorySize;

    public:
        bool isSat = true;
        size_t numJoin = 0;
        size_t numIntroduceForget = 0;
        size_t maxTableSize = 0;
        size_t maxBag;
        size_t maxMemoryBuffer;
        SolveConfig solve_cfg;
        dataStructure solutionType;
        const bool do_trace = false;
        const bool do_cache = true;
        size_t id_running = 0x100000000;

        std::map<int64_t, BagType*> cached_nodes;

        size_t cacheSize() {
            size_t s = 0;
            for (const auto& [key, value] : cached_nodes) {
                assert(value->cached_solution.has_value());
                s += dataStructureSize(value->cached_solution.value());
            }
            return s;
        }

        /**
         *
         * @param context_
         * @param queue_
         * @param program_
         * @param memorySize_
         * @param maxMemoryBuffer_
         */
        Solver(size_t memorySize_, size_t maxMemoryBuffer_, dataStructure solutionType_, size_t maxBag_, SolveConfig solve_cfg_, bool do_trace_, bool do_cache_) : memorySize(memorySize_), maxBag(maxBag_), maxMemoryBuffer(maxMemoryBuffer_), solve_cfg(solve_cfg_), solutionType(solutionType_), do_trace(do_trace_), do_cache(do_cache_) {}

        /**
         * function to solve the sat problem
         *
         * @param decomp    the tree decomposition
         * @param formula   the sat formula
         * @param node      the node to start from in the tree decompostion
         */
        void solveProblem(const satformulaType &formula, BagType &node, BagType &pnode, nodeType lastNode);


        static boost::multiprecision::cpp_bin_float_100 bagSum(BagType& bag);
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
        void solveIntroduceForget(const satformulaType &formula, BagType &pnode, BagType &node, BagType &cnode, bool leaf, nodeType nextNode);

        /**
         * function to solve a join node
         *
         * @param node      the node to save the solutions in
         * @param edge1     the first child node
         * @param edge2     the second child node
         * @param formula   the sat formula
         */
        void solveJoin(BagType &node, BagType &edge1, BagType &edge2, const satformulaType &formula, nodeType nextNode);

        /**
         *
         * @param to
         * @param from
         * @param numVars
         */
        TreeSolution<CudaMem> combineTree(TreeSolution<CudaMem> &t1, TreeSolution<CudaMem> &t2);
    };
}
#endif //GPUSAT_SOLVER_H_H
