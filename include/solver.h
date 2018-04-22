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
        std::vector<cl::Platform> &platforms; //TODO remove
        cl::Context &context;
        std::vector<cl::Device> &devices; //TODO remove
        cl::CommandQueue &queue;
        cl::Program &program;
        cl::Kernel &kernel; //TODO remove
        //max with before splitting the bags
        cl_long maxWidth;
        cl_long inci; //TODO remove
        graphTypes graph; //TODO remove

    public:
        cl_long isSat = 1;
        cl_long getStats = 1; //TODO remove
        cl_long numJoin = 0;
        cl_long numIntroduce = 0; //TODO remove
        cl_long numIntroduceForget = 0;
        cl_long numLeafs = 0; //TODO remove
        std::vector<cl_long> numHoldPaths; //TODO remove
        std::vector<cl_long> numSolPaths; //TODO remove

        Solver(std::vector<cl::Platform> &platforms_, cl::Context &context_, std::vector<cl::Device> &devices_, cl::CommandQueue &queue_, cl::Program &program_,
               cl::Kernel &kernel_, int width, bool incidence, int getStats) : platforms(platforms_), context(context_), devices(devices_), queue(queue_),
                                                                               program(program_),
                                                                               kernel(kernel_), maxWidth(width), inci(incidence), getStats(getStats) {
            int test = 1;
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
        void solveProblem(treedecType &decomp, satformulaType &formula, bagType &node, bagType &lastNode);

    protected:

        virtual void solveIntroduceForget(satformulaType &formula, bagType &pnode, bagType &node, bagType &cnode, bool leaf)=0;

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
    };

    class Solver_Primal : public Solver {
    public:
        Solver_Primal(std::vector<cl::Platform> &platforms_, cl::Context &context_, std::vector<cl::Device> &devices_, cl::CommandQueue &queue_, cl::Program &program_,
                      cl::Kernel &kernel_, int width, bool inzi, int getStats) : Solver(platforms_, context_, devices_, queue_, program_, kernel_, width, inzi,
                                                                                        getStats) { graph = PRIMAL; }

    protected:
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula);

        void solveIntroduceForget(satformulaType &, bagType &, bagType &, bagType &, bool leaf);
    };

    class Solver_Incidence : public Solver {
    public:
        Solver_Incidence(std::vector<cl::Platform> &platforms_, cl::Context &context_, std::vector<cl::Device> &devices_, cl::CommandQueue &queue_,
                         cl::Program &program_,
                         cl::Kernel &kernel_, int width, bool inzi, int getStats) : Solver(platforms_, context_, devices_, queue_, program_, kernel_, width, inzi,
                                                                                           getStats) { graph = INCIDENCE; }

    protected:
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula);

        void solveIntroduceForget(satformulaType &, bagType &, bagType &, bagType &, bool leaf);
    };

    class Solver_Dual : public Solver {
    public:
        Solver_Dual(std::vector<cl::Platform> &platforms_, cl::Context &context_, std::vector<cl::Device> &devices_, cl::CommandQueue &queue_, cl::Program &program_,
                    cl::Kernel &kernel_, int width, bool inzi, int getStats) : Solver(platforms_, context_, devices_, queue_, program_, kernel_, width, inzi,
                                                                                      getStats) { graph = DUAL; }

    protected:
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula);

        void solveIntroduceForget(satformulaType &, bagType &, bagType &, bagType &, bool leaf);
    };
}
#endif //GPUSAT_SOLVER_H_H
