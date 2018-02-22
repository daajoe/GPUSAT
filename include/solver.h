#ifndef GPUSAT_SOLVER_H_H
#define GPUSAT_SOLVER_H_H
#define alloca __builtin_alloca

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
        graphTypes graph;

    public:
        cl_long isSat = 1;
        cl_long getStats = 1;
        cl_long numJoin = 0;
        cl_long numIntroduce = 0;
        cl_long numForget = 0;
        cl_long numLeafs = 0;
        std::vector<cl_long> numHoldPaths;
        std::vector<cl_long> numSolPaths;

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
                      cl::Kernel &kernel_, int width, bool inzi, int getStats) : Solver(platforms_, context_, devices_, queue_, program_, kernel_, width, inzi,
                                                                                        getStats) { graph = PRIMAL; }

    protected:
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula);

        void solveIntroduce(satformulaType &formula, bagType &node, bagType &edge);

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

        void solveIntroduce(satformulaType &formula, bagType &node, bagType &edge);

        void solveIntroduceForget(satformulaType &, bagType &, bagType &, bagType &, bool leaf);
    };

    class Solver_Dual : public Solver {
    public:
        Solver_Dual(std::vector<cl::Platform> &platforms_, cl::Context &context_, std::vector<cl::Device> &devices_, cl::CommandQueue &queue_, cl::Program &program_,
                    cl::Kernel &kernel_, int width, bool inzi, int getStats) : Solver(platforms_, context_, devices_, queue_, program_, kernel_, width, inzi,
                                                                                      getStats) { graph = DUAL; }

    protected:
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula);

        void solveIntroduce(satformulaType &formula, bagType &node, bagType &edge);

        void solveIntroduceForget(satformulaType &, bagType &, bagType &, bagType &, bool leaf);
    };
}
#endif //GPUSAT_SOLVER_H_H
