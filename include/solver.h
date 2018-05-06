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
        //max with before splitting the bags
        cl_long maxWidth;

    public:
        cl_long isSat = 1;
        cl_long numJoin = 0;
        cl_long numIntroduceForget = 0;

        Solver(cl::Context &context_, cl::CommandQueue &queue_, cl::Program &program_, int width) : context(context_), queue(queue_), program(program_), maxWidth(width) {
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
        Solver_Primal(cl::Context &context_, cl::CommandQueue &queue_, cl::Program &program_, int width) : Solver(context_, queue_, program_, width) {}

    protected:
        void solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula);

        void solveIntroduceForget(satformulaType &, bagType &, bagType &, bagType &, bool leaf);
    };

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
    };
}
#endif //GPUSAT_SOLVER_H_H
