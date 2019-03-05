#ifndef GPUSAT_GPUSATPREPROCESSOR_H
#define GPUSAT_GPUSATPREPROCESSOR_H

#include <queue>
#include <types.h>
#include <unordered_map>

namespace gpusat {
    class Preprocessor {
    public:
        /**
         * preprocess the tree decomposition
         *
         * @param decomp
         *      the tree decomposition
         * @param combineWidth
         *      max width to combine bags
         */
        static void preprocessDecomp(bagType *decomp, cl_long combineWidth);

        /**
         * removes facts from the sat formula
         *
         * @param decomp
         *      the tree decomposition
         * @param formula
         *      the sat formula
         * @param defaultWeight
         *      for WMC the product of the weights of the removed literals
         */
        static void preprocessFacts(treedecType &decomp, satformulaType &formula, cl_double &defaultWeight);

    private:

        /**
         * used to relable a decomposition if a node is removed
         *
         * @param decomp
         *      the tree decomposition
         * @param id
         *      the id of the removed node
         */
        static void relableDecomp(bagType *decomp, cl_long id);

        /**
         * used to relable a formula if a variable is removed
         *
         * @param formula
         *      the formula
         * @param id
         *      the id of the removed variable
         */
        static void relableFormula(satformulaType &formula, cl_long id);
    };
}
#endif //GPUSAT_GPUSATPREPROCESSOR_H
