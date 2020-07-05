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
        static void preprocessDecomp(BagType& decomp, int64_t combineWidth);

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
        static void preprocessFacts(treedecType &decomp, satformulaType &formula, double &defaultWeight);

    private:

        /**
         * used to relabel a decomposition if a node is removed
         *
         * @param decomp
         *      the tree decomposition
         * @param id
         *      the id of the removed node
         */
        static void relabelDecomp(BagType& decomp, int64_t id);

        /**
         * used to relabel a formula if a variable is removed
         *
         * @param formula
         *      the formula
         * @param id
         *      the id of the removed variable
         */
        static void relabelFormula(satformulaType &formula, int64_t id);
    };
}
#endif //GPUSAT_GPUSATPREPROCESSOR_H
