#ifndef GPUSAT_GPUSATPREPROCESSOR_H
#define GPUSAT_GPUSATPREPROCESSOR_H

#include <queue>
#include <unordered_map>


#include <gpusat_types.h>

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
        static void preprocessDecomp(BagType& decomp, size_t combineWidth);

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
        static void preprocessFacts(satformulaType &formula, double &defaultWeight);

        static void checkNoFactInDecomp(BagType& decomp, const std::vector<int64_t>& var);

        static void relabelFormula(satformulaType &formula);

        /// build a map of variable renaming offsets for relabeling variables based on removed facts.
        /// the fact vector must be sorted.
        static std::vector<int64_t> buildRelabelMap(int64_t facts_and_vars, const std::vector<int64_t>& facts);
    };
}
#endif //GPUSAT_GPUSATPREPROCESSOR_H
