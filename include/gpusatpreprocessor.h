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
        * @param decomp             the tree decomposition
         * @param combineWidth      max width to combine bags
         */
        static void preprocessDecomp(bagType *decomp, cl_long combineWidth);

        /**
         * removes facts from the sat formula
         *
         * @param decomp            the tree decomposition
         * @param formula           the sat formula
         * @param gType             the type of graph
         * @param defaultWeight     for WMC the product of the weights of the removed literals
         */
        static void preprocessFacts(treedecType &decomp, satformulaType &formula, graphTypes gType, cl_double &defaultWeight);

    private:

        static void relableDecomp(bagType *decomp, cl_long id);

        static void relableFormula(satformulaType &formula, cl_long id);
    };
}
#endif //GPUSAT_GPUSATPREPROCESSOR_H
