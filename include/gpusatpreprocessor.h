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
             */
        static void preprocessDecomp(preebagType *decomp, cl_long combineWidth);

        static void preprocessFacts(preetreedecType decomp, satformulaType &formula, graphTypes gType, solType &defaultWeight);

    private:

        static void relableDecomp(preebagType *decomp, cl_long id);

        static void relableFormula(satformulaType &formula, cl_long id);
    };
}
#endif //GPUSAT_GPUSATPREPROCESSOR_H
