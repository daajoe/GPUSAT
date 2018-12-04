#ifndef GPUSAT_NUMJOINFITNESSFUNCTION_H
#define GPUSAT_NUMJOINFITNESSFUNCTION_H

#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    class NumJoinFitnessFunction : public htd::ITreeDecompositionFitnessFunction {

    public:
        NumJoinFitnessFunction() = default;

        ~NumJoinFitnessFunction() = default;

        htd::FitnessEvaluation *fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const override;

        NumJoinFitnessFunction *clone(void) const override;
    };
}

#endif //GPUSAT_NUMJOINFITNESSFUNCTION_H
