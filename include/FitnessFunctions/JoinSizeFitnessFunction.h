#ifndef GPUSAT_JOINSIZEFITNESSFUNCTION_H
#define GPUSAT_JOINSIZEFITNESSFUNCTION_H

#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    class JoinSizeFitnessFunction : public htd::ITreeDecompositionFitnessFunction {

    public:
        JoinSizeFitnessFunction() = default;

        ~JoinSizeFitnessFunction() = default;

        htd::FitnessEvaluation *fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const override;

        JoinSizeFitnessFunction *clone(void) const override;
    };
}

#endif //GPUSAT_JOINSIZEFITNESSFUNCTION_H
