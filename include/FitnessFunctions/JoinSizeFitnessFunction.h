#ifndef GPUSAT_JOINSIZEFITNESSFUNCTION_H
#define GPUSAT_JOINSIZEFITNESSFUNCTION_H

#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    /**
     * Fitness function that minimizes the number of vertices in a join node and then the width.
     */
    class JoinSizeFitnessFunction : public htd::ITreeDecompositionFitnessFunction {

    public:
        JoinSizeFitnessFunction() = default;

        ~JoinSizeFitnessFunction() = default;

        htd::FitnessEvaluation *fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const override;

        JoinSizeFitnessFunction *clone(void) const override;
    };
}

#endif //GPUSAT_JOINSIZEFITNESSFUNCTION_H
