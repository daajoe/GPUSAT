#ifndef GPUSAT_NUMJOINFITNESSFUNCTION_H
#define GPUSAT_NUMJOINFITNESSFUNCTION_H

#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    /**
     * Fitness function that minimizes the number of join nodes and then the width.
     */
    class NumJoinFitnessFunction : public htd::ITreeDecompositionFitnessFunction {

    public:
        NumJoinFitnessFunction() = default;

        ~NumJoinFitnessFunction() = default;

        htd::FitnessEvaluation *fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const override {
            return new htd::FitnessEvaluation(2,
                    -decomposition.joinNodeCount(),
                    -(double) (decomposition.maximumBagSize())
            );
        };

        NumJoinFitnessFunction *clone(void) const override {
            return new NumJoinFitnessFunction();
        }
    };
}

#endif //GPUSAT_NUMJOINFITNESSFUNCTION_H
