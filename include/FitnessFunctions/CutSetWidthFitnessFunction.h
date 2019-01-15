#ifndef GPUSAT_CUTSETWIDTHFITNESSFUNCTION_H
#define GPUSAT_CUTSETWIDTHFITNESSFUNCTION_H

#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    /**
     * Fitness function that first minimizes the cut set size and then the width.
     */
    class CutSetWidthFitnessFunction : public htd::ITreeDecompositionFitnessFunction {

    public:
        CutSetWidthFitnessFunction() = default;

        ~CutSetWidthFitnessFunction() = default;

        htd::FitnessEvaluation *fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const override;

        double getMaxCutSetSize(const htd::ITreeDecomposition &decomposition) const;

        CutSetWidthFitnessFunction *clone(void) const override;
    };
}

#endif //GPUSAT_CUTSETWIDTHFITNESSFUNCTION_H
