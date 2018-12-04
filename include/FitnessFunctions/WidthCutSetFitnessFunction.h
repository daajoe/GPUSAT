#ifndef GPUSAT_WIDTHCUTSETFITNESSFUNCTION_H
#define GPUSAT_WIDTHCUTSETFITNESSFUNCTION_H

#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    /**
     * Fitness function that first minimizes the width and then the cut set size.
     */
    class WidthCutSetFitnessFunction : public htd::ITreeDecompositionFitnessFunction {

    public:
        WidthCutSetFitnessFunction() = default;

        ~WidthCutSetFitnessFunction() = default;

        htd::FitnessEvaluation *fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const override;

        double getMaxCutSetSize(const htd::ITreeDecomposition &decomposition, htd::vertex_t vertex) const;

        WidthCutSetFitnessFunction *clone(void) const override;
    };
}

#endif //GPUSAT_WIDTHCUTSETFITNESSFUNCTION_H
