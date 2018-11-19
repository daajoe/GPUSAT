#ifndef GPUSAT_WIDTHCUTSETFITNESSFUNCTION_H
#define GPUSAT_WIDTHCUTSETFITNESSFUNCTION_H

#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    class WidthCutSetFitnessFunction : public htd::ITreeDecompositionFitnessFunction {

    public:
        WidthCutSetFitnessFunction(void);

        ~WidthCutSetFitnessFunction();

        htd::FitnessEvaluation *fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const override;

        double getMaxCutSetSize(const htd::ITreeDecomposition &decomposition, htd::vertex_t node) const;

        WidthCutSetFitnessFunction *clone(void) const override;
    };
}

#endif //GPUSAT_WIDTHCUTSETFITNESSFUNCTION_H
