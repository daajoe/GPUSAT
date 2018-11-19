#ifndef GPUSAT_WIDTHFITNESSFUNCTION_H
#define GPUSAT_WIDTHFITNESSFUNCTION_H

#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    class WidthFitnessFunction : public htd::ITreeDecompositionFitnessFunction {

    public:
        WidthFitnessFunction(void);

        ~WidthFitnessFunction();

        htd::FitnessEvaluation *fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const override;

        WidthFitnessFunction *clone(void) const override;
    };
}

#endif //GPUSAT_WIDTHFITNESSFUNCTION_H
