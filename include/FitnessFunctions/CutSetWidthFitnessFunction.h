#ifndef GPUSAT_CUTSETWIDTHFITNESSFUNCTION_H
#define GPUSAT_CUTSETWIDTHFITNESSFUNCTION_H

#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    class CutSetWidthFitnessFunction : public htd::ITreeDecompositionFitnessFunction {

    public:
        CutSetWidthFitnessFunction(void);

        ~CutSetWidthFitnessFunction();

        htd::FitnessEvaluation *fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const override;

        double getMaxCutSetSize(const htd::ITreeDecomposition &decomposition, htd::vertex_t node) const;

        CutSetWidthFitnessFunction *clone(void) const override;
    };
}

#endif //GPUSAT_CUTSETWIDTHFITNESSFUNCTION_H
