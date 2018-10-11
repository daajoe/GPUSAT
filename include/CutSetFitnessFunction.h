//
// Created by Markus on 10.10.2018.
//

#ifndef GPUSAT_CUTSETFITNESSFUNCTION_H
#define GPUSAT_CUTSETFITNESSFUNCTION_H

#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    class CutSetFitnessFunction : public htd::ITreeDecompositionFitnessFunction {

    public:
        CutSetFitnessFunction(void);

        ~CutSetFitnessFunction();

        htd::FitnessEvaluation *fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const override;

        double getMaxCutSetSize(const htd::ITreeDecomposition &decomposition, htd::vertex_t node) const;

        CutSetFitnessFunction *clone(void) const override;
    };
}

#endif //GPUSAT_CUTSETFITNESSFUNCTION_H
