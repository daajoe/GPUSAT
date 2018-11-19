#include <FitnessFunctions/WidthFitnessFunction.h>
#include <htd/ITreeDecomposition.hpp>
#include <unordered_set>
#include <algorithm>

namespace gpusat {
    htd::FitnessEvaluation *WidthFitnessFunction::fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const {
        return new htd::FitnessEvaluation(1, -(double) (decomposition.maximumBagSize()));
    }

    WidthFitnessFunction::WidthFitnessFunction(void) {

    }

    WidthFitnessFunction::~WidthFitnessFunction() {

    }

    WidthFitnessFunction *WidthFitnessFunction::clone(void) const {
        return new WidthFitnessFunction();
    }
}