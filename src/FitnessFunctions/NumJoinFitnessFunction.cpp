#include <FitnessFunctions/NumJoinFitnessFunction.h>
#include <htd/ITreeDecomposition.hpp>
#include <unordered_set>
#include <algorithm>

namespace gpusat {
    htd::FitnessEvaluation *NumJoinFitnessFunction::fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const {
        return new htd::FitnessEvaluation(2, -decomposition.joinNodeCount());
    }

    NumJoinFitnessFunction *NumJoinFitnessFunction::clone(void) const {
        return new NumJoinFitnessFunction();
    }
}