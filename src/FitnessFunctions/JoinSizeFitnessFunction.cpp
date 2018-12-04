#include <FitnessFunctions/JoinSizeFitnessFunction.h>
#include <htd/ITreeDecomposition.hpp>
#include <unordered_set>
#include <algorithm>

namespace gpusat {
    htd::FitnessEvaluation *JoinSizeFitnessFunction::fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const {
        long maxSize = 0;
        for (auto node:decomposition.joinNodes()) {
            maxSize = (decomposition.bagContent(node).size() > maxSize) ? decomposition.bagContent(node).size() : maxSize;
        }
        return new htd::FitnessEvaluation(2, -maxSize, -(double) (decomposition.maximumBagSize()));
    }


    JoinSizeFitnessFunction *JoinSizeFitnessFunction::clone(void) const {
        return new JoinSizeFitnessFunction();
    }
}