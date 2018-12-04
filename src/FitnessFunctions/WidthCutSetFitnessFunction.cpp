#include <FitnessFunctions/WidthCutSetFitnessFunction.h>
#include <htd/ITreeDecomposition.hpp>
#include <unordered_set>
#include <algorithm>

namespace gpusat {
    htd::FitnessEvaluation *WidthCutSetFitnessFunction::fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const {
        return new htd::FitnessEvaluation(2, -(double) (decomposition.maximumBagSize()), -getMaxCutSetSize(decomposition, decomposition.root()));
    }

    double gpusat::WidthCutSetFitnessFunction::getMaxCutSetSize(const htd::ITreeDecomposition &decomposition, htd::vertex_t vertex) const {
        double childSize = 0;
        std::vector<htd::vertex_t> currentNodes = decomposition.bagContent(vertex);
        std::vector<htd::vertex_t> cNodes;
        for (htd::vertex_t childVertex : decomposition.children(vertex)) {
            const std::vector<htd::vertex_t> &childNodes = decomposition.bagContent(childVertex);
            for (htd::vertex_t n:childNodes) {
                std::size_t s = cNodes.size();
                cNodes.push_back(n);
            }
            childSize = std::max(childSize, getMaxCutSetSize(decomposition, childVertex));
        }

        std::sort(currentNodes.begin(), currentNodes.end());
        std::sort(cNodes.begin(), cNodes.end());
        std::vector<htd::vertex_t> v;
        std::set_intersection(currentNodes.begin(), currentNodes.end(), cNodes.begin(), cNodes.end(), back_inserter(v));
        return std::max((double) v.size(), childSize);
    }

    WidthCutSetFitnessFunction *WidthCutSetFitnessFunction::clone(void) const {
        return new WidthCutSetFitnessFunction();
    }
}