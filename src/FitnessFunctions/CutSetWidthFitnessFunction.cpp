#include <FitnessFunctions/CutSetWidthFitnessFunction.h>
#include <htd/ITreeDecomposition.hpp>
#include <unordered_set>
#include <algorithm>

namespace gpusat {
    htd::FitnessEvaluation *CutSetWidthFitnessFunction::fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const {
        return new htd::FitnessEvaluation(2, -getMaxCutSetSize(decomposition), -(double) (decomposition.maximumBagSize()));
    }

    double gpusat::CutSetWidthFitnessFunction::getMaxCutSetSize(const htd::ITreeDecomposition &decomposition) const {
        double sizes = 0;
        for (auto a:decomposition.vertices()) {
            std::vector<htd::vertex_t> currentNodes = decomposition.bagContent(a);
            std::vector<htd::vertex_t> cNodes;
            for (htd::vertex_t childVertex : decomposition.children(a)) {
                const std::vector<htd::vertex_t> &childNodes = decomposition.bagContent(childVertex);
                for (htd::vertex_t n:childNodes) {
                    cNodes.push_back(n);
                }
            }

            std::sort(currentNodes.begin(), currentNodes.end());
            std::sort(cNodes.begin(), cNodes.end());
            std::vector<htd::vertex_t> v;
            std::set_intersection(currentNodes.begin(), currentNodes.end(), cNodes.begin(), cNodes.end(), back_inserter(v));
            sizes = std::max((double) v.size(), sizes);
        }
        return sizes;
    }

    CutSetWidthFitnessFunction *CutSetWidthFitnessFunction::clone(void) const {
        return new CutSetWidthFitnessFunction();
    }
}