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
        const std::vector<htd::vertex_t> &currentNodes = decomposition.bagContent(vertex);
        std::unordered_set<htd::vertex_t> nodes;
        nodes.insert(currentNodes.begin(), currentNodes.end());
        std::unordered_set<htd::vertex_t> cNodes;
        for (htd::vertex_t childVertex : decomposition.children(vertex)) {
            const std::vector<htd::vertex_t> &childNodes = decomposition.bagContent(childVertex);
            for (auto n:childNodes) {
                cNodes.insert(n);
            }
            childSize = std::max(childSize, getMaxCutSetSize(decomposition, childVertex));
        }

        std::vector<htd::vertex_t> v(nodes.size() + currentNodes.size());
        std::vector<htd::vertex_t>::iterator it = std::set_intersection(currentNodes.begin(), currentNodes.end(), nodes.begin(), nodes.end(), v.begin());
        double d = it - v.begin();
        return std::max(d, childSize);
    }

    WidthCutSetFitnessFunction::WidthCutSetFitnessFunction(void) {

    }

    WidthCutSetFitnessFunction::~WidthCutSetFitnessFunction() {

    }

    WidthCutSetFitnessFunction *WidthCutSetFitnessFunction::clone(void) const {
        return new WidthCutSetFitnessFunction();
    }
}