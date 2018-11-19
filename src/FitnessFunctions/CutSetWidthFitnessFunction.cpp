#include <FitnessFunctions/CutSetWidthFitnessFunction.h>
#include <htd/ITreeDecomposition.hpp>
#include <unordered_set>
#include <algorithm>

namespace gpusat {
    htd::FitnessEvaluation *CutSetWidthFitnessFunction::fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const {
        return new htd::FitnessEvaluation(2, -getMaxCutSetSize(decomposition, decomposition.root()), -(double) (decomposition.maximumBagSize()));
    }

    double gpusat::CutSetWidthFitnessFunction::getMaxCutSetSize(const htd::ITreeDecomposition &decomposition, htd::vertex_t node) const {
        double childSize = 0;
        std::vector<htd::vertex_t> currentNodes = decomposition.bagContent(node);
        std::vector<htd::vertex_t> childNodes;
        for (htd::vertex_t childNode : decomposition.children(node)) {
            std::vector<htd::vertex_t> childNodes_ = decomposition.bagContent(childNode);
            for (auto n:childNodes_) {
                childNodes.push_back(n);
            }
            childSize = std::max(childSize, getMaxCutSetSize(decomposition, childNode));
        }

        std::sort(currentNodes.begin(), currentNodes.end());
        std::sort(childNodes.begin(), childNodes.end());
        std::vector<htd::vertex_t> v(childNodes.size() + currentNodes.size());
        std::vector<htd::vertex_t>::iterator it = std::set_intersection(currentNodes.begin(), currentNodes.end(), childNodes.begin(), childNodes.end(), v.begin());
        double d = it - v.begin();
        return std::max(d, childSize);
    }

    CutSetWidthFitnessFunction::CutSetWidthFitnessFunction(void) {

    }

    CutSetWidthFitnessFunction::~CutSetWidthFitnessFunction() {

    }

    CutSetWidthFitnessFunction *CutSetWidthFitnessFunction::clone(void) const {
        return new CutSetWidthFitnessFunction();
    }
}