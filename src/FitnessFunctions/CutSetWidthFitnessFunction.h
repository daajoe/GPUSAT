#ifndef GPUSAT_CUTSETWIDTHFITNESSFUNCTION_H
#define GPUSAT_CUTSETWIDTHFITNESSFUNCTION_H

#include <htd/ITreeDecompositionFitnessFunction.hpp>
#include <unordered_set>
#include <algorithm>

namespace gpusat {
    /**
     * Fitness function that first minimizes the cut set size and then the width.
     */
    class CutSetWidthFitnessFunction : public htd::ITreeDecompositionFitnessFunction {

    public:
        CutSetWidthFitnessFunction() = default;

        ~CutSetWidthFitnessFunction() = default;

        htd::FitnessEvaluation *fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const override {
            return new htd::FitnessEvaluation(
                    2,
                    -getMaxCutSetSize(decomposition),
                    -(double) (decomposition.maximumBagSize())
            );
        }

        double getMaxCutSetSize(const htd::ITreeDecomposition &decomposition) const {
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
                std::set_intersection(
                        currentNodes.begin(), currentNodes.end(),
                        cNodes.begin(), cNodes.end(),
                back_inserter(v));
                sizes = std::max((double) v.size(), sizes);
            }
            return sizes;
        }

        CutSetWidthFitnessFunction *clone(void) const override {
            return new CutSetWidthFitnessFunction();
        }
    };
}

#endif //GPUSAT_CUTSETWIDTHFITNESSFUNCTION_H
