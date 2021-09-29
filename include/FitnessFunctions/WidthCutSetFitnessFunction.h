#ifndef GPUSAT_WIDTHCUTSETFITNESSFUNCTION_H
#define GPUSAT_WIDTHCUTSETFITNESSFUNCTION_H

#include <htd/ITreeDecompositionFitnessFunction.hpp>
#include <unordered_set>
#include <algorithm>

namespace gpusat {
    /**
     * Fitness function that first minimizes the width and then the cut set size.
     */
    class WidthCutSetFitnessFunction : public htd::ITreeDecompositionFitnessFunction {

    public:
        WidthCutSetFitnessFunction() = default;

        ~WidthCutSetFitnessFunction() = default;

        htd::FitnessEvaluation *fitness(const htd::IMultiHypergraph &graph, const htd::ITreeDecomposition &decomposition) const override {
            return new htd::FitnessEvaluation(2,
                    -(double) (decomposition.maximumBagSize()),
                    -getMaxCutSetSize(decomposition)
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
                        back_inserter(v)
                );
                sizes = std::max((double) v.size(), sizes);
            }
            return sizes;
        };

        WidthCutSetFitnessFunction *clone(void) const override {;
            return new WidthCutSetFitnessFunction();
        }
    };
}

#endif //GPUSAT_WIDTHCUTSETFITNESSFUNCTION_H
