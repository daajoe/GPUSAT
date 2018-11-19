#ifndef GPUSAT_DECOMPOSER_H
#define GPUSAT_DECOMPOSER_H

#include <string>
#include <htd/Hypergraph.hpp>
#include <htd/Helpers.hpp>
#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    class Decomposer {
    public:
        static std::string computeDecomposition(std::string formula, htd::ITreeDecompositionFitnessFunction *fitness, size_t n);

        static void parseProblemLine(std::string line, htd::Hypergraph &hypergraph);

        static void parseClauseLine(std::string line, htd::Hypergraph &hypergraph);
    };
}
#endif //GPUSAT_DECOMPOSER_H
