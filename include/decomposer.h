#ifndef GPUSAT_DECOMPOSER_H
#define GPUSAT_DECOMPOSER_H

#include <string>
#include <htd/Hypergraph.hpp>
#include <htd/Helpers.hpp>
#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    /**
     *
     */
    class Decomposer {
    public:
        /**
         *
         * @param formula
         * @param fitness
         * @param n
         * @return
         */
        static std::string computeDecomposition(std::string formula, htd::ITreeDecompositionFitnessFunction *fitness, size_t n);

        /**
         *
         * @param line
         * @param hypergraph
         */
        static void parseProblemLine(std::string line, htd::Hypergraph &hypergraph);

        /**
         *
         * @param line
         * @param hypergraph
         */
        static void parseClauseLine(std::string line, htd::Hypergraph &hypergraph);
    };
}
#endif //GPUSAT_DECOMPOSER_H
