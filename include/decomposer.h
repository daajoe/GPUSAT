#ifndef GPUSAT_DECOMPOSER_H
#define GPUSAT_DECOMPOSER_H

#include <string>
#include <htd/Hypergraph.hpp>
#include <htd/Helpers.hpp>
#include <htd/ITreeDecompositionFitnessFunction.hpp>

namespace gpusat {
    class Decomposer {
    public:
        /**
         * computes the decomposition of the primal graph of the given formula
         *
         * @param formula
         *      the formula in cnf format
         * @param fitness
         *      the fitness function
         * @param n
         *      number of iterations for the fitness function
         * @return
         *      the decomposition in td format
         */
        static std::string computeDecomposition(std::string formula, htd::ITreeDecompositionFitnessFunction *fitness, size_t n);

    private:
        /**
         * parses the problem line from the cnf formula
         *
         * @param line
         *      the problem line
         * @param hypergraph
         *      the graph to generate the decomposition from
         */
        static void parseProblemLine(std::string line, htd::Hypergraph &hypergraph);

        /**
         * parses a clause line from the cnf formula
         *
         * @param line
         *      the clause line
         * @param hypergraph
         *      the graph to generate the decomposition from
         */
        static void parseClauseLine(std::string line, htd::Hypergraph &hypergraph);
    };
}
#endif //GPUSAT_DECOMPOSER_H
