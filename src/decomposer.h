#ifndef GPUSAT_DECOMPOSER_H
#define GPUSAT_DECOMPOSER_H

#include <string>
#include <htd/Hypergraph.hpp>
#include <htd/Helpers.hpp>
#include <htd/ITreeDecompositionFitnessFunction.hpp>

#include <gpusat_types.h>

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
        static treedecType computeDecomposition(const satformulaType& formula, htd::ITreeDecompositionFitnessFunction *fitness, size_t n);

    private:
        /**
         * Build a tree of bags (treedecType) from a tree decomposition.
         */
        static treedecType htd_to_bags(const htd::ITreeDecomposition& decomposition, const struct satformulaType& formula);


        /**
         * Build a hypergraph for passing to HTD from a given formula.
         */
        static void gpusat_formula_to_hypergraph(htd::Hypergraph& hypergraph, const satformulaType& formula);
    };
}
#endif //GPUSAT_DECOMPOSER_H
