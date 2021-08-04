#ifndef GPUSAT_PARSER_H
#define GPUSAT_PARSER_H

#define GPU_HOST_ATTR

#include <queue>
#include <unordered_map>


#include <gpusat_types.h>

namespace gpusat {
    class CNFParser {
    public:

        /**
         * Constructor for an CNF parser.
         *
         * @param weighted
         *      indicates if weights should be assiciated with literals
         */
        CNFParser(bool weighted);

        /**
         * generates a sat formula from a given string
         *
         * @param formula
         *      the string representation of the sat formula
         * @return
         *      the sat formula
         */
        satformulaType parseSatFormula(std::istream& formula);

    private:
        bool wmc;

        /**
         * parses a problem line of the sat formula
         *
         * @param satformula
         *      object containing the sat formula
         * @param item
         *      the line
         * @param clauses
         *      object containing the clauses in the sat formula
         */
        void parseProblemLine(satformulaType &satformula, std::string item);

        /**
         * parses a weight line of the sat formula
         *
         * @param item
         *      the line
         * @param weights
         *      map containing the weights for each literal
         */
        void parseWeightLine(std::string item, std::unordered_map<int64_t, double> &weights);

        /**
         * parses a clause line of the sat formula
         *
         * @param ret
         *      the sat formula
         * @param item
         *      the clause line in cnf format
         * @param clause
         *      the clauses
         */
        void parseClauseLine(satformulaType &ret, std::string &item);

        /**
         * parses a solution line (not present in cnf format but some preprocessors use it)
         *
         * @param item
         *      the solution line
         */
        void parseSolutionLine(std::string item);
    };

    class TDParser {
    public:
        TDParser() {};

        /**
         * generates a treedec from a given string
         *
         * @param graph
         *      the string representation of the tree decomposition
         * @param removed_facts
         *      Sorted list of facts that have been removed during preprocessing.
         *      Remove them from nodes and relabel variables accordingly.
         * @return
         *      the tree decomposition
         */
        // IF NEEDED, implement construct a decomposition from the string,
        // then use htd_to_bags from decomposer.
        treedecType parseTreeDecomp(std::istream& graph, std::vector<int64_t>& removed_facts);

    private:

        /**
         * parse an edge from the tree decomposition
         *
         * @param item
         *      the line
         * @param edges
         *      queue containing all edges
         */
        void parseEdgeLine(std::string& item, std::vector<std::vector<int64_t>> &edges);

        /**
         * parses the start line from the tree decomposition
         *
         * @param ret
         *      the tree decomposition
         * @param item
         *      the line
         * @param edges
         *      queue containing all edges
         */
        void parseStartLine(treedecType &ret, std::string &item, std::vector<std::vector<int64_t>> &edges);

        /**
         * parses a pag from the tree decomposition
         *
         * @param ret
         *      object containing the tree decomposition
         * @param item
         *      a line from the decomposition
         */
        BagType parseBagLine(std::string& item);


        /**
         * Remove facts from a bag and relabel its variables.
         */

        static void removeFactsFromDecomposition(BagType& bag, std::vector<int64_t>& to_remove, const std::vector<int64_t>& relabel_map);
    };
}
#endif //GPUSAT_PARSER_H
