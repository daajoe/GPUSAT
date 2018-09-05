#ifndef GPUSAT_PARSER_H
#define GPUSAT_PARSER_H

#include <queue>
#include <types.h>
#include <unordered_map>

namespace gpusat {
    class CNFParser {
    public:

        /**
         *  Constructor for an CNF parser.
         *
         * @param weighted    indicates if weights should be assiciated with literals
         */
        CNFParser(bool weighted);

        /**
         * generates a sat formula from a given string
         *
         * @param formula   the string representation of the sat formula
         * @return          the sat formula
         */
        satformulaType parseSatFormula(std::string formula);

    private:
        bool wmc;

        /**
         * parses a problem line of the sat formula
         *
         * @param satformula    object containing the sat formula
         * @param item          the line
         * @param clauses       object containing the clauses in the sat formula
         */
        void parseProblemLine(satformulaType &satformula, std::string item);

        /**
         * parses a weight line of the sat formula
         *
         * @param  item         the line
         * @param weights      map containing the weights for each literal
         */
        void parseWeightLine(std::string item, std::unordered_map<cl_long, cl_double> &weights);

        /**
         * parses a clause line of the sat formula
         *
         * @param ret
         * @param item    the line
         * @param clause
         */
        void parseClauseLine(satformulaType &ret, std::string &item, std::vector<cl_long> *clause);
    };

    class TDParser {
    public:
        explicit TDParser(int i, bool b, int i1);

        /**
         * generates a treedec from a given string
         *
         * @param graph     the string representation of the tree decomposition
         * @return          the tree decomposition
         */
        treedecType parseTreeDecomp(std::string graph, satformulaType &formula, graphTypes gType);

        cl_double defaultWeight = 1.0;
        int combineWidth;
        cl_long preWidth = 0;
        cl_long postWidth = 0;
        cl_long preCut = 0;
        cl_long postCut = 0;
        cl_long preJoinSize = 0;
        cl_long postJoinSize = 0;
        cl_long preNumBags = 0;
        cl_long postNumBags = 0;

        void iterateDecompPre(bagType &bag);

        void iterateDecompPost(bagType &bag);

    private:
        bool factR;

        /**
         * parse an edge from the tree decomposition
         *
         * @param item      the line
         * @param edges     queue containing all edges
         */
        void parseEdgeLine(std::string item, std::vector<std::vector<cl_long>> &edges);

        /**
         * parses the start line from the tree decomposition
         *
         * @param ret       the tree decomposition
         * @param item      the line
         * @param edges     queue containing all edges
         */
        void parseStartLine(treedecType &ret, std::string &item, std::vector<std::vector<cl_long>> &edges);

        /**
         * parses a pag from the tree decomposition
         *
         * @param ret       object containing the tree decomposition
         * @param item      a line from the decomposition
         */
        void parseBagLine(treedecType &ret, std::string item);

        static void removeEdges(std::vector<std::vector<cl_long>> &node, cl_long id, cl_long preID);
    };
}
#endif //GPUSAT_PARSER_H
