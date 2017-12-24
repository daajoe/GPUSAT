#ifndef GPUSAT_PARSER_H
#define GPUSAT_PARSER_H
#define alloca __builtin_alloca

#include <queue>
#include <types.h>
#include <unordered_map>

namespace gpusat {
    class CNFParser {
    public:

        CNFParser(bool weighted);

        /**
                 * generates a treedec from a given string
                 *
                 * @param formula
                 *      the string representation of the tree decomposition
                 * @return
                 *      the tree decomposition
                 */
        satformulaType parseSatFormula(std::string formula);

    private:
        /**
         * parses a problem line from the sat formula
         *
         * @param satformula
         *      object containing the sat formula
         * @param item
         *      the line
         * @param clauses
         *      object containing the clauses in the sat formula
         */
        void parseProblemLine(satformulaType &satformula, std::string item);

        void parseWeightLine(std::string item, std::unordered_map<cl_long, solType> &weights);

        bool wmc;
    };

    class TDParser {
    public:
        explicit TDParser(int i, bool b);

        /**
         * generates a treedec from a given string
         *
         * @param graph
         *      the string representation of the tree decomposition
         * @return
         *      the tree decomposition
         */
        treedecType parseTreeDecomp(std::string graph, satformulaType &formula);

        solType defaultWeight = 1.0;

    private:
        /**
         * parse an edge from the tree decomposition
         *
         * @param item
         *      the line
         * @param edges
         *      queue containing all edges
         */
        void parseEdgeLine(std::string item, std::vector<std::vector<cl_long>> &edges);

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
        void parseStartLine(preetreedecType &ret, std::string &item, std::vector<std::vector<cl_long>> &edges);

        /**
         * parses a pag from the tree decomposition
         *
         * @param ret
         *      object containing the tree decomposition
         * @param item
         *      a line from the decomposition
         */
        void parseBagLine(preetreedecType &ret, std::string item);

        /**
         * preprocess the tree decomposition
         *
         * @param decomp
         *      the tree decomposition
         */
        void preprocessDecomp(preebagType *decomp);

        int combineWidth;

        void removeEdges(std::vector<std::vector<cl_long>> &node, cl_long id, cl_long preID);

        void preprocessFacts(preetreedecType &decomp, satformulaType &formula);

        void relableDecomp(preebagType *decomp, cl_long id);

        void relableFormula(satformulaType &formula, cl_long id);

        bool factR;
    };
}
#endif //GPUSAT_PARSER_H
