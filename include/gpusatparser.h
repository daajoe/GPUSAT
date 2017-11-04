#ifndef GPUSAT_PARSER_H
#define GPUSAT_PARSER_H
#define alloca __builtin_alloca

#include <queue>
#include <types.h>

namespace gpusat {
    class CNFParser {
    public:

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
         * parses a clause line from the sat formula
         *
         * @param item
         *      the line
         * @param clauses
         *      the clauses
         * @param clauseSize
         *      size of each clause
         */
        void parseClauseLine(std::string item, std::queue<std::queue<cl_long>> &clauses, cl_long &clauseSize);

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
        void parseProblemLine(satformulaType &satformula, std::string item, std::queue<std::queue<cl_long>> &clauses);

        void parseWeightLine(std::string item, std::vector<std::pair<cl_long, solType>> &weights);
    };

    class TDParser {
    public:
        TDParser(int i);

        /**
         * generates a treedec from a given string
         *
         * @param graph
         *      the string representation of the tree decomposition
         * @return
         *      the tree decomposition
         */
        treedecType parseTreeDecomp(std::string graph);

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
    };
}
#endif //GPUSAT_PARSER_H
