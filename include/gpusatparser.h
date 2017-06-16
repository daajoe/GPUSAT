#ifndef GPUSAT_PARSER_H
#define GPUSAT_PARSER_H

#include <queue>
#include <main.h>

/**
 * generates a treedec from a given string
 *
 * @param formula the string representation of the tree decomposition
 * @return the tree decomposition
 */
satformulaType parseSatFormula(std::string formula);

/**
 *
 * @param item
 * @param clauses
 * @param clauseSize
 */
void parseClauseLine(std::string item, std::queue<std::queue<cl_long>> *clauses, cl_long &clauseSize);

/**
 *
 * @param satformula
 * @param item
 * @param clauses
 */
void parseProblemLine(satformulaType &satformula, std::string item, std::queue<std::queue<cl_long>> *&clauses);

/**
 * generates a treedec from a given string
 *
 * @param graph the string representation of the tree decomposition
 * @return the tree decomposition
 */
treedecType parseTreeDecomp(std::string graph);

/**
 *
 * @param item
 * @param edges
 */
void parseEdgeLine(std::string item, std::queue<cl_long> **edges);

/**
 *
 * @param ret
 * @param item
 * @param edges
 */
void parseStartLine(treedecType &ret, std::string &item, std::queue<cl_long> **&edges);

/**
 *
 * @param ret
 * @param item
 */
void parseBagLine(treedecType ret, std::string item);

#endif //GPUSAT_PARSER_H
