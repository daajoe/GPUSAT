#ifndef GPUSAT_SATPARSER_H
#define GPUSAT_SATPARSER_H

#include <gpusatutils.h>
#include <queue>

/**
 * generates a treedec from a given string
 *
 * @param formula the string representation of the tree decomposition
 * @return the tree decomposition
 */
satformulaType parseSatFormula(std::string formula);

void parseClauseLine(std::string item, std::queue<std::queue<cl_long>> *clauses, cl_long &clauseSize);

void parseProblemLine(satformulaType &satformula, std::string item, std::queue<std::queue<cl_long>> *&clauses);

#endif //GPUSAT_SATPARSER_H
