#ifndef GPUSAT_SATPARSER_H
#define GPUSAT_SATPARSER_H

#include <gpusatutils.h>

/**
 * generates a treedec from a given string
 *
 * @param formula the string representation of the tree decomposition
 * @return the tree decomposition
 */
satformulaType parseSatFormula(basic_string<char> formula);

void parseClauseLine(basic_string<char> item, queue<queue<long>> *clauses);

void parseProblemLine(satformulaType &satformula, basic_string<char> item, queue<queue<long>> *&clauses);

#endif //GPUSAT_SATPARSER_H
