#ifndef GPUSAT_SATPARSER_H
#define GPUSAT_SATPARSER_H

#include <gpusatutils.h>

/**
 * generates a treedec from a given string
 *
 * @param formula the string representation of the tree decomposition
 * @return the tree decomposition
 */
satformulaType parseSatFormula(string formula);

void parseClauseLine(string item, queue<queue<varIdType>> *clauses);

void parseProblemLine(satformulaType &satformula, string item, queue<queue<varIdType>> *&clauses);

#endif //GPUSAT_SATPARSER_H
