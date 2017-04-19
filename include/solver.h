#ifndef GPUSAT_SOLVER_H
#define GPUSAT_SOLVER_H

#include "gpusatutils.h"

using namespace std;

int solveProblem(treedecType &decomp, satformulaType &formula, bagType &node);

void solveNode(treedecType *decomp, satformulaType *formula, bagType *node, long id);

#endif //GPUSAT_SOLVER_H
