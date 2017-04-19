#ifndef GPUSAT_SOLVER_H
#define GPUSAT_SOLVER_H

#include <gpusatutils.h>

void solveProblem(treedecType decomp, satformulaType formula, bagType node, cl::Context context, cl::Kernel kernel,
                  cl::Program program, cl::CommandQueue commandQueue);

#endif //GPUSAT_SOLVER_H
