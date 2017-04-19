#ifndef GPUSAT_SOLVER_H
#define GPUSAT_SOLVER_H

#include <gpusatutils.h>

using namespace std;

/**
 *
 * @param id thread id
 * @param solutions solutions array
 * @param solN array with n values of solution
 * @param numV number of variables in solution
 * @param edges array containing edges in bag
 * @param numE number of edges
 * @param vertices array of vertices in current bag
 */
void
solveJoin(long id, varIdType *solutions, long numV, long numE, vertexIdType *vertices, varIdType *edges, long numSol);

/**
 *
 * @param id thread id
 * @param solutions solutions array
 * @param solN array with n values of solution
 * @param numV number of variables in solution
 * @param edge array containing the edge in bag
 * @param edgeN array with n values of edge
 * @param numVE number of variables in edge
 * @param numESol number of solutions in edge
 * @param vertices array of vertices in current bag
 */
void solveForget(long id, varIdType *solutions,
                 long numV, varIdType *edge, long numVE, long numESol, vertexIdType *vertices);

void solveLeaf(varIdType *clauses, varIdType *numVarsC, clauseIdType numclauses, long id, varIdType *solutions,
               long numV, vertexIdType *vertices);

void
solveIntroduce(varIdType *clauses, varIdType *numVarsC, clauseIdType numclauses, long id, varIdType *solutions,
               long numV, varIdType *edge, long numVE, long numESol, vertexIdType *vertices);

int
checkBag(varIdType *clauses, varIdType *numVarsC, clauseIdType numclauses, long id, long numV, vertexIdType *vertices);

void solveProblem(treedecType &decomp, satformulaType &formula, bagType &node);

#endif //GPUSAT_SOLVER_H
