#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define stype double

stype solveIntroduce_(long numV, __global stype *edge, long numVE, __global long *variables, __global long *edgeVariables, long minId, long maxId,
                      long startIDEdge, __global double *weights, long id) {
    long otherId = 0;
    long a = 0, b = 0;
    double weight = 1.0;
    for (b = 0; b < numVE && a < numV; b++) {
        while ((variables[a] != edgeVariables[b])) {
            a++;
        }
        otherId = otherId | (((id >> a) & 1) << b);
        a++;
    };

    if (weights != 0) {
        for (b = 0, a = 0; a < numV; a++) {
            if ((variables[a] != edgeVariables[b])) {
                weight *= weights[((id >> a) & 1) > 0 ? variables[a] * 2 : variables[a] * 2 + 1];
            }
            if ((variables[a] == edgeVariables[b]) && (b < (numVE - 1))) {
                b++;
            }
        }
    }

    if (edge != 0 && otherId >= (minId) && otherId < (maxId)) {
        return edge[otherId - (startIDEdge)] * weight;
    } else if (edge == 0 && otherId >= (minId) && otherId < (maxId)) {
        return 0.0;
    } else {
        return -1.0;
    }

}

int checkBag(__global long *clauses, __global long *numVarsC, long numclauses, long id, long numV, __global long *variables) {
    long i, varNum = 0;
    long satC = 0, a, b;
    for (i = 0; i < numclauses; i++) {
        satC = 0;
        for (a = 0; a < numVarsC[i] && !satC; a++) {
            satC = 1;
            for (b = 0; b < numV; b++) {
                if ((clauses[varNum + a] == variables[b]) ||
                    (clauses[varNum + a] == -variables[b])) {
                    satC = 0;
                    if (clauses[varNum + a] < 0) {
                        if ((id & (1 << (b))) == 0) {
                            satC = 1;
                            break;
                        }
                    } else {
                        if ((id & (1 << (b))) > 0) {
                            satC = 1;
                            break;
                        }
                    }
                }
            }
        }
        varNum += numVarsC[i];
        if (!satC) {
            return 0;
        }
    }
    return 1;
}

__kernel void solveJoin(__global stype *solutions, __global stype *edge1, __global stype *edge2, __global long *variables, __global long *edgeVariables1,
                        __global long *edgeVariables2, long numV, long numVE1, long numVE2, __global long *minId1, __global long *maxId1, __global long *minId2,
                        __global long *maxId2, __global long *startIDNode, __global long *startIDEdge1, __global long *startIDEdge2, __global double *weights, __global
                        int *sols) {
    long id = get_global_id(0);
    stype tmp = -1, tmp_ = -1;
    double weight = 1;
    tmp = solveIntroduce_(numV, edge1, numVE1, variables, edgeVariables1, *minId1, *maxId1, *startIDEdge1, weights, id);
    tmp_ = solveIntroduce_(numV, edge2, numVE2, variables, edgeVariables2, *minId2, *maxId2, *startIDEdge2, weights, id);
    if (weights != 0) {
        for (int a = 0; a < numV; a++) {
            weight *= weights[((id >> a) & 1) > 0 ? variables[a] * 2 : variables[a] * 2 + 1];
        }
    }
    if (tmp >= 0.0) {
        solutions[id - (*startIDNode)] *= tmp;
        solutions[id - (*startIDNode)] /= weight;
    }
    if (tmp_ >= 0.0) {
        solutions[id - (*startIDNode)] *= tmp_;
    }
    if (solutions[id - (*startIDNode)] > 0) {
        *sols = 1;
    }
}

__kernel void solveIntroduce(__global long *clauses, __global long *numVarsC, long numclauses, __global stype *solutions, long numV, __global stype *edge, long numVE,
                             __global long *variables, __global long *edgeVariables, __global long *models, __global long *minId, __global long *maxId,
                             __global long *startIDNode, __global long *startIDEdge, __global double *weights, __global int *sols) {
    long id = get_global_id(0);
    stype tmp;
    tmp = solveIntroduce_(numV, edge, numVE, variables, edgeVariables, minId, maxId, startIDEdge, weights, id);
    if (tmp > 0.0) {
        int sat = checkBag(clauses, numVarsC, numclauses, id, numV, variables);
        if (sat != 1) {
            solutions[id - (*startIDNode)] = 0.0;
        } else {
            (*models) = 1;
            solutions[id - (*startIDNode)] = tmp;
        }
    } else if (tmp == 0.0) {
        solutions[id - (*startIDNode)] = 0.0;
    }
    if (solutions[id - (*startIDNode)] > 0) {
        *sols = 1;
    }
}

stype solveIntroduceF(__global long *clauses, __global long *numVarsC, long numclauses, long numV, __global stype *edge, long numVE,
                      __global long *variables, __global long *edgeVariables, long minId, long maxId,
                      long startIDEdge, __global double *weights, long id) {
    stype tmp;
    tmp = solveIntroduce_(numV, edge, numVE, variables, edgeVariables, minId, maxId, startIDEdge, weights, id);
    if (tmp > 0.0) {
        int sat = checkBag(clauses, numVarsC, numclauses, id, numV, variables);
        if (sat != 1) {
            return 0.0;
        } else {
            return tmp;
        }
    } else {
        return 0.0;
    }
}

__kernel void solveIntroduceForget(__global stype *solsF, __global long *varsF, __global stype *solsE,
                                   long numVE, __global long *varsE, long combinations, long numVF,
                                   long minIdE, long maxIdE, long startIDF,
                                   long startIDE, __global int *sols,
                                   long numVI, __global long *varsI,
                                   __global long *clauses, __global long *numVarsC, long numclauses, __global double *weights) {
    long id = get_global_id(0), templateId = 0;
    for (int i = 0, a = 0; i < numVI && a < numVF; i++) {
        if (varsI[i] == varsF[a]) {
            templateId = templateId | (((id >> a) & 1) << i);
            a++;
        }
    }

    for (int i = 0; i < combinations; i++) {
        long b = 0, otherId = templateId;
        for (int a = 0; a < numVI; a++) {
            if (b >= numVF || varsI[a] != varsF[b]) {
                otherId = otherId | (((i >> (a - b)) & 1) << a);
            } else {
                b++;
            }
        }
        solsF[id - (startIDF)] += solveIntroduceF(clauses, numVarsC, numclauses, numVI, solsE, numVE, varsI, varsE, minIdE, maxIdE, startIDE, weights, otherId);

    }
    if (solsF[id - (startIDF)] > 0) {
        *sols = 1;
    }
}