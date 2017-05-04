int checkBag(__global int *clauses, __global int *numVarsC, int numclauses, int id, int numV,
             __global int *variables) {
    int i, varNum = 0;
    int unsat = 0;
    for (i = 0; i < numclauses && !unsat; i++) {
        int satC = 0;
        int a;
        for (a = 0; a < numVarsC[i] && !satC; a++) {
            int found = 0;
            int b;
            for (b = 0; b < numV && !satC; b++) {
                if ((clauses[varNum + a] == variables[b]) ||
                    (clauses[varNum + a] == -variables[b])) {
                    found = 1;
                    if (clauses[varNum + a] < 0) {
                        if ((id & (1 << (numV - b - 1))) >> (numV - b - 1) == 0) {
                            satC = 1;
                            break;
                        }
                    } else {
                        if ((id & (1 << (numV - b - 1))) >> (numV - b - 1) == 1) {
                            satC = 1;
                            break;
                        }
                    }
                }
            }
            if (!found) {
                satC = 1;
            }
        }
        varNum += numVarsC[i];
        if (!satC) {
            unsat = 1;
        }
    }
    return unsat;
}

__kernel void solveJoin(__global int *solutions, __global int *edges, int numSol) {
    int id = get_global_id(0);
    solutions[id] = id;
    int n1 = edges[id];
    int n2 = edges[numSol + id];
    int i = id;
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    i = (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    solutions[id] = edges[id] == -1 || edges[numSol + id] == -1 ? (-1) : (n1 + n2 - i);
}

__kernel void solveForget(__global int *solutions, int numV, __global int *variables, __global int *edge, int numVE,
                          __global int *edgeVariables) {
    int id = get_global_id(0);
    if (edge[id] >= 0) {
        int a = 0, b = 0;
        int otherId = 0;
        for (a = 0; a < numV; a++) {
            int var1 = variables[a], var2 = edgeVariables[b];
            while ((var1 != var2)) {
                b++;
                var2 = edgeVariables[b];
            }
            otherId = otherId | (((id & (1 << (numVE - b - 1))) >> (numVE - b - 1)) << (numV - a - 1));
        }
        atomic_cmpxchg(&(solutions[otherId]), -2, edge[id]);
    }
}

__kernel void solveLeaf(__global int *clauses, __global int *numVarsC, int numclauses,
                        __global int *solutions, int numV, __global int *variables) {
    int id = get_global_id(0);
    int unsat = checkBag(clauses, numVarsC, numclauses, id, numV, variables);
    if (unsat) {
        solutions[id] = -1;
    } else {
        int i = id;
        i = i - ((i >> 1) & 0x55555555);
        i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
        solutions[id] = (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    }
}

__kernel void solveIntroduce(__global int *clauses, __global int *numVarsC, int numclauses,
                             __global int *solutions, int numV, __global int *edge, int numVE,
                             __global int *variables, __global int *edgeVariables) {
    int id = get_global_id(0);
    solutions[id] = 0;
    int a = 0, b = 0;
    int otherId = 0;
    for (b = 0; b < numVE && a < numV; b++) {
        int var1 = variables[a], var2 = edgeVariables[b];
        while ((var1 != var2)) {
            a++;
            var1 = variables[a];
        }
        otherId = otherId | (((id & (1 << (numV - a - 1))) >> (numV - a - 1)) << (numVE - b - 1));
        a++;
    };
    int n1 = id;
    n1 = n1 - ((n1 >> 1) & 0x55555555);
    n1 = (n1 & 0x33333333) + ((n1 >> 2) & 0x33333333);
    n1 = (((n1 + (n1 >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    int n2 = otherId;
    n2 = n2 - ((n2 >> 1) & 0x55555555);
    n2 = (n2 & 0x33333333) + ((n2 >> 2) & 0x33333333);
    n2 = (((n2 + (n2 >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    if (edge[otherId] >= 0) {
        int unsat = checkBag(clauses, numVarsC, numclauses, id, numV, variables);
        if (unsat || solutions[id] < 0) {
            solutions[id] = -1;
        } else {
            solutions[id] = n1 - n2 + edge[otherId];
        }
    } else {
        solutions[id] = -1;
    }
}