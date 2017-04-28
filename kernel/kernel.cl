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
    solutions[2 * id + 0] = id;
    int n1 = edges[2 * id + 1];
    int n2 = edges[2 * numSol + 2 * id + 1];
    solutions[2 * id + 1] = n1 * n2;
}

__kernel void solveForget(__global int *solutions, int numV, __global int *variables, __global int *edge, int numVE,
                          __global int *edgeVariables) {
    int id = get_global_id(0);
    if (edge[2 * id + 1] > 0) {
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
        solutions[2 * otherId + 0] = id;
        //printf("add: from: %i, to: %i, value: %i \n", id, otherId, edge[2 * id + 1]);
        atomic_add(&(solutions[2 * otherId + 1]), edge[2 * id + 1]);
    }
}

__kernel void solveLeaf(__global int *clauses, __global int *numVarsC, int numclauses,
                        __global int *solutions, int numV, __global int *variables) {
    int id = get_global_id(0);
    int unsat = checkBag(clauses, numVarsC, numclauses, id, numV, variables);
    solutions[2 * id + 0] = id;
    solutions[2 * id + 1] = !unsat;
}

__kernel void solveIntroduce(__global int *clauses, __global int *numVarsC, int numclauses,
                             __global int *solutions, int numV, __global int *edge, int numVE,
                             __global int *variables, __global int *edgeVariables) {
    int id = get_global_id(0);
    solutions[2 * id + 0] = id;
    solutions[2 * id + 1] = 0;
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
    if (edge[2 * otherId + 1] > 0) {
        int unsat = checkBag(clauses, numVarsC, numclauses, id, numV, variables);
        solutions[2 * id + 1] = !unsat * edge[2 * otherId + 1];
    }
}