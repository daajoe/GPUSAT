long checkBag(__global long *clauses, __global long *numVarsC, long numclauses, long id, long numV,
              __global long *variables) {
    long i, varNum = 0;
    for (i = 0; i < numclauses; i++) {
        long satC = 0, a, b;
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
            return 1;
        }
    }
    return 0;
}

__kernel void solveJoin(__global long *solutions, __global long *edges, long numSol) {
    long id = get_global_id(0);
    solutions[id] = edges[id] * edges[numSol + id];
}

__kernel void
solveForget(__global long *solutions, __global long *variablesCurrent, __global long *edge, long numVarsEdge,
            __global long *variablesEdge, long combinations, long numVarsCurrent) {
    long id = get_global_id(0), i = 0, a = 0, templateId = 0;
    for (i = 0; i < numVarsEdge && a < numVarsCurrent; i++) {
        if (variablesEdge[i] == variablesCurrent[a]) {
            templateId = templateId | (((id >> a) & 1) << i);
            a++;
        }
    }
    for (i = 0; i < combinations; i++) {
        long b = 0, otherId = templateId;
        for (a = 0; a < numVarsEdge; a++) {
            if (b >= numVarsCurrent || variablesEdge[a] != variablesCurrent[b]) {
                otherId = otherId | (((i >> (a - b)) & 1) << a);
            } else {
                b++;
            }
        }
        solutions[id] += edge[otherId];
    }
}

__kernel void solveLeaf(__global long *clauses, __global long *numVarsC, long numclauses,
                        __global long *solutions, long numV, __global long *variables) {
    long id = get_global_id(0);
    long unsat = checkBag(clauses, numVarsC, numclauses, id, numV, variables);
    solutions[id] = !unsat;
}

__kernel void solveIntroduce(__global long *clauses, __global long *numVarsC, long numclauses,
                             __global long *solutions, long numV, __global long *edge, long numVE,
                             __global long *variables, __global long *edgeVariables) {
    long id = get_global_id(0);
    long a = 0, b = 0;
    long otherId = 0;
    for (b = 0; b < numVE && a < numV; b++) {
        while ((variables[a] != edgeVariables[b])) {
            a++;
        }
        otherId = otherId | (((id >> a) & 1) << b);
        a++;
    };
    if (edge[otherId] > 0) {
        solutions[id] = !checkBag(clauses, numVarsC, numclauses, id, numV, variables) * edge[otherId];
    }
}