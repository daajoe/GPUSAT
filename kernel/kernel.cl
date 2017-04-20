int checkBag(__global long *clauses, __global long *numVarsC, long numclauses, long id, long numV,
             __global long *vertices) {
    long i, varNum = 0;
    int unsat = 0;
    for (i = 0; i < numclauses && !unsat; i++) {
        int satC = 0;
        long a;
        for (a = 0; a < numVarsC[i] && !satC; a++) {
            int found = 0;
            long b;
            for (b = 0; b < numV && !satC; b++) {
                if ((clauses[varNum + a] == vertices[b]) ||
                    (clauses[varNum + a] == -vertices[b])) {
                    found = 1;
                    if (clauses[varNum + a] < 0) {
                        if ((id & (1 << (numV - b - 1))) >> (numV - b - 1) == 0) {
                            satC = 1;
                        }
                    } else {
                        if ((id & (1 << (numV - b - 1))) >> (numV - b - 1) == 1) {
                            satC = 1;
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

__kernel void solveJoin(__global long *solutions, long numV, long numE, __global long *vertices,
                        __global long *edges, long numSol) {
    long id = get_global_id(0);
    long i;
    for (i = 0; i < numV; i++) {
        if ((id & (1 << (numV - i - 1))) >> (numV - i - 1)) {
            solutions[(numV + 1) * id + i] = vertices[i];
        } else {
            solutions[(numV + 1) * id + i] = vertices[i] * -1;
        }
    }
    long n1 = edges[(numV + 1) * id + numV];
    long n2 = edges[(numV + 1) * numSol + (numV + 1) * id + numV];
    solutions[(numV + 1) * id + numV] = n1 * n2;
    for (i = 2; i < numE; i++) {
        long n = edges[(numV + 1) * numSol * i + (numV + 1) * id + numV];
        solutions[(numV + 1) * id + numV] = solutions[(numV + 1) * id + numV] * n;
    }
}

__kernel void solveForget(__global long *solutions, long numV, __global long *edge, long numVE,
                          long numESol, __global long *vertices) {
    long id = get_global_id(0);
    long i;
    for (i = 0; i < numV; i++) {
        if ((id & (1 << (numV - i - 1))) >> (numV - i - 1)) {
            solutions[(numV + 1) * id + i] = vertices[i];
        } else {
            solutions[(numV + 1) * id + i] = vertices[i] * -1;
        }
    }
    solutions[(numV + 1) * id + numV] = 0;
    for (i = 0; i < numESol; i++) {
        __global long *otherSol = &edge[(numVE + 1) * i];
        if (otherSol[numVE] != 0) {
            long a = 0, b = 0;
            int eq = 1;
            for (a = 0; a < numV; a++) {
                long var1 = solutions[(numV + 1) * id + a], var2 = otherSol[b];
                while (((solutions[(numV + 1) * id + a] != otherSol[b]) &&
                        (solutions[(numV + 1) * id + a] != -otherSol[b]))) {
                    b++;
                    var2 = otherSol[b];
                }
                if ((var1 != var2)) {
                    eq = 0;
                    break;
                }
                b++;
            }
            if (eq) {
                solutions[(numV + 1) * id + numV] = solutions[(numV + 1) * id + numV] + otherSol[numVE];
            }
        }
    }
}

__kernel void solveLeaf(__global long *clauses, __global long *numVarsC, long numclauses,
                        __global long *solutions, long numV, __global long *vertices) {
    long id = get_global_id(0);
    int unsat = checkBag(clauses, numVarsC, numclauses, id, numV, vertices);
    long i;
    solutions[(numV + 1) * id + numV] = !unsat;
    for (i = 0; i < numV; i++) {
        if ((id & (1 << (numV - i - 1))) >> (numV - i - 1)) {
            solutions[(numV + 1) * id + i] = vertices[i];
        } else {
            solutions[(numV + 1) * id + i] = vertices[i] * -1;
        }
    }
}

__kernel void solveIntroduce(__global long *clauses, __global long *numVarsC, long numclauses,
                             __global long *solutions, long numV, __global long *edge, long numVE,
                             long numESol, __global long *vertices) {
    long id = get_global_id(0);
    int unsat = checkBag(clauses, numVarsC, numclauses, id, numV, vertices);
    long i;
    for (i = 0; i < numV; i++) {
        long vert = vertices[i];
        if ((id & (1 << (numV - i - 1))) >> (numV - i - 1)) {
            solutions[(numV + 1) * id + i] = vert;
        } else {
            solutions[(numV + 1) * id + i] = vert * -1;
        }
    }
    solutions[(numV + 1) * id + numV] = 0;
    if (!unsat) {
        for (i = 0; i < numESol; i++) {
            __global long *otherSol = &edge[(numVE + 1) * i];
            long a = 0, b = 0;
            int eq = 1;
            for (b = 0; b < numVE; b++) {
                long var1 = solutions[(numV + 1) * id + a], var2 = otherSol[b];
                while (((var1 != var2) && (-var1 != var2))) {
                    a++;
                    var1 = solutions[(numV + 1) * id + a];
                }
                if ((var1 != var2)) {
                    eq = 0;
                    break;
                }
                a++;
            }
            if (eq) {
                solutions[(numV + 1) * id + numV] = otherSol[numVE];
                break;
            }
        }
    }
}