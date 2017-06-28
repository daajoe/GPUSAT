/**
 * Operation to check if an assignment satisfies the clauses of a SAT formula.
 *
 * @param clauses
 *      the clauses in the SAT formula
 * @param numVarsC
 *      array containing the number of Variables in each clause
 * @param numclauses
 *      the number of clauses in the sat formula
 * @param id
 *      the id of the thread - used to get the variable assignment
 * @param numV
 *      the number of variables
 * @param variables
 *      a vector containing the ids of the variables
 * @return
 *      1 - if the assignment satisfies the formula
 *      0 - if the assignment doesn't satisfy the formula
 */
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

/**
 * Operation to solve a Join node in the decomposition.
 *
 * @param solutions
 *      array for saving the number of models for each assignment
 * @param edges
 *      array containing the solutions of the nodes in the join
 * @param numSol
 *      number of solutions in the edge nodes
 */
__kernel void solveJoin(__global long *solutions, __global long *edges, long numSol) {
    long id = get_global_id(0);
    solutions[id] = edges[id] * edges[numSol + id];
}

/**
 * Operation to solve a Forget node in the decomposition.
 *
 * @param solutions
 *      array for saving the number of models for each assignment
 * @param variablesCurrent
 *      array containing the ids of the variables in the current bag
 * @param edge
 *      array containing the solutions in the last node
 * @param numVarsEdge
 *      number of variables in the edge bag
 * @param variablesEdge
 *      array containing the ids of the variables in the next bag
 * @param combinations
 *      the number of solutions that relate to this bag from the next bag
 * @param numVarsCurrent
 *      number of variables in the current bag
 */
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

/**
 * Operation to solve a Leaf node in the decomposition.
 *
 * @param clauses
 *      array containing the clauses of the sat formula
 * @param numVarsC
 *      array containing the number of variables for each clause
 * @param numclauses
 *      number of clauses in the sat formula
 * @param solutions
 *      array for saving the number of models for each assignment
 * @param numV
 *      number of variables in the bag
 * @param variables
 *      array containing the ids of the variables in the bag
 */
__kernel void solveLeaf(__global long *clauses, __global long *numVarsC, long numclauses,
                        __global long *solutions, long numV, __global long *variables) {
    long id = get_global_id(0);
    long unsat = checkBag(clauses, numVarsC, numclauses, id, numV, variables);
    solutions[id] = !unsat;
}

/**
 * Operation to solve a Introduce node in the decomposition.
 *
 * @param clauses
 *      array containing the clauses in the sat formula
 * @param numVarsC
 *      array containing the number of variables for each clause
 * @param numclauses
 *      the number of clauses
 * @param solutions
 *      array for saving the number of models for each assignment
 * @param numV
 *      the number of variables in the current bag
 * @param edge
 *      the number of models for each assignment of the next bag
 * @param numVE
 *      the number of variables in the next bag
 * @param variables
 *      the ids of the variables in the current bag
 * @param edgeVariables
 *      the ids of the variables in the next bag
 */
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
    } else {
        solutions[id] = 0;
    }
}