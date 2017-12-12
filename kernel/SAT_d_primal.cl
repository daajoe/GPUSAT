#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define stype double

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
stype solveIntroduce_(long numV, __global stype *edge, long numVE, __global long *variables, __global long *edgeVariables, long minId, long maxId,
                      long startIDEdge, long id) {
    long otherId = 0;
    long a = 0, b = 0;
    for (b = 0; b < numVE && a < numV; b++) {
        while ((variables[a] != edgeVariables[b])) {
            a++;
        }
        otherId = otherId | (((id >> a) & 1) << b);
        a++;
    };

    if (edge != 0 && otherId >= (minId) && otherId < (maxId)) {
        return edge[otherId - (startIDEdge)];
    } else if (edge == 0 && otherId >= (minId) && otherId < (maxId)) {
        return 0.0;
    } else {
        return -1.0;
    }
}

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
int checkBag(__global long *clauses, __global long *numVarsC, long numclauses, long id, long numV, __global long *variables) {
    long i, varNum = 0;
    long satC = 0, a, b;
    // iterate through all clauses
    for (i = 0; i < numclauses; i++) {
        satC = 0;
        // iterate through clause variables
        for (a = 0; a < numVarsC[i] && !satC; a++) {
            satC = 1;
            //check current variables
            for (b = 0; b < numV; b++) {
                // check if clause is satisfied
                if ((clauses[varNum + a] == variables[b]) ||
                    (clauses[varNum + a] == -variables[b])) {
                    satC = 0;
                    if (clauses[varNum + a] < 0) {
                        //clause contains negative var and var is assigned negative
                        if ((id & (1 << (b))) == 0) {
                            satC = 1;
                            break;
                        }
                    } else {
                        //clause contains positive var and var is assigned positive
                        if ((id & (1 << (b))) > 0) {
                            satC = 1;
                            break;
                        }
                    }
                }
            }
        }
        varNum += numVarsC[i];
        // we have an unsattisifed clause
        if (!satC) {
            return 0;
        }
    }
    return 1;
}

/**
 * Operation to solve a Join node in the decomposition.
 *
 * @param solutions
 *      array to save the number of solutions of the join
 * @param edge1
 *      array containing the number of solutions in the first edge
 * @param edge2
 *      array containing the number of solutions in the second edge
 * @param variables
 *      the variables in the join bag
 * @param edgeVariables1
 *      the variables in the bag of the first edge
 * @param edgeVariables2
 *      the variables in the bag of the second edge
 * @param numV
 *      the number of variables in the join bag
 * @param numVE1
 *      the number of variables in the first edge
 * @param numVE2
 *      the number of variables in the second edge
 */
__kernel void solveJoin(__global stype *solutions, __global stype *edge1, __global stype *edge2, __global long *variables, __global long *edgeVariables1,
                        __global long *edgeVariables2, long numV, long numVE1, long numVE2, long minId1, long maxId1, long minId2,
                        long maxId2, long startIDNode, long startIDEdge1, long startIDEdge2, __global int *sols) {
    long id = get_global_id(0);
    stype tmp = -1, tmp_ = -1;
    // get solution count from first edge
    tmp = solveIntroduce_(numV, edge1, numVE1, variables, edgeVariables1, minId1, maxId1, startIDEdge1, id);
    // get solution count from second edge
    tmp_ = solveIntroduce_(numV, edge2, numVE2, variables, edgeVariables2, minId2, maxId2, startIDEdge2, id);


    // we have some solutions in edge1
    if (tmp >= 0.0) {
        solutions[id - (startIDNode)] *= tmp;
    }

    // we have some solutions in edge2
    if (tmp_ >= 0.0) {
        solutions[id - (startIDNode)] *= tmp_;
    }

    if (solutions[id - (startIDNode)] > 0) {
        *sols = 1;
    }
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
__kernel void solveIntroduce(__global long *clauses, __global long *numVarsC, long numclauses, __global stype *solutions, long numV, __global stype *edge, long numVE,
                             __global long *variables, __global long *edgeVariables, __global long *models, __global long *minId, __global long *maxId,
                             __global long *startIDNode, __global long *startIDEdge, __global int *sols) {
    long id = get_global_id(0);
    stype tmp;
    // get solution count from the edge
    tmp = solveIntroduce_(numV, edge, numVE, variables, edgeVariables, minId, maxId, startIDEdge, id);
    if (tmp > 0.0) {
        //check if assignment satisfies the given clauses
        int sat = checkBag(clauses, numVarsC, numclauses, id, numV, variables);
        if (sat != 1) {
            solutions[id - (*startIDNode)] = 0.0;
        } else {
            (*models) = 1;
            solutions[id - (*startIDNode)] = tmp;
        }
    } else if (tmp == 0.0) {
        //solution count of the edge is 0
        solutions[id - (*startIDNode)] = 0.0;
    }
    if (solutions[id - (*startIDNode)] > 0) {
        *sols = 1;
    }
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
stype solveIntroduceF(__global long *clauses, __global long *numVarsC, long numclauses, long numV, __global stype *edge, long numVE,
                      __global long *variables, __global long *edgeVariables, long minId, long maxId,
                      long startIDEdge, long id) {
    stype tmp;
    if (edge != 0) {
        // get solutions count edge
        tmp = solveIntroduce_(numV, edge, numVE, variables, edgeVariables, minId, maxId, startIDEdge, id);
    } else {
        // no edge - solve leaf
        tmp = 1.0;
    }
    if (tmp > 0.0) {
        // check if assignment satisfies the given clauses
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
__kernel void solveIntroduceForget(__global stype *solsF, __global long *varsF, __global stype *solsE,
                                   long numVE, __global long *varsE, long combinations, long numVF,
                                   long minIdE, long maxIdE, long startIDF,
                                   long startIDE, __global int *sols,
                                   long numVI, __global long *varsI,
                                   __global long *clauses, __global long *numVarsC, long numclauses) {
    long id = get_global_id(0);
    if (numVI != numVF) {
        long templateId = 0;
        // generate templateId
        for (int i = 0, a = 0; i < numVI && a < numVF; i++) {
            if (varsI[i] == varsF[a]) {
                templateId = templateId | (((id >> a) & 1) << i);
                a++;
            }
        }

        // iterate through all corresponding edge solutions
        for (int i = 0; i < combinations; i++) {
            long b = 0, otherId = templateId;
            for (int a = 0; a < numVI; a++) {
                if (b >= numVF || varsI[a] != varsF[b]) {
                    otherId = otherId | (((i >> (a - b)) & 1) << a);
                } else {
                    b++;
                }
            }
            // get solution count of the corresponding assignment in the edge
            solsF[id - (startIDF)] += solveIntroduceF(clauses, numVarsC, numclauses, numVI, solsE, numVE, varsI, varsE, minIdE, maxIdE, startIDE, otherId);
        }
    } else {
        // no forget variables, only introduce
        solsF[id - (startIDF)] += solveIntroduceF(clauses, numVarsC, numclauses, numVI, solsE, numVE, varsI, varsE, minIdE, maxIdE, startIDE, id);
    }
    if (solsF[id - (startIDF)] > 0) {
        *sols = 1;
    }
}