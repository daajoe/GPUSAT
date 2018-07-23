#if defined(cl_khr_fp64)
# pragma OPENCL EXTENSION cl_khr_fp64: enable
# pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#elif defined(cl_amd_fp64)
# pragma OPENCL EXTENSION cl_amd_fp64: enable
# pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#else
# error double precision is not supported
#endif
#define stype double

//#define __kernel
//#define __global

typedef struct {
    long id;
    double count;
} tableElement;

double getCount(long id, __global tableElement *elements, long size) {
    for (long i = 0; i < size; i++) {
        if (elements[(id + i) % size].id == id) {
            double c = elements[(id + i) % size].count;
            if (c == -1.0) {
                return 1.0;
            } else {
                return c;
            }
        }
    }
    return 0;
}

void setCount(long id, __global tableElement *elements, long size, double count, __global long *counts) {
    for (long i = 0; i < size; i++) {
        long oldId = atom_cmpxchg(&(elements[(id + i) % size].id), -1, id);
        if (oldId == -1) {
            elements[(id + i) % size].count = count;
            atom_add(counts, 1);
            return;
        } else if (oldId == id) {
            if (elements[(id + i) % size].count == -1.0) {
                atom_add(counts, 1);
            } else if (elements[(id + i) % size].count > 0 && counts == 0) {
                atom_sub(counts, 1);
            }
            elements[(id + i) % size].count = count;
            return;
        }
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
stype solveIntroduce_(long numV, __global tableElement *edge, long numVE, __global long *variables, __global long *edgeVariables, long minId, long maxId,
                      long startIDEdge, __global double *weights, long id, long tableSize) {
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

    //weighted model count
    if (weights != 0) {
        for (b = 0, a = 0; a < numV; a++) {
            if (edgeVariables == 0 || (variables[a] != edgeVariables[b])) {
                weight *= weights[((id >> a) & 1) > 0 ? variables[a] * 2 : variables[a] * 2 + 1];
            }
            if (edgeVariables != 0 && (variables[a] == edgeVariables[b]) && (b < (numVE - 1))) {
                b++;
            }
        }
    }

    if (edge != 0 && otherId >= (minId) && otherId < (maxId)) {
        return getCount(otherId, edge, tableSize) * weight;
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
__kernel void solveJoin(__global tableElement *solutions, __global tableElement *edge1, __global tableElement *edge2, __global long *variables, __global long *edgeVariables1,
                        __global long *edgeVariables2, long numV, long numVE1, long numVE2, long minId1, long maxId1, long minId2,
                        long maxId2, long startIDNode, long startIDEdge1, long startIDEdge2, __global double *weights, __global long *sols, long tableSize, long tableSizeE1, long tableSizeE2) {
    long id = get_global_id(0);
    stype tmp = -1, tmp_ = -1;
    double weight = 1;
    // get solution count from first edge
    tmp = solveIntroduce_(numV, edge1, numVE1, variables, edgeVariables1, minId1, maxId1, startIDEdge1, weights, id, tableSizeE1);
    // get solution count from second edge
    tmp_ = solveIntroduce_(numV, edge2, numVE2, variables, edgeVariables2, minId2, maxId2, startIDEdge2, weights, id, tableSizeE2);
    // weighted model count
    if (weights != 0) {
        for (int a = 0; a < numV; a++) {
            weight *= weights[((id >> a) & 1) > 0 ? variables[a] * 2 : variables[a] * 2 + 1];
        }
    }

    // we have some solutions in edge1
    if (tmp >= 0.0) {
        setCount(id, solutions, tableSize, getCount(id, solutions, tableSize) * tmp, sols);
        setCount(id, solutions, tableSize, getCount(id, solutions, tableSize) / weight, sols);
    }

    // we have some solutions in edge2
    if (tmp_ >= 0.0) {
        setCount(id, solutions, tableSize, getCount(id, solutions, tableSize) * tmp_, sols);
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
stype solveIntroduceF(__global long *clauses, __global long *numVarsC, long numclauses, long numV, __global tableElement *edge, long numVE,
                      __global long *variables, __global long *edgeVariables, long minId, long maxId,
                      long startIDEdge, __global double *weights, long id, long tableSize) {
    stype tmp;
    if (edge != 0) {
        // get solutions count edge
        tmp = solveIntroduce_(numV, edge, numVE, variables, edgeVariables, minId, maxId, startIDEdge, weights, id, tableSize);
    } else {
        // no edge - solve leaf
        tmp = 1.0;

        //weighted model count
        if (weights != 0) {
            for (int i = 0; i < numV; i++) {
                tmp *= weights[((id >> i) & 1) > 0 ? variables[i] * 2 : variables[i] * 2 + 1];
            }
        }
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
__kernel void solveIntroduceForget(__global tableElement *solsF, __global long *varsF, __global tableElement *solsE,
                                   long numVE, __global long *varsE, long combinations, long numVF,
                                   long minIdE, long maxIdE, long startIDF,
                                   long startIDE, __global long *sols,
                                   long numVI, __global long *varsI,
                                   __global long *clauses, __global long *numVarsC, long numclauses, __global double *weights, long tableSizeF, long tableSizeE) {
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
        for (long i = 0; i < combinations; i++) {
            long b = 0, otherId = templateId;
            for (int a = 0; a < numVI; a++) {
                if (b >= numVF || varsI[a] != varsF[b]) {
                    otherId = otherId | (((i >> (a - b)) & 1) << a);
                } else {
                    b++;
                }
            }
            // get solution count of the corresponding assignment in the edge
            double tmp = solveIntroduceF(clauses, numVarsC, numclauses, numVI, solsE, numVE, varsI, varsE, minIdE, maxIdE, startIDE, weights, otherId, tableSizeE) + getCount(id, solsF, tableSizeF);
            if (tmp > 0) setCount(id, solsF, tableSizeF, tmp, sols);
        }
    } else {
        // no forget variables, only introduce
        double tmp = solveIntroduceF(clauses, numVarsC, numclauses, numVI, solsE, numVE, varsI, varsE, minIdE, maxIdE, startIDE, weights, id, tableSizeE) + getCount(id, solsF, tableSizeF);
        if (tmp > 0) setCount(id, solsF, tableSizeF, tmp, sols);

    }
}