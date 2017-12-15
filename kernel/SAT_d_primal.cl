#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define stype double

// Operation to solve a Introduce node in the decomposition.
stype
solveIntroduce_(long numV, __global stype *edge, long numVE, __global long *variables, __global long *edgeVariables, long minId, long maxId, long startIDEdge, long id) {
    long otherId = 0;
    long a = 0, b = 0;
    for (b = 0; b < numVE & a < numV; b++) {
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

// Operation to check if an assignment satisfies the clauses of a SAT formula.
int checkBag(__global long *clauses, __global long *numVarsC, long numclauses, long id, long numV, __global long *variables) {
    long i, varNum = 0;
    long satC = 0, a, b;
    // iterate through all clauses
    for (i = 0; i < numclauses; i++) {
        satC = 0;
        // iterate through clause variables
        for (a = 0; a < numVarsC[i] & !satC; a++) {
            //check current variables
            for (b = 0; b < numV & !satC; b++) {
                // check if clause is satisfied
                satC |= ((clauses[varNum + a] == variables[b]) | (clauses[varNum + a] == -variables[b])) & (clauses[varNum + a] < 0) & ((id & (1 << (b))) == 0)
                        | ((clauses[varNum + a] == variables[b]) | (clauses[varNum + a] == -variables[b])) & (clauses[varNum + a] > 0) & ((id & (1 << (b))) > 0);
            }
        }
        varNum += numVarsC[i];
        // we have an unsatisfied clause
        if (!satC) {
            return 0;
        }
    }
    return 1;
}

// Operation to solve a Join node in the decomposition.
__kernel void
solveJoin(__global stype *solutions, __global stype *edge1, __global stype *edge2, __global long *variables, __global long *edgeVariables1, __global long *edgeVariables2,
          long numV, long numVE1, long numVE2, long minId1, long maxId1, long minId2, long maxId2, long startIDNode, long startIDEdge1, long startIDEdge2,
          __global int *sols) {
    long id = get_global_id(0);
    stype tmp = -1, tmp_ = -1;
    // get solution count from first edge
    tmp = solveIntroduce_(numV, edge1, numVE1, variables, edgeVariables1, minId1, maxId1, startIDEdge1, id);
    // get solution count from second edge
    tmp_ = solveIntroduce_(numV, edge2, numVE2, variables, edgeVariables2, minId2, maxId2, startIDEdge2, id);

    solutions[id - (startIDNode)] *= (((stype) (tmp < 0.0)) + (tmp >= 0.0) * tmp) * (((stype) (tmp_ < 0.0)) + (tmp_ >= 0.0) * tmp_);
    *sols |= (solutions[id - (startIDNode)] > 0);
}

// Operation to solve a Introduce node in the decomposition.
stype solveIntroduceF(__global long *clauses, __global long *numVarsC, long numclauses, long numV, __global stype *edge, long numVE, __global long *variables,
                      __global long *edgeVariables, long minId, long maxId, long startIDEdge, long id) {
    stype tmp = solveIntroduce_(numV, edge, numVE, variables, edgeVariables, minId, maxId, startIDEdge, id);

    if (tmp > 0.0) {
        // check if assignment satisfies the given clauses
        int sat = checkBag(clauses, numVarsC, numclauses, id, numV, variables);
        return ((double) (sat == 1)) * tmp;
    } else {
        return 0.0;
    }
}

// Operation to solve Introduce and Forget nodes in the decomposition.
__kernel void
solveIntroduceForget(__global stype *solsF, __global long *varsF, __global stype *solsE, long numVE, __global long *varsE, long combinations, long numVF, long minIdE,
                     long maxIdE, long startIDF, long startIDE, __global int *sols, long numVI, __global long *varsI, __global long *clauses, __global long *numVarsC,
                     long numclauses) {
    long id = get_global_id(0);
    long templateId = 0;
    // generate templateId
    for (int i = 0, a = 0; i < numVI & a < numVF; i++) {
        templateId |= (((varsI[i] == varsF[a]) & ((id >> a) & 1)) << i);
        a += (varsI[i] == varsF[a]);
    }

    // iterate through all corresponding edge solutions
    for (int i = 0; i < combinations; i++) {
        long b = 0, otherId = templateId;
        for (int a = 0; a < numVI; a++) {
            otherId |= (((b >= numVF | varsI[a] != varsF[b]) & ((i >> (a - b)) & 1)) << a);
            b += (b < numVF) & (varsI[a] == varsF[b]);
        }
        // get solution count of the corresponding assignment in the edge
        solsF[id - (startIDF)] += solveIntroduceF(clauses, numVarsC, numclauses, numVI, solsE, numVE, varsI, varsE, minIdE, maxIdE, startIDE, otherId);
    }
    *sols |= solsF[id - (startIDF)] > 0;
}


// Operation to solve a Leaf node in the decomposition.
__kernel void
solveIntroduceForgetLeaf(__global stype *solsF, __global long *varsF, __global stype *solsE, long numVE, __global long *varsE, long combinations, long numVF, long minIdE,
                         long maxIdE, long startIDF, long startIDE, __global int *sols, long numVI, __global long *varsI, __global long *clauses, __global long *numVarsC,
                         long numclauses) {
    long id = get_global_id(0);
    long templateId = 0;
    // generate templateId
    for (int i = 0, a = 0; i < numVI & a < numVF; i++) {
        templateId |= (((varsI[i] == varsF[a]) & ((id >> a) & 1)) << i);
        a += (varsI[i] == varsF[a]);
    }

    // iterate through all corresponding edge solutions
    for (int i = 0; i < combinations; i++) {
        long b = 0, otherId = templateId;
        for (int a = 0; a < numVI; a++) {
            otherId |= (((b >= numVF | varsI[a] != varsF[b]) & ((i >> (a - b)) & 1)) << a);
            b += (b < numVF) & (varsI[a] == varsF[b]);
        }
        // get solution count of the corresponding assignment in the edge
        solsF[id - (startIDF)] += checkBag(clauses, numVarsC, numclauses, otherId, numVI, varsI);
    }
    *sols |= solsF[id - (startIDF)] > 0;
}