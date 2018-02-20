#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define stype double

typedef struct {
    long numC;
    long numCE1;
    long numCE2;
    long minId1;
    long maxId1;
    long minId2;
    long maxId2;
    long startIDNode;
    long startIDEdge1;
    long startIDEdge2;
    long numV;
    long numVE1;
    long numVE2;
} sJVars;

typedef struct {
    long numCI;
    long numCE;
    long numVF;
    long numVI;
    long numVE;
    long combinations;
    long minIdE;
    long maxIdE;
    long startIDF;
    long startIDE;
    long numCF;
} sIFVars;

typedef struct {
    long numV;
    long numVE;
    long minId;
    long maxId;
    long numC;
    long numCE;
    long startIDEdge;
    long id;
} sIVars;

typedef struct {
    long numC;
    long numV;
    long clauseId;
} cFVars;


stype checkFalsifiable(cFVars params,
                       __global long *clauseVars,
                       __global long *numVarsC,
                       __global long *vars) {
    // iterate through all clauses
    for (long b = 0; b < params.numV; b++) {
        int negative = 0, positive = 0;
        long varNum = 0;
        long v = vars[b];
        for (long i = 0; i < params.numC; i++) {
            if ((params.clauseId >> i) & 1) {
                // iterate through clause
                for (long a = 0; a < numVarsC[i]; a++) {
                    long c = clauseVars[varNum + a];
                    if (c == v) {
                        // variable is contained positive in a clause
                        positive = 1;
                        break;
                    }
                    if (c == -v) {
                        // variable is contained negative in a clause
                        negative = 1;
                        break;
                    }
                }
            }
            varNum += numVarsC[i];
        }
        if (positive && negative) {
            return 0.0;
        }
    }
    return 1.0;
}

double solveIntroduce_(sIVars params,
                       __global long *clauseIds, __global long *clauseIdsE,
                       __global long *numVarsC, __global long *numVarsCE,
                       __global stype *solsE,
                       __global long *vars, __global long *varsE,
                       __global long *clauseVars, __global long *clauseVarsE) {
    long otherId = 0;
    long a = 0, b = 0;
    for (b = 0; b < params.numCE && a < params.numC; b++) {
        while (clauseIds[a] != clauseIdsE[b]) {
            a++;
        }
        otherId = otherId | (((params.id >> a) & 1) << b);
        a++;
    };

    if (solsE != 0 && otherId >= (params.minId) && otherId < (params.maxId)) {
        int a = 0;
        stype tmp = solsE[otherId - (params.startIDEdge)];
        for (int b = 0; a < params.numC; a++) {
            while (b < params.numCE && clauseIds[a] == clauseIdsE[b]) {
                b++;
                a++;
            }
            if (params.numCE == 0 || clauseIds[a] != clauseIdsE[b]) {
                if ((params.id >> a) & 1) {
                    //|var(C) /\ (var(x(t'))\var(A\{C}))|
                    long startIDC = 0;
                    for (int i = 0; i < a; ++i) {
                        startIDC += numVarsC[i];
                    }
                    long exponent = 0;
                    //var(C)
                    for (int j = 0; j < numVarsC[a]; ++j) {
                        long var = abs(clauseVars[j + startIDC]);
                        int found = 0;
                        int subfound = 0;
                        //var(x(t'))
                        for (int k = 0; k < params.numVE; ++k) {
                            if (var == varsE[k]) {
                                found = 1;
                                break;
                            }
                        }
                        long startClauses = 0;
                        for (int k = 0; k < a; ++k) {
                            for (int i = 0; i < numVarsC[k]; ++i) {
                                if (var == abs(clauseVars[i + startClauses])) {
                                    found = 1;
                                    break;
                                }
                            }
                            startClauses += numVarsC[k];
                        }
                        //var(A\{C})
                        long startIDCE = 0;
                        for (int i = 0; i < params.numCE && !subfound; ++i) {
                            if ((otherId >> i) & 1) {
                                for (int l = 0; l < numVarsCE[i] && !subfound; ++l) {
                                    if (var == abs(clauseVarsE[l + startIDCE])) {
                                        subfound = 1;
                                        break;
                                    }
                                }
                            }
                            startIDCE += numVarsCE[i];
                        }

                        startClauses = 0;
                        for (int k = 0; k < a; ++k) {
                            if ((params.id >> k) & 1) {
                                for (int i = 0; i < numVarsC[k]; ++i) {
                                    if (var == abs(clauseVars[i + startClauses])) {
                                        subfound = 1;
                                        break;
                                    }
                                }
                            }
                            startClauses += numVarsC[k];
                        }

                        if (found && !subfound) {
                            ++exponent;
                        }
                    }
                    tmp /= 1 << exponent;
                } else {
                    //|var(C)\var(x(t'))|
                    long startIDC = 0;
                    for (int i = 0; i < a; ++i) {
                        startIDC += numVarsC[i];
                    }
                    long exponent = 0;
                    for (int j = 0; j < numVarsC[a]; ++j) {
                        int found = 0;
                        long varNum = 0;
                        long var = abs(clauseVars[j + startIDC]);
                        //var(x(t'))
                        for (int l = 0; l < params.numVE; ++l) {
                            if (var == varsE[l]) {
                                found = 1;
                                break;
                            }
                        }
                        long startClauses = 0;
                        for (int k = 0; k < a; ++k) {
                            for (int i = 0; i < numVarsC[k]; ++i) {
                                if (var == abs(clauseVars[i + startClauses])) {
                                    found = 1;
                                    break;
                                }
                            }
                            startClauses += numVarsC[k];
                        }
                        if (!found) {
                            ++exponent;
                        }
                    }
                    tmp *= 1 << exponent;
                }
            }
        };
        if (tmp > 0.0) {
            cFVars cFparams;
            cFparams.clauseId = params.id;
            cFparams.numC = params.numC;
            cFparams.numV = params.numV;
            tmp *= checkFalsifiable(cFparams, clauseVars, numVarsC, vars);
        }
        return tmp;
    } else if (solsE == 0 && otherId >= (params.minId) && otherId < (params.maxId)) {
        return 0.0;
    } else {
        return -1.0;
    }
}


__kernel void solveJoin(sJVars params,
                        __global stype *sol, __global stype *solE1, __global stype *solE2,
                        __global int *sols,
                        __global long *clauseVars,
                        __global long *clauseIds, __global long *clauseIdsE1, __global long *clauseIdsE2,
                        __global long *numVarsC,
                        __global long *variables, __global long *varsE1, __global long *varsE2,
                        __global long *numVarsCE1, __global long *numVarsCE2,
                        __global long *clauseVarsE1, __global long *clauseVarsE2) {
    long id = get_global_id(0);
    stype tmp = -1, tmp_ = -1;
    long otherId = 0;
    // get solution count from first edge
    sIVars iparams1;
    iparams1.numV = params.numV;
    iparams1.numVE = params.numVE1;
    iparams1.minId = params.minId1;
    iparams1.maxId = params.maxId1;
    iparams1.numC = params.numC;
    iparams1.numCE = params.numCE1;
    iparams1.startIDEdge = params.startIDEdge1;
    iparams1.id = id;
    tmp = solveIntroduce_(iparams1, clauseIds, clauseIdsE1, numVarsC, numVarsCE1, solE1, variables, varsE1, clauseVars, clauseVarsE1);
    // get solution count from second edge
    sIVars iparams2;
    iparams2.numV = params.numV;
    iparams2.numVE = params.numVE2;
    iparams2.minId = params.minId2;
    iparams2.maxId = params.maxId2;
    iparams2.numC = params.numC;
    iparams2.numCE = params.numCE2;
    iparams2.startIDEdge = params.startIDEdge2;
    iparams2.id = id;
    tmp_ = solveIntroduce_(iparams2, clauseIds, clauseIdsE2, numVarsC, numVarsCE2, solE1, variables, varsE2, clauseVars, clauseVarsE2);
    // we have some solutions in edge1

    if (tmp >= 0.0) {
        // |var(x(t))\var(A)|
        long exponent = 0;
        for (int j = 0; j < params.numV; ++j) {
            int found = 0;
            long varNum = 0;
            for (int i = 0; i < params.numC && !found; i++) {
                if (((id >> i) & 1) == 1) {
                    for (int a = 0; a < numVarsC[i] && !found; a++) {
                        if (variables[j] == abs(clauseVars[varNum + a])) {
                            found = 1;
                        }
                    }
                }
                varNum += numVarsC[i];
            }
            if (!found) {
                exponent++;
            }

        }
        sol[id - (params.startIDNode)] *= tmp;
        sol[id - (params.startIDNode)] /= 1 << exponent;

    }

    // we have some solutions in edge2
    if (tmp_ >= 0.0) {
        sol[id - (params.startIDNode)] *= tmp_;
    }

    if (sol[id - (params.startIDNode)] > 0) {
        *sols = 1;
    }
}

stype solveIntroduceF(sIVars params,
                      __global long *clauseIds, __global long *clauseIdsE,
                      __global long *numVarsC, __global long *numVarsCE,
                      __global stype *edge,
                      __global long *vars, __global long *varsE,
                      __global long *clauseVars, __global long *clauseVarsE) {
    stype tmp;
    if (edge != 0) {
        // get solutions count edge
        sIVars iparams1;
        iparams1.numC = params.numC;
        iparams1.numCE = params.numCE;
        iparams1.minId = params.minId;
        iparams1.maxId = params.maxId;
        iparams1.startIDEdge = params.startIDEdge;
        iparams1.id = params.id;
        iparams1.numV = params.numV;
        iparams1.numVE = params.numVE;
        tmp = solveIntroduce_(params, clauseIds, clauseIdsE, numVarsC, numVarsCE, edge, vars, varsE, clauseVars, clauseVarsE);
    } else {
        // no edge - solve leaf
        long exponent = 0;
        for (int j = 0; j < params.numV; ++j) {
            int found = 0;
            long varNum = 0;
            for (int i = 0; i < params.numC && !found; i++) {
                if (((params.id >> i) & 1) == 1) {
                    for (int a = 0; a < numVarsC[i] && !found; a++) {
                        if (vars[j] == abs(clauseVars[varNum + a])) {
                            found = 1;
                        }
                    }
                }
                varNum += numVarsC[i];
            }
            if (!found) {
                exponent++;
            }

        }
        tmp = 1 << exponent;
        cFVars cFparams;
        cFparams.clauseId = params.id;
        cFparams.numC = params.numC;
        cFparams.numV = params.numV;
        tmp *= checkFalsifiable(cFparams, clauseVars, numVarsC, vars);
    }
    return tmp;
}

__kernel void
solveIntroduceForget(sIFVars params,
                     __global stype *solsF, __global stype *solsE,
                     __global long *clauseIdsF, __global long *clauseIdsI, __global long *clauseIdsE,
                     __global long *varsF, __global long *varsE, __global long *varsI,
                     __global int *sols,
                     __global long *numVarsCE, __global long *numVarsCI,
                     __global long *clauseVarsI, __global long *clauseVarsE) {
    long id = get_global_id(0);
    if (params.numCI != params.numCF) {
        long templateId = 0;
        // generate templateId
        for (int i = 0, a = 0; i < params.numCI && a < params.numCF; i++) {
            if (clauseIdsI[i] == clauseIdsF[a]) {
                templateId = templateId | (((id >> a) & 1) << i);
                a++;
            }
        }
        for (int i = 0; i < params.combinations; ++i) {
            long b = 0, otherId = templateId;
            for (int a = 0; a < params.numCI; a++) {
                if (b >= params.numCF || clauseIdsI[a] != clauseIdsF[b]) {
                    otherId = otherId | (((i >> (a - b)) & 1) << a);
                } else {
                    b++;
                }
            }
            // get solution count of the corresponding assignment in the edge
            sIVars iparams;
            iparams.numV = params.numVI;
            iparams.numVE = params.numVE;
            iparams.minId = params.minIdE;
            iparams.maxId = params.maxIdE;
            iparams.numC = params.numCI;
            iparams.numCE = params.numCE;
            iparams.startIDEdge = params.startIDE;
            iparams.id = otherId;
            solsF[id - (params.startIDF)] += solveIntroduceF(iparams, clauseIdsI, clauseIdsE, numVarsCI, numVarsCE, solsE, varsI, varsE, clauseVarsI, clauseVarsE) * (1 - ((popcount(i) % 2 == 1) * 2));
        }
    } else {
        // no forget variables, only introduce
        sIVars iparams;
        iparams.numV = params.numVI;
        iparams.numVE = params.numVE;
        iparams.minId = params.minIdE;
        iparams.maxId = params.maxIdE;
        iparams.numC = params.numCI;
        iparams.numCE = params.numCE;
        iparams.startIDEdge = params.startIDE;
        iparams.id = id;
        solsF[id - (params.startIDF)] += solveIntroduceF(iparams, clauseIdsI, clauseIdsE, numVarsCI, numVarsCE, solsE, varsI, varsE, clauseVarsI, clauseVarsE);
    }
    if (solsF[id - (params.startIDF)] > 0) {
        *sols = 1;
    }
}


