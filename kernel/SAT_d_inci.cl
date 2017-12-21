#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define stype double

int isNotSat(unsigned long assignment, __global long *clause, __global unsigned long *variables);

// Operation to solve a Join node in the decomposition.
__kernel void
solveJoin(__global stype *nSol, __global stype *e1Sol, __global stype *e2Sol, unsigned long minIDe1, unsigned long maxIDe1, unsigned long minIDe2, unsigned long maxIDe2,
          unsigned long startIDn, unsigned long startIDe1, unsigned long startIDe2, unsigned long numClauses, __global unsigned long *nVars, __global long *sols) {
    unsigned long combinations = ((unsigned long) exp2((double) numClauses));
    unsigned long start2 = 0, end2 = combinations - 1;

    unsigned long id = get_global_id(0);
    unsigned long mask = id & (((unsigned long) exp2((double) numClauses)) - 1);
    unsigned long templateID = id >> numClauses << numClauses;
    double tmpSol = 0;
    //sum up the solution count for all subsets of Clauses (A1,A2) where the intersection of A1 and A2 = A
    for (; start2 < combinations && e2Sol[(templateID | start2) - (startIDe2)] == 0; start2++);
    for (; end2 > 0 && e2Sol[(templateID | end2) - (startIDe2)] == 0; end2--);
    for (int a = 0; a < combinations; a++) {
        if ((templateID | a) >= minIDe1 && (templateID | a) < maxIDe1 && e1Sol[(templateID | a) - (startIDe1)] != 0) {
            for (int b = start2; b <= end2; b++) {
                if (((a | b)) == mask && ((templateID | b) >= minIDe2 && (templateID | b) < maxIDe2) && e2Sol[(templateID | b) - (startIDe2)] != 0) {
                    tmpSol += e1Sol[(templateID | a) - (startIDe1)] * e2Sol[(templateID | b) - (startIDe2)];
                }
            }
        }
    }
    if (tmpSol != 0.0) {
        nSol[id - (startIDn)] += tmpSol;
    }
    if (nSol[id - (startIDn)] > 0) {
        *sols = 1;
    }
}

// Operation to solve a Introduce node in the decomposition.
__kernel void
solveIntroduce(__global stype *nSol, __global stype *eSol, __global long *clauses, unsigned long cLen, __global unsigned long *nVars, __global unsigned long *eVars,
               unsigned long numNV, unsigned long numEV, __global unsigned long *nClauses, __global unsigned long *eClauses, unsigned long numNC, unsigned long numEC,
               unsigned long startIDn, unsigned long startIDe, unsigned long minIDe, unsigned long maxIDe, __global long *sols) {
    unsigned long id = get_global_id(0);
    unsigned long assignment = id >> numNC, templateID = 0;
    unsigned long a = 0, b = 0, c = 0, i = 0, notSAT = 0, base = 0;
    //check clauses
    for (a = 0, b = 0, i = 0; a < numNC; i++) {
        if (i == 0 || clauses[i] == 0) {
            if (clauses[i] == 0) i++;
            if (nClauses[a] == eClauses[b]) {
                b++;
            } else if (isNotSat(assignment, &clauses[i], nVars) == ((id >> a) & 1)) {
                nSol[id - startIDn] = 0.0;
                return;
            }
            a++;
        }
    }
    unsigned long d = 0;
    int baseSum = 0;
    //check variables
    for (i = 0, c = 0; i < cLen; i++) {
        if (clauses[i] == 0) {
            baseSum = 0;
            if (nClauses[c] == eClauses[d]) {
                d++;
            }
            c++;
        } else {
            for (a = 0; a < numNV; a++) {
                if ((((id >> c) & 1) == 0) && (clauses[i] == nVars[a] * (((assignment >> a) & 1) > 0 ? 1 : -1))) {
                    nSol[id - startIDn] = 0.0;
                    return;
                }
                if ((baseSum == 0) && (nClauses[c] == eClauses[d]) && (((id >> c) & 1) == 1) &&
                    (clauses[i] == nVars[a] * (((assignment >> a) & 1) > 0 ? 1 : -1))) {
                    base++;
                    baseSum = 1;
                }
            }
        }
    }

    //generate template variables
    for (b = 0, a = 0; a < numNV; a++) {
        if (nVars[a] == eVars[b]) {
            templateID |= ((id >> (a + numNC)) & 1) << (b + numEC);
            b++;
        }
    }

    //generate template clauses
    for (b = 0, a = 0; a < numNC; a++) {
        if (nClauses[a] == eClauses[b]) {
            templateID |= ((id >> a) & 1) << b;
            b++;
        }
    }

    unsigned long combinations = (unsigned long) exp2((double) base);
    unsigned long otherID = templateID, nc = 0, ec = 0, x = 0, index = 0, rec;

    if (numNV != numEV) {
        for (i = 0, c = 0; i < combinations; i++) {
            otherID = templateID;
            index = 0;

            for (ec = 0, nc = 0, x = 0; nc < numNC; nc++, x++) {
                rec = 0;
                if (eClauses[ec] == nClauses[nc]) {
                    for (; clauses[x] != 0; x++) {
                        for (a = 0, b = 0; a < numNV && rec == 0; a++) {
                            if (clauses[x] == (nVars[a] * (((assignment >> a) & 1) > 0 ? 1 : -1))) {
                                otherID &= ~(((i >> index) & 1) << ec);
                                index++;
                                rec = 1;
                            }
                            if (nVars[a] == eVars[b]) {
                                b++;
                            } else {
                            }
                        }
                    }
                    ec++;
                } else {
                    for (; clauses[x] != 0; x++);
                }
            }

            if (otherID >= (minIDe) && otherID < (maxIDe)) {
                nSol[id - (startIDn)] += eSol[otherID - (startIDe)];
            }
        }
    } else {
        if (otherID >= (minIDe) && otherID < (maxIDe)) {
            nSol[id - (startIDn)] += eSol[otherID - (startIDe)];
        }
    }
    if (nSol[id - (startIDn)] > 0) {
        *sols = 1;
    }

}

//check if Clause is not Satisfiable
int isNotSat(unsigned long assignment, __global long *clause, __global unsigned long *variables) {
    int a = 0, i = 0;
    for (a = 0; variables[a] != 0; a++) {
        for (i = 0; clause[i] != 0; i++) {
            if (clause[i] == variables[a] || clause[i] == -variables[a]) {
                if ((clause[i] > 0 && ((assignment >> a) & 1) == 1) ||
                    (clause[i] < 0 && ((assignment >> a) & 1) == 0)) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

// introduce function for the introduce forget operation
stype solveIntroduceF(__global stype *eSol, __global long *clauses, unsigned long cLen, __global unsigned long *nVars, __global unsigned long *eVars, unsigned long numNV,
                      unsigned long numEV, __global unsigned long *nClauses, __global unsigned long *eClauses, unsigned long numNC, unsigned long numEC,
                      unsigned long startIDe, unsigned long minIDe, unsigned long maxIDe, long id) {
    unsigned long assignment = id >> numNC, templateID = 0;
    unsigned long a = 0, b = 0, c = 0, i = 0, notSAT = 0, base = 0;
    //check clauses
    for (a = 0, b = 0, i = 0; a < numNC; i++) {
        if (i == 0 || clauses[i] == 0) {
            if (clauses[i] == 0) i++;
            if (nClauses[a] == eClauses[b]) {
                b++;
            } else if (isNotSat(assignment, &clauses[i], nVars) == ((id >> a) & 1)) {
                return 0.0;
            }
            a++;
        }
    }
    unsigned long d = 0;
    int baseSum = 0;
    //check variables
    for (i = 0, c = 0; i < cLen; i++) {
        if (clauses[i] == 0) {
            baseSum = 0;
            if (nClauses[c] == eClauses[d]) {
                d++;
            }
            c++;
        } else {
            for (a = 0; a < numNV; a++) {
                if ((((id >> c) & 1) == 0) && (clauses[i] == nVars[a] * (((assignment >> a) & 1) > 0 ? 1 : -1))) {
                    return 0.0;
                }
                if ((baseSum == 0) && (nClauses[c] == eClauses[d]) && (((id >> c) & 1) == 1) &&
                    (clauses[i] == nVars[a] * (((assignment >> a) & 1) > 0 ? 1 : -1))) {
                    base++;
                    baseSum = 1;
                }
            }
        }
    }

    //template variables
    for (b = 0, a = 0; a < numNV; a++) {
        if (nVars[a] == eVars[b]) {
            templateID |= ((id >> (a + numNC)) & 1) << (b + numEC);
            b++;
        }
    }

    //template clauses
    for (b = 0, a = 0; a < numNC; a++) {
        if (nClauses[a] == eClauses[b]) {
            templateID |= ((id >> a) & 1) << b;
            b++;
        }
    }

    unsigned long combinations = (unsigned long) exp2((double) base);
    unsigned long otherID = templateID, nc = 0, ec = 0, x = 0, index = 0, rec;

    stype tmp = 0.0;
    if (numNV != numEV) {
        for (i = 0, c = 0; i < combinations; i++) {
            otherID = templateID;
            index = 0;

            for (ec = 0, nc = 0, x = 0; nc < numNC; nc++, x++) {
                rec = 0;
                if (eClauses[ec] == nClauses[nc]) {
                    for (; clauses[x] != 0; x++) {
                        for (a = 0, b = 0; a < numNV && rec == 0; a++) {
                            if (clauses[x] == (nVars[a] * (((assignment >> a) & 1) > 0 ? 1 : -1))) {
                                otherID &= ~(((i >> index) & 1) << ec);
                                index++;
                                rec = 1;
                            }
                            if (nVars[a] == eVars[b]) {
                                b++;
                            } else {
                            }
                        }
                    }
                    ec++;
                } else {
                    for (; clauses[x] != 0; x++);
                }
            }

            if (otherID >= minIDe && otherID < maxIDe) {
                if (eSol != 0) {
                    tmp += eSol[otherID - startIDe];
                } else {
                    tmp += 1.0;
                }
            }
        }
    } else {
        if (otherID >= minIDe && otherID < maxIDe) {
            if (eSol != 0) {
                tmp += eSol[otherID - startIDe];
            } else {
                tmp += 1.0;
            }
        }
    }
    return tmp;
}

// combination of the introduce and forget operation
__kernel void
solveIntroduceForget(__global stype *solsF, __global stype *solsE, __global unsigned long *varsF, __global unsigned long *varsE, unsigned long numVF, unsigned long numVE,
                     __global unsigned long *fClauses, __global unsigned long *eClauses, unsigned long numCF, unsigned long numCE, unsigned long startIDf,
                     unsigned long startIDe, unsigned long minIDE, unsigned long maxIDE, __global long *sols, __global unsigned long *varsI, unsigned long numVI,
                     __global unsigned long *iClauses, unsigned long numCI, __global long *clauses, unsigned long cLen) {
    unsigned long id = get_global_id(0);
    unsigned long a = 0, b = 0, templateId = 0, i = 0;
    unsigned long combinations = (unsigned long) exp2((double) numVI - numVF);
    if (numVI != numVF || numCI != numCF) {
        //generate template clauses
        for (a = 0, b = 0; a < numCI; a++) {
            if (fClauses[b] == iClauses[a]) {
                templateId = templateId | (((id >> b) & 1) << a);
                b++;
            } else {
                templateId = templateId | (1 << a);
            }
        }
        //generate template variables
        for (a = 0, b = 0; a < numVI && b < numVF; a++) {
            if (varsF[b] == varsI[a]) {
                templateId = templateId | (((id >> (b + numCF)) & 1) << (a + numCI));
                b++;
            }
        }

        // iterate through all corresponding assignments in the edge
        for (i = 0; i < combinations; i++) {
            long b = 0, otherId = templateId;
            for (a = 0; a < numVI; a++) {
                if (b >= numVF || varsI[a] != varsF[b]) {
                    otherId = otherId | (((i >> (a - b)) & 1) << (a + numCI));
                } else {
                    b++;
                }
            }
            // get solution count from edge
            solsF[id - (startIDf)] += solveIntroduceF(solsE, clauses, cLen, varsI, varsE, numVI, numVE, iClauses, eClauses, numCI, numCE, startIDe, minIDE, maxIDE,
                                                      otherId);
        }
    } else {
        // only solve introduce if there is no forget
        solsF[id - (startIDf)] += solveIntroduceF(solsE, clauses, cLen, varsI, varsE, numVI, numVE, iClauses, eClauses, numCI, numCE, startIDe, minIDE, maxIDE, id);
    }
    if (solsF[id - (startIDf)] > 0) {
        *sols = 1;
    }
}