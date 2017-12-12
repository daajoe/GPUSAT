#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define stype double

int isNotSat(unsigned long assignment, __global long *clause, __global unsigned long *variables);

/**
 * Operation to solve a Join node in the decomposition.
 *
 * @param nSol
 *      array containing the number of solutions for each assignment for the current node
 * @param e1Sol
 *      array containing the number of solutions for each assignment for the first edge
 * @param e2Sol
 *      array containing the number of solutions for each assignment for the second edge
 * @param minIDe1
 *      min id of the first edge
 * @param maxIDe1
 *      max id of the first edge
 * @param minIDe2
 *      min id of the second edge
 * @param maxIDe2
 *      max id of the second edge
 * @param startIDn
 *      start id of the current node
 * @param startIDe1
 *      start id of the first edge
 * @param startIDe2
 *      of the second edge
 * @param numClauses
 *      number of clauses in the current node
 */
__kernel void solveJoin(__global stype *nSol, __global stype *e1Sol, __global stype *e2Sol,
                        unsigned long minIDe1, unsigned long maxIDe1,
                        unsigned long minIDe2, unsigned long maxIDe2,
                        unsigned long startIDn, unsigned long startIDe1, unsigned long startIDe2,
                        unsigned long numClauses,
                        __global double *weights, __global unsigned long *nVars, __global int *sols) {
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
        if (weights != 0) {
            double weight = 1;
            unsigned long assignment = id >> numClauses;
            for (int a = 0; nVars[a] != 0; a++) {
                weight *= weights[((assignment >> a) & 1) > 0 ? nVars[a] * 2 : nVars[a] * 2 + 1];
            }
            nSol[id - (startIDn)] += tmpSol / weight;
        } else {
            nSol[id - (startIDn)] += tmpSol;
        }
    }
    if (nSol[id - (startIDn)] > 0) {
        *sols = 1;
    }
}

/**
 * Operation to solve a Introduce node in the decomposition.
 *
 * @param nSol
 *      array containing the number of solutions for each assignment for the current node
 * @param eSol
 *      array containing the number of solutions for each assignment for the edge
 * @param clauses
 *      array containing the clauses of the current node, negated atoms are negative
 * @param cLen
 *      length of the clauses array
 * @param nVars
 *      array containing the ids of the variables in the current node
 * @param eVars
 *      array containing the ids of the variables in the edge
 * @param numNV
 *      number of variables in the current node
 * @param numEV
 *      number of variables in the edge
 * @param nClauses
 *      array containing the clause ids of the current node
 * @param eClauses
 *      array containing the clause ids of the edge
 * @param numNC
 *      number of clauses in the current node
 * @param numEC
 *      number of clauses in the edge
 * @param startIDn
 *      start id of the current node
 * @param startIDe
 *      start id of the edge
 * @param minID
 *      min id of the edge
 * @param maxID
 *      max id of the edge
 */
__kernel void solveIntroduce(__global stype *nSol, __global stype *eSol,
                             __global long *clauses, unsigned long cLen,
                             __global unsigned long *nVars, __global unsigned long *eVars,
                             unsigned long numNV, unsigned long numEV,
                             __global unsigned long *nClauses, __global unsigned long *eClauses,
                             unsigned long numNC, unsigned long numEC,
                             unsigned long startIDn, unsigned long startIDe,
                             unsigned long minIDe, unsigned long maxIDe, __global double *weights, __global int *sols) {
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

    double weight = 1;

    if (weights != 0) {
        for (b = 0, a = 0; nVars[a] != 0; a++) {
            if ((nVars[a] != eVars[b])) {
                weight *= weights[((assignment >> a) & 1) > 0 ? nVars[a] * 2 : nVars[a] * 2 + 1];
            }
            if (nVars[a] == eVars[b] && eVars[b] != 0) {
                b++;
            }
        }
    }
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
    nSol[id - (startIDn)] *= weight;
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

/**
 * introduce function for the introduce forget operation
 *
 * @param nSol
 *      array containing the number of solutions for each assignment for the current node
 * @param eSol
 *      array containing the number of solutions for each assignment for the edge
 * @param clauses
 *      array containing the clauses of the current node, negated atoms are negative
 * @param cLen
 *      length of the clauses array
 * @param nVars
 *      array containing the ids of the variables in the current node
 * @param eVars
 *      array containing the ids of the variables in the edge
 * @param numNV
 *      number of variables in the current node
 * @param numEV
 *      number of variables in the edge
 * @param nClauses
 *      array containing the clause ids of the current node
 * @param eClauses
 *      array containing the clause ids of the edge
 * @param numNC
 *      number of clauses in the current node
 * @param numEC
 *      number of clauses in the edge
 * @param startIDe
 *      start id of the edge
 * @param minIDe
 *      min id of the edge
 * @param maxIDe
 *      max id of the edge
 */
stype solveIntroduceF(__global stype *eSol,
                      __global long *clauses, unsigned long cLen,
                      __global unsigned long *nVars, __global unsigned long *eVars,
                      unsigned long numNV, unsigned long numEV,
                      __global unsigned long *nClauses, __global unsigned long *eClauses,
                      unsigned long numNC, unsigned long numEC,
                      unsigned long startIDe,
                      unsigned long minIDe, unsigned long maxIDe, __global double *weights,
                      long id) {
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

    double weight = 1;

    if (weights != 0) {
        for (b = 0, a = 0; nVars[a] != 0; a++) {
            if ((nVars[a] != eVars[b])) {
                weight *= weights[((assignment >> a) & 1) > 0 ? nVars[a] * 2 : nVars[a] * 2 + 1];
            }
            if (nVars[a] == eVars[b] && eVars[b] != 0) {
                b++;
            }
        }
    }
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
    tmp *= weight;
    return tmp;
}

/**
 * combination of the introduce and forget operation
 *
 * @param solsF
 *      array containing the number of solutions for each assignment for the current node
 * @param solsE
 *      array containing the number of solutions for each assignment for the edge
 * @param varsF
 *      array containing the ids of the variables in the current node
 * @param varsE
 *      array containing the ids of the variables in the edge node
 * @param numVF
 *      the number of variables in the current node
 * @param numVE
 *      the number of variables in the edge node
 * @param fClauses
 *      array containing the clause ids of the current node
 * @param eClauses
 *      array containing the clause ids of the edge node
 * @param numCF
 *      number of clauses in the current node
 * @param numCE
 *      number of clauses in the edge node
 * @param startIDf
 *      start id of the current node
 * @param startIDe
 *      start id of the edge node
 * @param minIDE
 *      min id of the edge
 * @param maxIDE
 *      max id of the edge
 * @param sols
 *      flag, indicating that there are solutions in the current bag
 * @param varsI
 *      array containing the ids of the variables in the introduce node
 * @param numVI
 *      array containing the number of variables in the introduce node
 * @param iClauses
 *      array containing the clause ids of the introduce node
 * @param numCI
 *      number of clauses in the introduce node
 * @param clauses
 *      array containing the clauses of the current node, negated atoms are negative
 * @param cLen
 *      length of the clauses array
 * @param weights
 *      array containing the weights of each variable
 */
__kernel void solveIntroduceForget(__global stype *solsF, __global stype *solsE,
                                   __global unsigned long *varsF, __global unsigned long *varsE,
                                   unsigned long numVF, unsigned long numVE,
                                   __global unsigned long *fClauses, __global unsigned long *eClauses,
                                   unsigned long numCF, unsigned long numCE,
                                   unsigned long startIDf, unsigned long startIDe,
                                   unsigned long minIDE, unsigned long maxIDE, __global int *sols,
                                   __global unsigned long *varsI, unsigned long numVI, __global unsigned long *iClauses, unsigned long numCI,
                                   __global long *clauses, unsigned long cLen, __global double *weights) {
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
                                                      weights, otherId);
        }
    } else {
        // only solve introduce if there is no forget
        solsF[id - (startIDf)] += solveIntroduceF(solsE, clauses, cLen, varsI, varsE, numVI, numVE, iClauses, eClauses, numCI, numCE, startIDe, minIDE, maxIDE,
                                                  weights, id);
    }
    if (solsF[id - (startIDf)] > 0) {
        *sols = 1;
    }
}