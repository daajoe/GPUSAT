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
                        __global unsigned long *minIDe1, __global unsigned long *maxIDe1,
                        __global unsigned long *minIDe2, __global unsigned long *maxIDe2,
                        __global unsigned long *startIDn, __global unsigned long *startIDe1, __global unsigned long *startIDe2,
                        __global unsigned long *numClauses,
                        __global double *weights, __global unsigned long *nVars, __global int *sols) {
    unsigned long id = get_global_id(0);
    unsigned long mask = id & (((unsigned long) exp2((double) *numClauses)) - 1);
    unsigned long combinations = ((unsigned long) exp2((double) *numClauses));
    unsigned long templateID = id >> *numClauses << *numClauses;
    unsigned long assignment = id >> *numClauses;
    double tmpSol = 0;
    double weight = 1;
    unsigned long i = 0;
    if (weights != 0) {
        for (int a = 0; nVars[a] != 0; a++) {
            weight *= weights[((assignment >> a) & 1) > 0 ? nVars[a] * 2 : nVars[a] * 2 + 1];
        }
    }
    //sum up all subsets of Clauses (A1,A2) where the intersection of A1 and A2 = A
    unsigned long start2 = 0, end2 = combinations - 1;
    for (; start2 < combinations && e2Sol[(templateID | start2) - (*startIDe2)] == 0; start2++);
    for (; end2 > 0 && e2Sol[(templateID | end2) - (*startIDe2)] == 0; end2--);
    for (int a = 0; a < combinations; a++) {
        if ((templateID | a) >= *minIDe1 && (templateID | a) < *maxIDe1 && e1Sol[(templateID | a) - (*startIDe1)] != 0) {
            for (int b = start2; b <= end2; b++) {
                if (((a | b)) == mask && ((templateID | b) >= *minIDe2 && (templateID | b) < *maxIDe2) && e2Sol[(templateID | b) - (*startIDe2)] != 0) {
                    tmpSol += e1Sol[(templateID | a) - (*startIDe1)] * e2Sol[(templateID | b) - (*startIDe2)];
                }
            }
        }
    }
    if (tmpSol != 0.0) {
        nSol[id - (*startIDn)] += tmpSol / weight;
    }
    if (nSol[id - (*startIDn)] > 0) {
        *sols = 1;
    }
}

/**
 * Operation to solve a Forget node in the decomposition.
 *
 * @param nSol
 *      array containing the number of solutions for each assignment for the current node
 * @param eSol
 *      array containing the number of solutions for each assignment for the edge
 * @param nVars
 *      array containing the variable ids of the current node
 * @param eVars
 *      array containing the variable ids of the edge
 * @param numNVars
 *      array containing the number of variables in the current node
 * @param numEVars
 *      array containing the number of variables in the edge
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
__kernel void solveForget(__global stype *nSol, __global stype *eSol,
                          __global unsigned long *nVars, __global unsigned long *eVars,
                          __global unsigned long *numNVars, __global unsigned long *numEVars,
                          __global unsigned long *nClauses, __global unsigned long *eClauses,
                          __global unsigned long *numNC, __global unsigned long *numEC,
                          __global unsigned long *startIDn, __global unsigned long *startIDe,
                          __global unsigned long *minID, __global unsigned long *maxID, __global int *sols) {
    unsigned long id = get_global_id(0);
    unsigned long a = 0, b = 0, templateId = 0, i = 0;
    unsigned long combinations = (unsigned long) exp2((double) *numEVars - *numNVars);
    //clauses
    for (a = 0, b = 0; a < *numEC; a++) {
        if (nClauses[b] == eClauses[a]) {
            templateId = templateId | (((id >> b) & 1) << a);
            b++;
        } else {
            templateId = templateId | (1 << a);
        }
    }
    //variables
    for (a = 0, b = 0; a < *numEVars && b < *numNVars; a++) {
        if (nVars[b] == eVars[a]) {
            templateId = templateId | (((id >> (b + *numNC)) & 1) << (a + *numEC));
            b++;
        }
    }

    stype tmp, tmp_;
    for (i = 0; i < combinations; i++) {
        long b = 0, otherId = templateId;
        for (a = 0; a < *numEVars; a++) {
            if (b >= *numNVars || eVars[a] != nVars[b]) {
                otherId = otherId | (((i >> (a - b)) & 1) << (a + *numEC));
            } else {
                b++;
            }
        }
        if (otherId >= (*minID) && otherId < (*maxID)) {
            nSol[id - (*startIDn)] += eSol[otherId - (*startIDe)];
        }
    }
    if (nSol[id - (*startIDn)] > 0) {
        *sols = 1;
    }
}

/**
 * Operation to solve a Leaf node in the decomposition.
 *
 * @param nSol
 *      array containing the number of solutions for each assignment for the current node
 * @param clauses
 *      array containing the clauses of the current node, negated atoms are negative
 * @param nVars
 *      array containing the ids of the variables in the current node
 * @param numNC
 *      number of clauses in the current node
 * @param startID
 *      start id of the current node
 * @param models
 *      set to 1 if models are found
 */
__kernel void solveLeaf(__global stype *nSol,
                        __global long *clauses,
                        __global unsigned long *nVars,
                        __global unsigned long *numNC,
                        __global unsigned long *startID,
                        __global unsigned long *models,
                        __global double *weights, __global int *sols) {
    unsigned long id = get_global_id(0);
    unsigned long assignment = id >> *numNC;
    unsigned long i = 0, a = 0;
    double weight = 1;

    if (weights != 0) {
        for (a = 0; nVars[a] != 0; a++) {
            weight *= weights[((assignment >> a) & 1) > 0 ? nVars[a] * 2 : nVars[a] * 2 + 1];
        }
    }
    for (i = 0, a = 0; a < *numNC; i++) {
        if (i == 0 || clauses[i] == 0) {
            if (clauses[i] == 0) i++;
            if (isNotSat(assignment, &clauses[i], nVars) == ((id >> a) & 1)) {
                nSol[id - *startID] = 0.0;
                return;
            }
            a++;
        }
    }
    *models = 1;
    *sols = 1;
    nSol[id - *startID] = weight;
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
 * @param isSAT
 *      set to 1 if models are found
 */
__kernel void solveIntroduce(__global stype *nSol, __global stype *eSol,
                             __global long *clauses, __global unsigned long *cLen,
                             __global unsigned long *nVars, __global unsigned long *eVars,
                             __global unsigned long *numNV, __global unsigned long *numEV,
                             __global unsigned long *nClauses, __global unsigned long *eClauses,
                             __global unsigned long *numNC, __global unsigned long *numEC,
                             __global unsigned long *startIDn, __global unsigned long *startIDe,
                             __global unsigned long *minIDe, __global unsigned long *maxIDe,
                             __global unsigned long *isSAT, __global double *weights, __global int *sols) {
    unsigned long id = get_global_id(0);
    unsigned long assignment = id >> *numNC, templateID = 0;
    unsigned long a = 0, b = 0, c = 0, i = 0, notSAT = 0, base = 0;
    //check clauses
    for (a = 0, b = 0, i = 0; a < *numNC; i++) {
        if (i == 0 || clauses[i] == 0) {
            if (clauses[i] == 0) i++;
            if (nClauses[a] == eClauses[b]) {
                b++;
            } else if (isNotSat(assignment, &clauses[i], nVars) == ((id >> a) & 1)) {
                nSol[id - *startIDn] = 0.0;
                return;
            }
            a++;
        }
    }
    unsigned long d = 0;
    int baseSum = 0;
    //check variables
    for (i = 0, c = 0; i < *cLen; i++) {
        if (clauses[i] == 0) {
            baseSum = 0;
            if (nClauses[c] == eClauses[d]) {
                d++;
            }
            c++;
        } else {
            for (a = 0; a < *numNV; a++) {
                if ((((id >> c) & 1) == 0) && (clauses[i] == nVars[a] * (((assignment >> a) & 1) > 0 ? 1 : -1))) {
                    nSol[id - *startIDn] = 0.0;
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

    //template variables
    for (b = 0, a = 0; a < *numNV; a++) {
        if (nVars[a] == eVars[b]) {
            templateID |= ((id >> (a + *numNC)) & 1) << (b + *numEC);
            b++;
        }
    }

    //template clauses
    for (b = 0, a = 0; a < *numNC; a++) {
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
    if (*numNV != *numEV) {
        for (i = 0, c = 0; i < combinations; i++) {
            otherID = templateID;
            index = 0;

            for (ec = 0, nc = 0, x = 0; nc < *numNC; nc++, x++) {
                rec = 0;
                if (eClauses[ec] == nClauses[nc]) {
                    for (; clauses[x] != 0; x++) {
                        for (a = 0, b = 0; a < *numNV && rec == 0; a++) {
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

            if (otherID >= (*minIDe) && otherID < (*maxIDe)) {
                nSol[id - (*startIDn)] += eSol[otherID - (*startIDe)];
            }
        }
    } else {
        if (otherID >= (*minIDe) && otherID < (*maxIDe)) {
            nSol[id - (*startIDn)] += eSol[otherID - (*startIDe)];
        }
    }
    nSol[id - (*startIDn)] *= weight;
    if (nSol[id - (*startIDn)] > 0) {
        *isSAT = 1;
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
