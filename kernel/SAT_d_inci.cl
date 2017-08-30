//#define __kernel
//#define __global

#define stype double

int isNotSat(unsigned long assignment, __global long *clause, __global unsigned long *variables);

/**
 * Operation to solve a Join node in the decomposition.
 *
 * @param nSol
 * @param e1Sol
 * @param e2Sol
 * @param nVar
 * @param e1Var
 * @param e2Var
 * @param minIDe1
 * @param minIDe2
 * @param maxIDe1
 * @param maxIDe2
 * @param startIDn
 * @param startIDe1
 * @param startIDe2
 */
__kernel void
solveJoin(__global stype *nSol, __global stype *e1Sol, __global stype *e2Sol,
          __global unsigned long *minIDe1, __global unsigned long *maxIDe1,
          __global unsigned long *minIDe2, __global unsigned long *maxIDe2,
          __global unsigned long *startIDn, __global unsigned long *startIDe1, __global unsigned long *startIDe2,
          __global unsigned long *numClauses) {
    unsigned long id = get_global_id(0);
    unsigned long mask = id & (((unsigned long) exp2((double) *numClauses)) - 1);
    unsigned long combinations = ((unsigned long) exp2((double) *numClauses));
    unsigned long templateID = id >> *numClauses << *numClauses;
    //sum up all subsets of Clauses (A1,A2) where the intersection of A1 and A2 = A
    unsigned long start2 = 0, end2 = combinations - 1;
    for (; start2 < combinations && e2Sol[(templateID | start2) - (*startIDe2)] == 0; start2++);
    for (; end2 > 0 && e2Sol[(templateID | end2) - (*startIDe2)] == 0; end2--);
    for (int a = 0; a < combinations; a++) {
        if ((templateID | a) >= *minIDe1 && (templateID | a) < *maxIDe1 && e1Sol[(templateID | a) - (*startIDe1)] != 0) {
            for (int b = start2; b <= end2; b++) {
                if (((a | b)) == mask && ((templateID | b) >= *minIDe2 && (templateID | b) < *maxIDe2) && e2Sol[(templateID | b) - (*startIDe2)] != 0) {
                    nSol[id - (*startIDn)] += e1Sol[(templateID | a) - (*startIDe1)] * e2Sol[(templateID | b) - (*startIDe2)];
                }
            }
        }
    }
}

/**
 * Operation to solve a Forget node in the decomposition.
 *
 * @param nVars
 * @param eVars
 * @param nSol
 * @param eSol
 * @param minID
 * @param maxID
 * @param startIDn
 * @param startIDe
 */
__kernel void
solveForget(__global stype *nSol, __global stype *eSol,
            __global unsigned long *nVars, __global unsigned long *eVars,
            __global unsigned long *numNVars, __global unsigned long *numEVars,
            __global unsigned long *nClauses, __global unsigned long *eClauses,
            __global unsigned long *numNC, __global unsigned long *numEC,
            __global unsigned long *startIDn, __global unsigned long *startIDe,
            __global unsigned long *minID, __global unsigned long *maxID) {
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
}

/**
 * Operation to solve a Leaf node in the decomposition.
 *
 * @param clauses
 * @param variables
 * @param solutions
 * @param startID
 * @param models
 */
__kernel void
solveLeaf(__global stype *nSol,
          __global long *clauses,
          __global unsigned long *nVars,
          __global unsigned long *numNC,
          __global unsigned long *startID,
          __global unsigned long *models) {
    unsigned long id = get_global_id(0);
    unsigned long assignment = id >> *numNC;
    unsigned long i = 0, a = 0;
    for (i = 0; a < *numNC; i++) {
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
    nSol[id - *startID] = 1.0;
}

/**
 * Operation to solve a Introduce node in the decomposition.
 *
 * @param clauses
 * @param nVars
 * @param eVars
 * @param nSol
 * @param eSol
 * @param models
 * @param minID
 * @param maxID
 * @param startIDn
 * @param startIDe
 */
__kernel void
solveIntroduce(__global stype *nSol, __global stype *eSol,
               __global long *clauses, __global unsigned long *cLen,
               __global unsigned long *nVars, __global unsigned long *eVars,
               __global unsigned long *numNV, __global unsigned long *numEV,
               __global unsigned long *nClauses, __global unsigned long *eClauses,
               __global unsigned long *numNC, __global unsigned long *numEC,
               __global unsigned long *startIDn, __global unsigned long *startIDe,
               __global unsigned long *minIDe, __global unsigned long *maxIDe,
               __global unsigned long *isSAT) {
    unsigned long id = get_global_id(0);
    unsigned long assignment = id >> *numNC, templateID = 0;
    unsigned long a = 0, b = 0, c = 0, i = 0, notSAT = 0, base = 0;
    //if (id == 3) printf("start");

    //check clauses
    for (a = 0, b = 0, i = 0; a < *numNC; i++) {
        if (i == 0 || clauses[i] == 0) {
            if (clauses[i] == 0) i++;
            if (nClauses[a] == eClauses[b]) {
                b++;
            } else if (isNotSat(assignment, &clauses[i], nVars) == ((id >> a) & 1)) {
                nSol[id - *startIDn] = 0.0;
                //if (id == 3) printf("check Clause, n: %i, e: %i, a: %i, b:%i",nClauses[a],eClauses[b],a,b);
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
                //if (id == 6) printf("__id: %i, clause: %i, var: %i ", id, clauses[i],nVars[a]);
                if ((((id >> c) & 1) == 0) && (clauses[i] == nVars[a] * (((assignment >> a) & 1) > 0 ? 1 : -1))) {
                    nSol[id - *startIDn] = 0.0;
                    //if (id == 3) printf("check variable");
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

    if (*numNV != *numEV) {
        for (i = 0, c = 0; i < combinations; i++) {
            otherID = templateID;
            index = 0;

            for (ec = 0, nc = 0, x = 0; nc < *numNC; nc++, x++) {
                rec = 0;
                if (eClauses[ec] == nClauses[nc]) {
                    for (; clauses[x] != 0; x++) {
                        for (a = 0, b = 0; a < *numNV && rec == 0; a++) {
                            if (clauses[x] == (nVars[a] * (((assignment >> a) & 1) > 0 ? 1 : -1)))  {
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

    if (nSol[id - (*startIDn)] > 0) {
        *isSAT = 1;
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
