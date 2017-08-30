//#define __kernel
//#define __global

#define stype double

typedef struct {
    stype x[4];
} d4_Type;

void d4_mul(__global d4_Type *a, __global d4_Type *b, d4_Type *ret);

void d4_add(__global d4_Type *a, __global d4_Type *b, __global d4_Type *ret);

void d4_add2(__global d4_Type *a, d4_Type *b, __global d4_Type *ret);

void d4_div(d4_Type *a, d4_Type *b, d4_Type *ret);

void d4_assign(d4_Type *a, d4_Type *b);
void d4_assign_(__global d4_Type *a, d4_Type *b);

void new_d4(stype d, stype d1, stype d2, stype d3, d4_Type *ret);

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
solveJoin(__global d4_Type *nSol, __global d4_Type *e1Sol, __global d4_Type *e2Sol,
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
    for (; start2 < combinations && e2Sol[(templateID | start2) - (*startIDe2)].x[0] == 0; start2++);
    for (; end2 > 0 && e2Sol[(templateID | end2) - (*startIDe2)].x[0] == 0; end2--);
    for (int a = 0; a < combinations; a++) {
        if ((templateID | a) >= *minIDe1 && (templateID | a) < *maxIDe1 && e1Sol[(templateID | a) - (*startIDe1)].x[0] != 0) {
            for (int b = start2; b <= end2; b++) {
                if (((a | b)) == mask && ((templateID | b) >= *minIDe2 && (templateID | b) < *maxIDe2) && e2Sol[(templateID | b) - (*startIDe2)].x[0] != 0) {
                    d4_Type tmp;
                    d4_mul(&e1Sol[(templateID | a) - (*startIDe1)], &e2Sol[(templateID | b) - (*startIDe2)], &tmp);
                    d4_add2(&nSol[id - (*startIDn)], &tmp, &nSol[id - (*startIDn)]);
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
solveForget(__global d4_Type *nSol, __global d4_Type *eSol,
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

    d4_Type tmp, tmp_;
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
            d4_add(&nSol[id - (*startIDn)], &eSol[otherId - (*startIDe)], &nSol[id - (*startIDn)]);
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
solveLeaf(__global d4_Type *nSol,
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
                nSol[id - *startID].x[0] = 0.0;
                nSol[id - *startID].x[1] = 0.0;
                nSol[id - *startID].x[2] = 0.0;
                nSol[id - *startID].x[3] = 0.0;
                return;
            }
            a++;
        }
    }
    *models = 1;
    nSol[id - *startID].x[0] = 1.0;
    nSol[id - *startID].x[1] = 0.0;
    nSol[id - *startID].x[2] = 0.0;
    nSol[id - *startID].x[3] = 0.0;
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
solveIntroduce(__global d4_Type *nSol, __global d4_Type *eSol,
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

    //check clauses
    for (a = 0, b = 0, i = 0; a < *numNC; i++) {
        if (i == 0 || clauses[i] == 0) {
            if (clauses[i] == 0) i++;
            if (nClauses[a] == eClauses[b]) {
                b++;
            } else if (isNotSat(assignment, &clauses[i], nVars) == ((id >> a) & 1)) {
                nSol[id - *startIDn].x[0] = 0.0;
                nSol[id - *startIDn].x[1] = 0.0;
                nSol[id - *startIDn].x[2] = 0.0;
                nSol[id - *startIDn].x[3] = 0.0;
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
                    nSol[id - *startIDn].x[0] = 0.0;
                    nSol[id - *startIDn].x[1] = 0.0;
                    nSol[id - *startIDn].x[2] = 0.0;
                    nSol[id - *startIDn].x[3] = 0.0;
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
                d4_add(&nSol[id - (*startIDn)], &eSol[otherID - (*startIDe)], &nSol[id - (*startIDn)]);
            }
        }
    } else {
        if (otherID >= (*minIDe) && otherID < (*maxIDe)) {
            d4_add(&nSol[id - (*startIDn)], &eSol[otherID - (*startIDe)], &nSol[id - (*startIDn)]);
        }
    }

    if (nSol[id - (*startIDn)].x[0] > 0) {
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


/**
 * adaptation of https://github.com/scibuilder/QD for opencl
 */

///headers
#define _QD_SPLITTER 134217729.0 // = 2^27 + 1
#define _QD_SPLIT_THRESH 6.69692879491417e+299 // = 2^996

void to_d4(stype x, d4_Type *ret);

void d4_neg(d4_Type *x, d4_Type *ret);

void d4_minus(d4_Type *a, d4_Type *b, d4_Type *ret);

void d4_log(d4_Type *a, d4_Type *ret);

stype d4_quick_two_sum(stype a, stype b, stype *err);

stype d4_two_sum(stype a, stype b, stype *err);

stype d4_quick_three_accum(stype *a, stype *b, stype c);

void d4_renorm(stype *c0, stype *c1, stype *c2, stype *c3);

void d4_split(stype a, stype *hi, stype *lo);

stype d4_two_prod(stype a, stype b, stype *err);

void d4_three_sum(stype *a, stype *b, stype *c);

void d4_renorm_(stype *c0, stype *c1, stype *c2, stype *c3, stype *c4);

void d4_three_sum2(stype *a, stype *b, stype *c);;

void d4_mul_qd_d(d4_Type *a, stype b, d4_Type *ret);

void d4_ldexp(d4_Type *a, int n, d4_Type *ret);

bool d4_is_zero(d4_Type *x);

bool d4_is_one(d4_Type *x);

void d4_mul_pwr2(d4_Type *a, stype b, d4_Type *ret);

void d4_sqr(d4_Type *a, d4_Type *ret);

void d4_pow(d4_Type *a, int n, d4_Type *ret);

bool d4_l(d4_Type *a, d4_Type *b) {
    return (a->x[0] < b->x[0] ||
            (a->x[0] == b->x[0] && (a->x[1] < b->x[1]
                                    || (a->x[1] == b->x[1] && (a->x[2] < b->x[2]
                                                               ||
                                                               (a->x[2] == b->x[2] && a->x[3] < b->x[3]))))));
}

///implementation
void to_d4(stype x, d4_Type *ret) {
    ret->x[0] = x;
    ret->x[1] = 0.0;
    ret->x[2] = 0.0;
    ret->x[3] = 0.0;
}

void d4_neg(d4_Type *x, d4_Type *ret) {
    ret->x[0] = -x->x[0];
    ret->x[1] = -x->x[1];
    ret->x[2] = -x->x[2];
    ret->x[3] = -x->x[3];
}

void new_d4(stype d, stype d1, stype d2, stype d3, d4_Type *ret) {
    ret->x[0] = d;
    ret->x[1] = d1;
    ret->x[2] = d2;
    ret->x[3] = d3;
}

stype d4_quick_two_sum(stype a, stype b, stype *err) {
    stype s = a + b;
    (*err) = b - (s - a);
    return s;
}

stype d4_two_sum(stype a, stype b, stype *err) {
    stype s = a + b;
    stype bb = s - a;
    (*err) = (a - (s - bb)) + (b - bb);
    return s;
}

stype d4_quick_three_accum(stype *a, stype *b, stype c) {
    stype s;
    bool za, zb;

    s = d4_two_sum((*b), c, b);
    s = d4_two_sum((*a), s, a);

    za = ((*a) != 0.0);
    zb = ((*b) != 0.0);

    if (za && zb)
        return s;

    if (!zb) {
        (*b) = (*a);
        (*a) = s;
    } else {
        (*a) = s;
    }

    return 0.0;
}

void d4_renorm(stype *c0, stype *c1, stype *c2, stype *c3) {
    stype s0, s1, s2 = 0.0, s3 = 0.0;

    if (isinf((*c0))) return;

    s0 = d4_quick_two_sum((*c2), (*c3), c3);
    s0 = d4_quick_two_sum((*c1), s0, c2);
    (*c0) = d4_quick_two_sum((*c0), s0, c1);

    s0 = (*c0);
    s1 = (*c1);
    if (s1 != 0.0) {
        s1 = d4_quick_two_sum(s1, (*c2), &s2);
        if (s2 != 0.0)
            s2 = d4_quick_two_sum(s2, (*c3), &s3);
        else
            s1 = d4_quick_two_sum(s1, (*c3), &s2);
    } else {
        s0 = d4_quick_two_sum(s0, (*c2), &s1);
        if (s1 != 0.0)
            s1 = d4_quick_two_sum(s1, (*c3), &s2);
        else
            s0 = d4_quick_two_sum(s0, (*c3), &s1);
    }

    (*c0) = s0;
    (*c1) = s1;
    (*c2) = s2;
    (*c3) = s3;
}

void d4_add(__global d4_Type *a, __global d4_Type *b, __global d4_Type *ret) {
    int i, j, k;
    stype s, t;
    stype u, v;
    d4_Type x;
    to_d4(0.0, &x);

    i = j = k = 0;
    if (fabs(a->x[i]) > fabs(b->x[j]))
        u = a->x[i++];
    else
        u = b->x[j++];
    if (fabs(a->x[i]) > fabs(b->x[j]))
        v = a->x[i++];
    else
        v = b->x[j++];

    u = d4_quick_two_sum(u, v, &v);

    while (k < 4) {
        if (i >= 4 && j >= 4) {
            x.x[k] = u;
            if (k < 3)
                x.x[++k] = v;
            break;
        }

        if (i >= 4)
            t = b->x[j++];
        else if (j >= 4)
            t = a->x[i++];
        else if (fabs(a->x[i]) > fabs(b->x[j])) {
            t = a->x[i++];
        } else
            t = b->x[j++];

        s = d4_quick_three_accum(&u, &v, t);

        if (s != 0.0) {
            x.x[k++] = s;
        }
    }

    for (k = i; k < 4; k++)
        x.x[3] += a->x[k];
    for (k = j; k < 4; k++)
        x.x[3] += b->x[k];

    d4_renorm(&x.x[0], &x.x[1], &x.x[2], &x.x[3]);
    d4_assign_(ret, &x);
}

void d4_add2(__global d4_Type *a, d4_Type *b, __global d4_Type *ret) {
    int i, j, k;
    stype s, t;
    stype u, v;
    d4_Type x;
    to_d4(0.0, &x);

    i = j = k = 0;
    if (fabs(a->x[i]) > fabs(b->x[j]))
        u = a->x[i++];
    else
        u = b->x[j++];
    if (fabs(a->x[i]) > fabs(b->x[j]))
        v = a->x[i++];
    else
        v = b->x[j++];

    u = d4_quick_two_sum(u, v, &v);

    while (k < 4) {
        if (i >= 4 && j >= 4) {
            x.x[k] = u;
            if (k < 3)
                x.x[++k] = v;
            break;
        }

        if (i >= 4)
            t = b->x[j++];
        else if (j >= 4)
            t = a->x[i++];
        else if (fabs(a->x[i]) > fabs(b->x[j])) {
            t = a->x[i++];
        } else
            t = b->x[j++];

        s = d4_quick_three_accum(&u, &v, t);

        if (s != 0.0) {
            x.x[k++] = s;
        }
    }

    for (k = i; k < 4; k++)
        x.x[3] += a->x[k];
    for (k = j; k < 4; k++)
        x.x[3] += b->x[k];

    d4_renorm(&x.x[0], &x.x[1], &x.x[2], &x.x[3]);
    d4_assign_(ret, &x);
}

void d4_add_(d4_Type *a, d4_Type *b, d4_Type *ret) {
    int i, j, k;
    stype s, t;
    stype u, v;
    d4_Type x;
    to_d4(0.0, &x);

    i = j = k = 0;
    if (fabs(a->x[i]) > fabs(b->x[j]))
        u = a->x[i++];
    else
        u = b->x[j++];
    if (fabs(a->x[i]) > fabs(b->x[j]))
        v = a->x[i++];
    else
        v = b->x[j++];

    u = d4_quick_two_sum(u, v, &v);

    while (k < 4) {
        if (i >= 4 && j >= 4) {
            x.x[k] = u;
            if (k < 3)
                x.x[++k] = v;
            break;
        }

        if (i >= 4)
            t = b->x[j++];
        else if (j >= 4)
            t = a->x[i++];
        else if (fabs(a->x[i]) > fabs(b->x[j])) {
            t = a->x[i++];
        } else
            t = b->x[j++];

        s = d4_quick_three_accum(&u, &v, t);

        if (s != 0.0) {
            x.x[k++] = s;
        }
    }

    for (k = i; k < 4; k++)
        x.x[3] += a->x[k];
    for (k = j; k < 4; k++)
        x.x[3] += b->x[k];

    d4_renorm(&x.x[0], &x.x[1], &x.x[2], &x.x[3]);
    d4_assign(ret, &x);
}

void d4_split(stype a, stype *hi, stype *lo) {
    stype temp;
    if (a > _QD_SPLIT_THRESH || a < -_QD_SPLIT_THRESH) {
        a *= 3.7252902984619140625e-09; // 2^-28
        temp = _QD_SPLITTER * a;
        (*hi) = temp - (temp - a);
        (*lo) = a - (*hi);
        (*hi) *= 268435456.0;
        (*lo) *= 268435456.0;
    } else {
        temp = _QD_SPLITTER * a;
        (*hi) = temp - (temp - a);
        (*lo) = a - (*hi);
    }
}

stype d4_two_prod(stype a, stype b, stype *err) {
    stype a_hi, a_lo, b_hi, b_lo;
    stype p = a * b;
    d4_split(a, &a_hi, &a_lo);
    d4_split(b, &b_hi, &b_lo);
    (*err) = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    return p;
}

void d4_three_sum(stype *a, stype *b, stype *c) {
    stype t1, t2, t3;
    t1 = d4_two_sum((*a), (*b), &t2);
    (*a) = d4_two_sum((*c), t1, &t3);
    (*b) = d4_two_sum(t2, t3, c);
}

void d4_renorm_(stype *c0, stype *c1, stype *c2, stype *c3, stype *c4) {
    stype s0, s1, s2 = 0.0, s3 = 0.0;

    if (isinf((*c0))) return;

    s0 = d4_quick_two_sum((*c3), (*c4), c4);
    s0 = d4_quick_two_sum((*c2), s0, c3);
    s0 = d4_quick_two_sum((*c1), s0, c2);
    (*c0) = d4_quick_two_sum((*c0), s0, c1);

    s0 = (*c0);
    s1 = (*c1);

    s0 = d4_quick_two_sum((*c0), (*c1), &s1);
    if (s1 != 0.0) {
        s1 = d4_quick_two_sum(s1, (*c2), &s2);
        if (s2 != 0.0) {
            s2 = d4_quick_two_sum(s2, (*c3), &s3);
            if (s3 != 0.0)
                s3 += (*c4);
            else
                s2 += (*c4);
        } else {
            s1 = d4_quick_two_sum(s1, (*c3), &s2);
            if (s2 != 0.0)
                s2 = d4_quick_two_sum(s2, (*c4), &s3);
            else
                s1 = d4_quick_two_sum(s1, (*c4), &s2);
        }
    } else {
        s0 = d4_quick_two_sum(s0, (*c2), &s1);
        if (s1 != 0.0) {
            s1 = d4_quick_two_sum(s1, (*c3), &s2);
            if (s2 != 0.0)
                s2 = d4_quick_two_sum(s2, (*c4), &s3);
            else
                s1 = d4_quick_two_sum(s1, (*c4), &s2);
        } else {
            s0 = d4_quick_two_sum(s0, (*c3), &s1);
            if (s1 != 0.0)
                s1 = d4_quick_two_sum(s1, (*c4), &s2);
            else
                s0 = d4_quick_two_sum(s0, (*c4), &s1);
        }
    }

    (*c0) = s0;
    (*c1) = s1;
    (*c2) = s2;
    (*c3) = s3;
}

void d4_mul(__global d4_Type *a, __global d4_Type *b, d4_Type *ret) {
    stype p0, p1, p2, p3, p4, p5;
    stype q0, q1, q2, q3, q4, q5;
    stype p6, p7, p8, p9;
    stype q6, q7, q8, q9;
    stype r0, r1;
    stype t0, t1;
    stype s0, s1, s2;

    p0 = d4_two_prod(a->x[0], b->x[0], &q0);

    p1 = d4_two_prod(a->x[0], b->x[1], &q1);
    p2 = d4_two_prod(a->x[1], b->x[0], &q2);

    p3 = d4_two_prod(a->x[0], b->x[2], &q3);
    p4 = d4_two_prod(a->x[1], b->x[1], &q4);
    p5 = d4_two_prod(a->x[2], b->x[0], &q5);

    d4_three_sum(&p1, &p2, &q0);

    d4_three_sum(&p2, &q1, &q2);
    d4_three_sum(&p3, &p4, &p5);

    s0 = d4_two_sum(p2, p3, &t0);
    s1 = d4_two_sum(q1, p4, &t1);
    s2 = q2 + p5;
    s1 = d4_two_sum(s1, t0, &t0);
    s2 += (t0 + t1);

    p6 = d4_two_prod(a->x[0], b->x[3], &q6);
    p7 = d4_two_prod(a->x[1], b->x[2], &q7);
    p8 = d4_two_prod(a->x[2], b->x[1], &q8);
    p9 = d4_two_prod(a->x[3], b->x[0], &q9);

    q0 = d4_two_sum(q0, q3, &q3);
    q4 = d4_two_sum(q4, q5, &q5);
    p6 = d4_two_sum(p6, p7, &p7);
    p8 = d4_two_sum(p8, p9, &p9);

    t0 = d4_two_sum(q0, q4, &t1);
    t1 += (q3 + q5);

    r0 = d4_two_sum(p6, p8, &r1);
    r1 += (p7 + p9);

    q3 = d4_two_sum(t0, r0, &q4);
    q4 += (t1 + r1);

    t0 = d4_two_sum(q3, s1, &t1);
    t1 += q4;

    t1 += a->x[1] * b->x[3] + a->x[2] * b->x[2] + a->x[3] * b->x[1] + q6 + q7 + q8 + q9 + s2;

    d4_renorm_(&p0, &p1, &s0, &t0, &t1);
    ret->x[0] = p0;
    ret->x[1] = p1;
    ret->x[2] = s0;
    ret->x[3] = t0;
}


void d4_minus(d4_Type *a, d4_Type *b, d4_Type *ret) {
    d4_Type c;
    d4_neg(b, &c);
    d4_Type d;
    d4_add_(a, &c, &d);
    d4_assign(ret, &d);
}

void d4_three_sum2(stype *a, stype *b, stype *c) {
    stype t1, t2, t3;
    t1 = d4_two_sum((*a), (*b), &t2);
    (*a) = d4_two_sum((*c), t1, &t3);
    (*b) = t2 + t3;
}

void d4_mul_qd_d(d4_Type *a, stype b, d4_Type *ret) {
    stype p0, p1, p2, p3;
    stype q0, q1, q2;
    stype s0, s1, s2, s3, s4;

    p0 = d4_two_prod(a->x[0], b, &q0);
    p1 = d4_two_prod(a->x[1], b, &q1);
    p2 = d4_two_prod(a->x[2], b, &q2);
    p3 = a->x[3] * b;

    s0 = p0;

    s1 = d4_two_sum(q0, p1, &s2);

    d4_three_sum(&s2, &q1, &p2);

    d4_three_sum2(&q1, &q2, &p3);
    s3 = q1;

    s4 = q2 + p2;

    d4_renorm_(&s0, &s1, &s2, &s3, &s4);
    new_d4(s0, s1, s2, s3, ret);
}

void d4_div(d4_Type *a, d4_Type *b, d4_Type *ret) {
    stype q0, q1, q2, q3;

    d4_Type r, tmp;

    q0 = a->x[0] / b->x[0];
    d4_mul_qd_d(b, q0, &tmp);
    d4_minus(a, &tmp, &r);

    q1 = r.x[0] / b->x[0];
    d4_mul_qd_d(b, q1, &tmp);
    d4_minus(&r, &tmp, &r);

    q2 = r.x[0] / b->x[0];
    d4_mul_qd_d(b, q2, &tmp);
    d4_minus(&r, &tmp, &r);

    q3 = r.x[0] / b->x[0];
    d4_mul_qd_d(b, q3, &tmp);
    d4_minus(&r, &tmp, &r);

    stype q4 = r.x[0] / b->x[0];

    d4_renorm_(&q0, &q1, &q2, &q3, &q4);

    new_d4(q0, q1, q2, q3, ret);
}

void d4_assign(d4_Type *a, d4_Type *b) {
    a->x[0] = b->x[0];
    a->x[1] = b->x[1];
    a->x[2] = b->x[2];
    a->x[3] = b->x[3];
}

void d4_assign_(__global d4_Type *a, d4_Type *b) {
    a->x[0] = b->x[0];
    a->x[1] = b->x[1];
    a->x[2] = b->x[2];
    a->x[3] = b->x[3];
}
