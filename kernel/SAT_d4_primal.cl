#define stype double

typedef struct {
    stype x[4];
} d4_Type;

void d4_mul(d4_Type *a, __global d4_Type *b, __global d4_Type *ret);

void d4_mul_w(d4_Type *a, __global d4_Type *b, d4_Type *ret);

void d4_mul_w_(d4_Type *a, d4_Type *b, d4_Type *ret);

void d4_add(d4_Type *a, d4_Type *b, d4_Type *ret);

void d4_div(__global d4_Type *a, d4_Type *b, __global d4_Type *ret);

void d4_assign(d4_Type *a, d4_Type *b);

void new_d4(stype d, stype d1, stype d2, stype d3, d4_Type *ret);

void new_d4_(stype d, stype d1, stype d2, stype d3, __global d4_Type *ret);

void to_d4(stype x, d4_Type *ret);

void solveIntroduce_(long numV, __global d4_Type *edge, long numVE, __global long *variables, __global long *edgeVariables, d4_Type *ret, __global long *minId,
                     __global long *maxId, __global long *startIDEdge, __global d4_Type *weights) {
    long id = get_global_id(0);
    long otherId = 0;
    long a = 0, b = 0;
    d4_Type weight;
    to_d4(1.0, &weight);
    for (b = 0; b < numVE && a < numV; b++) {
        while ((variables[a] != edgeVariables[b])) {
            a++;
        }
        otherId = otherId | (((id >> a) & 1) << b);
        a++;
    };

    if (weights != 0) {
        for (b = 0, a = 0; a < numV; a++) {
            if ((variables[a] != edgeVariables[b])) {
                d4_mul_w(&weight, &weights[((id >> a) & 1) > 0 ? variables[a] * 2 : variables[a] * 2 + 1], &weight);
            }
            if (variables[a] == edgeVariables[b] && b < (numVE - 1)) {
                b++;
            }
        }
    }

    if (otherId >= (*minId) && otherId < (*maxId)) {
        ret->x[0] = edge[otherId - (*startIDEdge)].x[0];
        ret->x[1] = edge[otherId - (*startIDEdge)].x[1];
        ret->x[2] = edge[otherId - (*startIDEdge)].x[2];
        ret->x[3] = edge[otherId - (*startIDEdge)].x[3];
        d4_mul_w_(&weight, ret, ret);
    } else {
        ret->x[0] = -1.0;
        ret->x[1] = 0.0;
        ret->x[2] = 0.0;
        ret->x[3] = 0.0;
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
    for (i = 0; i < numclauses; i++) {
        satC = 0;
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
__kernel void solveJoin(__global d4_Type *solutions, __global d4_Type *edge1, __global d4_Type *edge2, __global long *variables, __global long *edgeVariables1,
                        __global long *edgeVariables2, long numV, long numVE1, long numVE2, __global long *minId1, __global long *maxId1, __global long *minId2,
                        __global long *maxId2, __global long *startIDNode, __global long *startIDEdge1, __global long *startIDEdge2, __global
                        d4_Type *weights) {
    long id = get_global_id(0);
    d4_Type tmp, tmp_;
    d4_Type weight;
    to_d4(1.0, &weight);
    solveIntroduce_(numV, edge1, numVE1, variables, edgeVariables1, &tmp, minId1, maxId1, startIDEdge1, weights);
    solveIntroduce_(numV, edge2, numVE2, variables, edgeVariables2, &tmp_, minId2, maxId2, startIDEdge2, weights);
    if (weights != 0) {
        for (int a = 0; a < numV; a++) {
            d4_mul_w(&weight, &weights[((id >> a) & 1) > 0 ? variables[a] * 2 : variables[a] * 2 + 1], &weight);
        }
    }
    if (tmp.x[0] >= 0.0) {
        d4_mul(&tmp, &solutions[id - (*startIDNode)], &solutions[id - (*startIDNode)]);
        d4_div(&solutions[id - (*startIDNode)], &weight, &solutions[id - (*startIDNode)]);
    }

    if (tmp_.x[0] >= 0.0) {
        d4_mul(&tmp_, &solutions[id - (*startIDNode)], &solutions[id - (*startIDNode)]);
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
 */
__kernel void solveForget(__global d4_Type *solutions, __global long *variablesCurrent, __global d4_Type *edge, long numVarsEdge, __global long *variablesEdge,
                          long combinations, long numVarsCurrent, __global long *minId, __global long *maxId, __global long *startIDNode, __global
                          long *startIDEdge) {
    long id = get_global_id(0), i = 0, a = 0, templateId = 0, test = 0;
    for (i = 0; i < numVarsEdge && a < numVarsCurrent; i++) {
        if (variablesEdge[i] == variablesCurrent[a]) {
            templateId = templateId | (((id >> a) & 1) << i);
            a++;
        }
    }
    d4_Type tmp, tmp_;
    for (i = 0; i < combinations; i++) {
        long b = 0, otherId = templateId;
        for (a = 0; a < numVarsEdge; a++) {
            if (b >= numVarsCurrent || variablesEdge[a] != variablesCurrent[b]) {
                otherId = otherId | (((i >> (a - b)) & 1) << a);
            } else {
                b++;
            }
        }
        if (otherId >= (*minId) && otherId < (*maxId)) {
            tmp.x[0] = solutions[id - (*startIDNode)].x[0];
            tmp.x[1] = solutions[id - (*startIDNode)].x[1];
            tmp.x[2] = solutions[id - (*startIDNode)].x[2];
            tmp.x[3] = solutions[id - (*startIDNode)].x[3];

            tmp_.x[0] = edge[otherId - (*startIDEdge)].x[0];
            tmp_.x[1] = edge[otherId - (*startIDEdge)].x[1];
            tmp_.x[2] = edge[otherId - (*startIDEdge)].x[2];
            tmp_.x[3] = edge[otherId - (*startIDEdge)].x[3];

            d4_add(&tmp, &tmp_, &tmp);

            solutions[id - (*startIDNode)].x[0] = tmp.x[0];
            solutions[id - (*startIDNode)].x[1] = tmp.x[1];
            solutions[id - (*startIDNode)].x[2] = tmp.x[2];
            solutions[id - (*startIDNode)].x[3] = tmp.x[3];
        }
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
__kernel void solveLeaf(__global long *clauses, __global long *numVarsC, long numclauses, __global d4_Type *solutions, long numV, __global long *variables,
                        __global long *models, __global long *startID, __global d4_Type *weights) {
    long id = get_global_id(0);
    int sat = checkBag(clauses, numVarsC, numclauses, id, numV, variables);
    d4_Type weight;
    to_d4(1.0, &weight);
    if (weights != 0) {
        for (int i = 0; i < numV; i++) {
            d4_mul_w(&weight, &weights[((id >> i) & 1) > 0 ? variables[i] * 2 : variables[i] * 2 + 1], &weight);
        }
    }
    if (sat == 1) {
        (*models) = 1;
        solutions[id - (*startID)].x[0] = weight.x[0];
        solutions[id - (*startID)].x[1] = weight.x[1];
        solutions[id - (*startID)].x[2] = weight.x[2];
        solutions[id - (*startID)].x[3] = weight.x[3];
    } else {
        solutions[id - (*startID)].x[0] = 0.0;
        solutions[id - (*startID)].x[1] = 0.0;
        solutions[id - (*startID)].x[2] = 0.0;
        solutions[id - (*startID)].x[3] = 0.0;
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
__kernel void solveIntroduce(__global long *clauses, __global long *numVarsC, long numclauses, __global d4_Type *solutions, long numV, __global d4_Type *edge,
                             long numVE, __global long *variables, __global long *edgeVariables, __global long *models, __global long *minId, __global
                             long *maxId,
                             __global long *startIDNode, __global long *startIDEdge, __global d4_Type *weights) {
    long id = get_global_id(0);
    d4_Type tmp;
    solveIntroduce_(numV, edge, numVE, variables, edgeVariables, &tmp, minId, maxId, startIDEdge, weights);
    if (tmp.x[0] > 0.0) {
        int sat = checkBag(clauses, numVarsC, numclauses, id, numV, variables);
        if (sat != 1) {
            solutions[id - (*startIDNode)].x[0] = 0.0;
            solutions[id - (*startIDNode)].x[1] = 0.0;
            solutions[id - (*startIDNode)].x[2] = 0.0;
            solutions[id - (*startIDNode)].x[3] = 0.0;
        } else {
            (*models) = 1;
            solutions[id - (*startIDNode)].x[0] = tmp.x[0];
            solutions[id - (*startIDNode)].x[1] = tmp.x[1];
            solutions[id - (*startIDNode)].x[2] = tmp.x[2];
            solutions[id - (*startIDNode)].x[3] = tmp.x[3];
        }
    } else if (tmp.x[0] == 0.0) {
        solutions[id - (*startIDNode)].x[0] = 0.0;
        solutions[id - (*startIDNode)].x[1] = 0.0;
        solutions[id - (*startIDNode)].x[2] = 0.0;
        solutions[id - (*startIDNode)].x[3] = 0.0;
    }
}


/**
 * adaptation of https://github.com/scibuilder/QD for opencl
 */

///headers
#define _QD_SPLITTER 134217729.0 // = 2^27 + 1
#define _QD_SPLIT_THRESH 6.69692879491417e+299 // = 2^996

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
            (a->x[0] == b->x[0] && (a->x[1] < b->x[1] ||
                                    (a->x[1] == b->x[1] && (a->x[2] < b->x[2] ||
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

void new_d4_(stype d, stype d1, stype d2, stype d3, __global d4_Type *ret) {
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

void d4_add(d4_Type *a, d4_Type *b, d4_Type *ret) {
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

void d4_mul(d4_Type *a, __global d4_Type *b, __global d4_Type *ret) {
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

void d4_mul_w(d4_Type *a, __global d4_Type *b, d4_Type *ret) {
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

void d4_mul_w_(d4_Type *a, d4_Type *b, d4_Type *ret) {
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
    d4_add(a, &c, &d);
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

void d4_div(__global d4_Type *a, d4_Type *b, __global d4_Type *ret) {
    stype q0, q1, q2, q3;

    d4_Type r, tmp, tmp_a;

    q0 = a->x[0] / b->x[0];
    tmp_a.x[0] = a->x[0];
    tmp_a.x[1] = a->x[1];
    tmp_a.x[2] = a->x[2];
    tmp_a.x[3] = a->x[3];
    d4_mul_qd_d(b, q0, &tmp);
    d4_minus(&tmp_a, &tmp, &r);

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

    new_d4_(q0, q1, q2, q3, ret);
}

void d4_assign(d4_Type *a, d4_Type *b) {
    a->x[0] = b->x[0];
    a->x[1] = b->x[1];
    a->x[2] = b->x[2];
    a->x[3] = b->x[3];
}
