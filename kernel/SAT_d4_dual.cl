R"(
#if defined(cl_khr_fp64)
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
#  pragma OPENCL EXTENSION cl_amd_fp64: enable
#else
#  error double precision is not supported
#endif
#define stype double

typedef struct {
    stype x[4];
} d4_Type;

void d4_assign_(__global d4_Type *a, d4_Type *b);

void d4_mul(d4_Type *a, __global d4_Type *b, __global d4_Type *ret);

void d4_mul_w(d4_Type *a, __global d4_Type *b, d4_Type *ret);

void d4_mul_w_(d4_Type *a, d4_Type *b, d4_Type *ret);

void d4_add(d4_Type *a, d4_Type *b, d4_Type *ret);

void d4_add_g(d4_Type *a, __global d4_Type *b, __global d4_Type *ret);

void d4_div(__global d4_Type *a, d4_Type *b, __global d4_Type *ret);

void d4_div_(d4_Type *a, d4_Type *b, d4_Type *ret);

void d4_assign(d4_Type *a, d4_Type *b);

void new_d4(stype d, stype d1, stype d2, stype d3, d4_Type *ret);

void new_d4_(stype d, stype d1, stype d2, stype d3, __global d4_Type *ret);

void to_d4(stype x, d4_Type *ret);

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


d4_Type checkFalsifiable(cFVars params,
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
                    }
                    if (c == -v) {
                        // variable is contained negative in a clause
                        negative = 1;
                    }
                }
            }
            varNum += numVarsC[i];
        }
        if (positive == 1 && negative == 1) {
            d4_Type ret;
            new_d4(0.0, 0.0, 0.0, 0.0, &ret);
            return ret;
        }
    }
    d4_Type ret;
    new_d4(1.0, 0.0, 0.0, 0.0, &ret);
    return ret;
}

d4_Type solveIntroduce_(sIVars params,
                       __global long *clauseIds, __global long *clauseIdsE,
                       __global long *numVarsC, __global long *numVarsCE,
                       __global d4_Type *solsE,
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
        d4_Type tmp = solsE[otherId - (params.startIDEdge)];
        for (int b = 0; a < params.numC; a++) {
            while (b < params.numCE && a < params.numC && clauseIds[a] == clauseIdsE[b]) {
                b++;
                a++;
            }
            if (a < params.numC && (b >= params.numCE || clauseIds[a] != clauseIdsE[b])) {
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
                            }
                        }
                        long startClauses = 0;
                        for (int k = 0; k < a; ++k) {
                            for (int i = 0; i < numVarsC[k]; ++i) {
                                if (var == abs(clauseVars[i + startClauses])) {
                                    found = 1;
                                }
                            }
                            startClauses += numVarsC[k];
                        }
                        //var(A\{C})
                        long startIDCE = 0;
                        for (int i = 0; i < params.numCE; ++i) {
                            if ((otherId >> i) & 1) {
                                for (int l = 0; l < numVarsCE[i]; ++l) {
                                    if (var == abs(clauseVarsE[l + startIDCE])) {
                                        subfound = 1;
                                    }
                                }
                                if (subfound == 1) {
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
                                    }
                                }
                            }
                            startClauses += numVarsC[k];
                        }

                        if (found == 1 && subfound == 0) {
                            ++exponent;
                        }
                    }
                    d4_Type ret;
                    new_d4( 1 << exponent, 0.0, 0.0, 0.0, &ret);
                    d4_div_(&tmp,&ret,&tmp);
                } else {
                    //|var(C)\var(x(t'))|
                    long startIDC = 0;
                    for (int i = 0; i < a; ++i) {
                        startIDC += numVarsC[i];
                    }
                    long exponent = 0;
                    for (int j = 0; j < numVarsC[a]; ++j) {
                        long found = 0;
                        long varNum = 0;
                        long var = abs(clauseVars[j + startIDC]);
                        //var(x(t'))
                        for (int l = 0; l < params.numVE; ++l) {
                            if (var == varsE[l]) {
                                found = 1;
                            }
                        }
                        long startClauses = 0;
                        for (int k = 0; k < a; ++k) {
                            for (int i = 0; i < numVarsC[k]; ++i) {
                                if (var == abs(clauseVars[i + startClauses])) {
                                    found = 1;
                                }
                            }
                            startClauses += numVarsC[k];
                        }
                        if (found == 0) {
                            ++exponent;
                        }
                    }
                    d4_Type ret;
                    new_d4( 1 << exponent, 0.0, 0.0, 0.0, &ret);
                    d4_mul_w_(&ret,&tmp,&tmp);
                }
            }
        };
        if (tmp.x[0] > 0.0) {
            cFVars cFparams;
            cFparams.clauseId = params.id;
            cFparams.numC = params.numC;
            cFparams.numV = params.numV;
            d4_Type ret=checkFalsifiable(cFparams, clauseVars, numVarsC, vars);
            d4_mul_w_(&ret,&tmp,&tmp);
        }
        return tmp;
    } else if (solsE == 0 && otherId >= (params.minId) && otherId < (params.maxId)) {
        d4_Type ret;
        new_d4(0.0, 0.0, 0.0, 0.0, &ret);
        return ret;
    } else {
        d4_Type ret;
        new_d4(-1.0, 0.0, 0.0, 0.0, &ret);
        return ret;
    }
}


__kernel void solveJoin(sJVars params,
                        __global d4_Type *sol, __global d4_Type *solE1, __global d4_Type *solE2,
                        __global int *sols,
                        __global long *clauseVars,
                        __global long *clauseIds, __global long *clauseIdsE1, __global long *clauseIdsE2,
                        __global long *numVarsC,
                        __global long *variables, __global long *varsE1, __global long *varsE2,
                        __global long *numVarsCE1, __global long *numVarsCE2,
                        __global long *clauseVarsE1, __global long *clauseVarsE2) {
    long id = get_global_id(0);
    d4_Type tmp, tmp_;
    new_d4(-1.0, 0.0, 0.0, 0.0, &tmp);
    new_d4(-1.0, 0.0, 0.0, 0.0, &tmp_);
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
    tmp_ = solveIntroduce_(iparams2, clauseIds, clauseIdsE2, numVarsC, numVarsCE2, solE2, variables, varsE2, clauseVars, clauseVarsE2);
    // we have some solutions in edge1

    if (tmp.x[0] >= 0.0) {
        // |var(x(t))\var(A)|
        long exponent = 0;
        for (int j = 0; j < params.numV; ++j) {
            int found = 0;
            long varNum = 0;
            for (int i = 0; i < params.numC; i++) {
                if (((id >> i) & 1) == 1) {
                    for (int a = 0; a < numVarsC[i]; a++) {
                        if (variables[j] == abs(clauseVars[varNum + a])) {
                            found = 1;
                        }
                    }
                }
                varNum += numVarsC[i];
            }
            if (found == 0) {
                exponent++;
            }

        }
        d4_Type ret;
        new_d4( 1 << exponent, 0.0, 0.0, 0.0, &ret);
        d4_div_(&tmp,&ret,&tmp);
        d4_mul(&tmp,&sol[id - (params.startIDNode)] ,&sol[id - (params.startIDNode)] );

    }

    // we have some solutions in edge2
    if (tmp_.x[0] >= 0.0) {
        d4_mul(&tmp_,&sol[id - (params.startIDNode)] ,&sol[id - (params.startIDNode)] );
    }
    if (sol[id - (params.startIDNode)].x[0] > 0) {
        *sols = 1;
    }
}

d4_Type solveIntroduceF(sIVars params,
                      __global long *clauseIds, __global long *clauseIdsE,
                      __global long *numVarsC, __global long *numVarsCE,
                      __global d4_Type *edge,
                      __global long *vars, __global long *varsE,
                      __global long *clauseVars, __global long *clauseVarsE) {
    d4_Type tmp;
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
            for (int i = 0; i < params.numC; i++) {
                if (((params.id >> i) & 1) == 1) {
                    for (int a = 0; a < numVarsC[i]; a++) {
                        if (vars[j] == abs(clauseVars[varNum + a])) {
                            found = 1;
                        }
                    }
                }
                varNum += numVarsC[i];
            }
            if (found == 0) {
                exponent++;
            }

        }
        new_d4(1 << exponent,0.0,0.0,0.0,&tmp);
        cFVars cFparams;
        cFparams.clauseId = params.id;
        cFparams.numC = params.numC;
        cFparams.numV = params.numV;
        d4_Type tmp2 =checkFalsifiable(cFparams, clauseVars, numVarsC, vars);
        d4_mul_w_(&tmp,&tmp2,&tmp);
    }
    return tmp;
}

__kernel void
solveIntroduceForget(sIFVars params,
                     __global d4_Type *solsF, __global d4_Type *solsE,
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
            d4_Type tmp = solveIntroduceF(iparams, clauseIdsI, clauseIdsE, numVarsCI, numVarsCE, solsE, varsI, varsE, clauseVarsI, clauseVarsE);
            if (tmp.x[0] > 0) {
                d4_Type ret;
                new_d4((1 - ((popcount(i) % 2 == 1) * 2)), 0.0, 0.0, 0.0, &ret);
                d4_mul_w_(&tmp,&ret,&tmp);
                d4_add_g(&tmp, &solsF[id - (params.startIDF)], &solsF[id - (params.startIDF)]);
            }
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
        d4_Type tmp = solveIntroduceF(iparams, clauseIdsI, clauseIdsE, numVarsCI, numVarsCE, solsE, varsI, varsE, clauseVarsI, clauseVarsE);
        if (tmp.x[0] > 0) {
            d4_add_g(&tmp, &solsF[id - (params.startIDF)], &solsF[id - (params.startIDF)]);
        }
    }
    if (solsF[id - (params.startIDF)].x[0] > 0) {
        *sols = 1;
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

void d4_add_g(d4_Type *a, __global d4_Type *b, __global d4_Type *ret) {
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

void d4_div_(d4_Type *a, d4_Type *b, d4_Type *ret) {
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
)"