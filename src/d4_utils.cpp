/**
 * adaptation of https://github.com/scibuilder/QD for opencl
 */

#include <d4_utils.h>
#include <math.h>

namespace gpusat {

    solType *new_d4(solType *x) {
        solType *ret = new solType;
        ret->x[0] = x->x[0];
        ret->x[1] = x->x[1];
        ret->x[2] = x->x[2];
        ret->x[3] = x->x[3];
        return ret;
    }

    solType *to_d4(double x) {
        solType *ret = new solType;
        ret->x[0] = x;
        ret->x[1] = 0.0;
        ret->x[2] = 0.0;
        ret->x[3] = 0.0;
        return ret;
    }

    solType *d4_neg(solType *x) {
        solType *ret = new solType;
        ret->x[0] = -x->x[0];
        ret->x[1] = -x->x[1];
        ret->x[2] = -x->x[2];
        ret->x[3] = -x->x[3];
        return ret;
    }

    solType *d4_log(solType *a) {

        if (a->x[0] == 1.0 && a->x[1] == 0.0 && a->x[2] == 0.0 && a->x[3] == 0.0) {
            return to_d4(0.0);
        }

        if (a->x[0] <= 0.0) {
            return new_d4(&d4_inf);
        }

        if (a->x[0] == 0.0) {
            return new_d4(&d4_neg_inf);
        }

        solType *x = new solType;
        x->x[0] = std::log(a->x[0]);

        d4_assign(x, d4_minus(d4_add(x, d4_mul(a, d4_exp(d4_neg(x)))), new_d4(&d4_one)));
        d4_assign(x, d4_minus(d4_add(x, d4_mul(a, d4_exp(d4_neg(x)))), new_d4(&d4_one)));
        d4_assign(x, d4_minus(d4_add(x, d4_mul(a, d4_exp(d4_neg(x)))), new_d4(&d4_one)));

        return new_d4(x);
    }

    solType *_log10(solType *a) {
        solType *b;
        d4_assign(b, d4_div(d4_log(a), &d4_log10));
        return new_d4(b);
    }

    int d4_to_int(solType *a) {
        return static_cast<int>(a->x[0]);
    }

    solType *d4_abs(solType *a) {
        return (a->x[0] < 0.0) ? d4_neg(a) : a;
    }

    solType *d4_floor(solType *a) {
        double x0, x1, x2, x3;
        x1 = x2 = x3 = 0.0;
        x0 = std::floor(a->x[0]);

        if (x0 == a->x[0]) {
            x1 = std::floor(a->x[1]);

            if (x1 == a->x[1]) {
                x2 = std::floor(a->x[2]);

                if (x2 == a->x[2]) {
                    x3 = std::floor(a->x[3]);
                }
            }

            d4_renorm(x0, x1, x2, x3);
            return new_d4(x0, x1, x2, x3);
        }

        return new_d4(x0, x1, x2, x3);
    }

    solType *new_d4(double d, double d1, double d2, double d3) {
        solType *x = new solType;
        x->x[0] = d;
        x->x[1] = d1;
        x->x[2] = d2;
        x->x[3] = d3;
        return x;
    }

    std::string
    d4_to_string(solType *x_, int precision, int width, std::ios_base::fmtflags fmt, bool showpos, bool uppercase,
                 char fill) {
        std::string s;
        bool fixed = (fmt & std::ios_base::fixed) != 0;
        bool sgn = true;
        int i, e = 0;
        solType *x = new_d4(x_);

        if (std::isinf(x->x[0])) {
            if (x->x[0] < 0.0 || (x->x[0] == 0.0 && x->x[1] < 0.0))
                s += '-';
            else if (showpos)
                s += '+';
            else
                sgn = false;
            s += uppercase ? "INF" : "inf";
        } else if (std::isnan(x->x[0]) || std::isnan(x->x[1]) || std::isnan(x->x[2]) || std::isnan(x->x[3])) {
            s = uppercase ? "NAN" : "nan";
            sgn = false;
        } else {
            if (x->x[0] < 0.0 || (x->x[0] == 0.0 && x->x[1] < 0.0))
                s += '-';
            else if (showpos)
                s += '+';
            else
                sgn = false;

            if (x->x[0] == 0.0 && x->x[1] == 0.0) {
                s += '0';
                if (precision > 0) {
                    s += '.';
                    s.append(precision, '0');
                }
            } else {
                int off = (fixed ? (1 + d4_to_int(d4_floor(_log10(d4_abs(x))))) : 1);
                int d = precision + off;

                int d_with_extra = d;
                if (fixed)
                    d_with_extra = std::max(120, d);

                if (fixed && (precision == 0) && d4_l(d4_abs(x), to_d4(1.0))) {
                    if (d4_ge(d4_abs(x), to_d4(0.5)))
                        s += '1';
                    else
                        s += '0';

                    return s;
                }

                if (fixed && d <= 0) {
                    s += '0';
                    if (precision > 0) {
                        s += '.';
                        s.append(precision, '0');
                    }
                } else {

                    char *t;
                    int j;

                    if (fixed) {
                        t = new char[d_with_extra + 1];
                        d4_to_digits(x, t, e, d_with_extra);
                    } else {
                        t = new char[d + 1];
                        d4_to_digits(x, t, e, d);
                    }


                    if (fixed) {
                        d4_round_string_qd(t, d + 1, &off);

                        if (off > 0) {
                            for (i = 0; i < off; i++) s += t[i];
                            if (precision > 0) {
                                s += '.';
                                for (j = 0; j < precision; j++, i++) s += t[i];
                            }
                        } else {
                            s += "0.";
                            if (off < 0) s.append(-off, '0');
                            for (i = 0; i < d; i++) s += t[i];
                        }
                    } else {
                        s += t[0];
                        if (precision > 0) s += '.';

                        for (i = 1; i <= precision; i++)
                            s += t[i];

                    }
                    delete[] t;
                }
            }

            if (fixed && (precision > 0)) {
                double from_string = atof(s.c_str());

                if (fabs(from_string / x->x[0]) > 3.0) {

                    int point_position;
                    char temp;

                    for (i = 1; i < s.length(); i++) {
                        if (s[i] == '.') {
                            s[i] = s[i - 1];
                            s[i - 1] = '.';
                            break;
                        }
                    }

                    from_string = atof(s.c_str());
                    if (fabs(from_string / x->x[0]) > 3.0) {
                    }
                }
            }

            if (!fixed) {
                s += uppercase ? 'E' : 'e';
                d4_append_expn(s, e);
            }
        }

        int len = s.length();
        if (len < width) {
            int delta = width - len;
            if (fmt & std::ios_base::internal) {
                if (sgn)
                    s.insert(static_cast<std::string::size_type>(1), delta, fill);
                else
                    s.insert(static_cast<std::string::size_type>(0), delta, fill);
            } else if (fmt & std::ios_base::left) {
                s.append(delta, fill);
            } else {
                s.insert(static_cast<std::string::size_type>(0), delta, fill);
            }
        }

        return s;
    }

    double d4_quick_two_sum(double a, double b, double &err) {
        double s = a + b;
        err = b - (s - a);
        return s;
    }

    double d4_two_sum(double a, double b, double &err) {
        double s = a + b;
        double bb = s - a;
        err = (a - (s - bb)) + (b - bb);
        return s;
    }

    double d4_quick_three_accum(double &a, double &b, double c) {
        double s;
        bool za, zb;

        s = d4_two_sum(b, c, b);
        s = d4_two_sum(a, s, a);

        za = (a != 0.0);
        zb = (b != 0.0);

        if (za && zb)
            return s;

        if (!zb) {
            b = a;
            a = s;
        } else {
            a = s;
        }

        return 0.0;
    }

    void d4_renorm(double &c0, double &c1, double &c2, double &c3) {
        double s0, s1, s2 = 0.0, s3 = 0.0;

        if (isinf(c0)) return;

        s0 = d4_quick_two_sum(c2, c3, c3);
        s0 = d4_quick_two_sum(c1, s0, c2);
        c0 = d4_quick_two_sum(c0, s0, c1);

        s0 = c0;
        s1 = c1;
        if (s1 != 0.0) {
            s1 = d4_quick_two_sum(s1, c2, s2);
            if (s2 != 0.0)
                s2 = d4_quick_two_sum(s2, c3, s3);
            else
                s1 = d4_quick_two_sum(s1, c3, s2);
        } else {
            s0 = d4_quick_two_sum(s0, c2, s1);
            if (s1 != 0.0)
                s1 = d4_quick_two_sum(s1, c3, s2);
            else
                s0 = d4_quick_two_sum(s0, c3, s1);
        }

        c0 = s0;
        c1 = s1;
        c2 = s2;
        c3 = s3;
    }

    solType *d4_add(solType *a, solType *b) {
        int i, j, k;
        double s, t;
        double u, v;
        solType *x = to_d4(0.0);

        i = j = k = 0;
        if (abs(a->x[i]) > abs(b->x[j]))
            u = a->x[i++];
        else
            u = b->x[j++];
        if (abs(a->x[i]) > abs(b->x[j]))
            v = a->x[i++];
        else
            v = b->x[j++];

        u = d4_quick_two_sum(u, v, v);

        while (k < 4) {
            if (i >= 4 && j >= 4) {
                x->x[k] = u;
                if (k < 3)
                    x->x[++k] = v;
                break;
            }

            if (i >= 4)
                t = b->x[j++];
            else if (j >= 4)
                t = a->x[i++];
            else if (abs(a->x[i]) > abs(b->x[j])) {
                t = a->x[i++];
            } else
                t = b->x[j++];

            s = d4_quick_three_accum(u, v, t);

            if (s != 0.0) {
                x->x[k++] = s;
            }
        }

        for (k = i; k < 4; k++)
            x->x[3] += a->x[k];
        for (k = j; k < 4; k++)
            x->x[3] += b->x[k];

        d4_renorm(x->x[0], x->x[1], x->x[2], x->x[3]);
        return new_d4(x);
    }

    void d4_split(double a, double &hi, double &lo) {
        double temp;
        if (a > _QD_SPLIT_THRESH || a < -_QD_SPLIT_THRESH) {
            a *= 3.7252902984619140625e-09;  // 2^-28
            temp = _QD_SPLITTER * a;
            hi = temp - (temp - a);
            lo = a - hi;
            hi *= 268435456.0;
            lo *= 268435456.0;
        } else {
            temp = _QD_SPLITTER * a;
            hi = temp - (temp - a);
            lo = a - hi;
        }
    }

    double d4_two_prod(double a, double b, double &err) {
        double a_hi, a_lo, b_hi, b_lo;
        double p = a * b;
        d4_split(a, a_hi, a_lo);
        d4_split(b, b_hi, b_lo);
        err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
        return p;
    }

    void d4_three_sum(double &a, double &b, double &c) {
        double t1, t2, t3;
        t1 = d4_two_sum(a, b, t2);
        a = d4_two_sum(c, t1, t3);
        b = d4_two_sum(t2, t3, c);
    }

    void d4_renorm(double &c0, double &c1, double &c2, double &c3, double &c4) {
        double s0, s1, s2 = 0.0, s3 = 0.0;

        if (isinf(c0)) return;

        s0 = d4_quick_two_sum(c3, c4, c4);
        s0 = d4_quick_two_sum(c2, s0, c3);
        s0 = d4_quick_two_sum(c1, s0, c2);
        c0 = d4_quick_two_sum(c0, s0, c1);

        s0 = c0;
        s1 = c1;

        s0 = d4_quick_two_sum(c0, c1, s1);
        if (s1 != 0.0) {
            s1 = d4_quick_two_sum(s1, c2, s2);
            if (s2 != 0.0) {
                s2 = d4_quick_two_sum(s2, c3, s3);
                if (s3 != 0.0)
                    s3 += c4;
                else
                    s2 += c4;
            } else {
                s1 = d4_quick_two_sum(s1, c3, s2);
                if (s2 != 0.0)
                    s2 = d4_quick_two_sum(s2, c4, s3);
                else
                    s1 = d4_quick_two_sum(s1, c4, s2);
            }
        } else {
            s0 = d4_quick_two_sum(s0, c2, s1);
            if (s1 != 0.0) {
                s1 = d4_quick_two_sum(s1, c3, s2);
                if (s2 != 0.0)
                    s2 = d4_quick_two_sum(s2, c4, s3);
                else
                    s1 = d4_quick_two_sum(s1, c4, s2);
            } else {
                s0 = d4_quick_two_sum(s0, c3, s1);
                if (s1 != 0.0)
                    s1 = d4_quick_two_sum(s1, c4, s2);
                else
                    s0 = d4_quick_two_sum(s0, c4, s1);
            }
        }

        c0 = s0;
        c1 = s1;
        c2 = s2;
        c3 = s3;
    }

    solType *d4_mul(solType *a, solType *b) {
        double p0, p1, p2, p3, p4, p5;
        double q0, q1, q2, q3, q4, q5;
        double p6, p7, p8, p9;
        double q6, q7, q8, q9;
        double r0, r1;
        double t0, t1;
        double s0, s1, s2;

        p0 = d4_two_prod(a->x[0], b->x[0], q0);

        p1 = d4_two_prod(a->x[0], b->x[1], q1);
        p2 = d4_two_prod(a->x[1], b->x[0], q2);

        p3 = d4_two_prod(a->x[0], b->x[2], q3);
        p4 = d4_two_prod(a->x[1], b->x[1], q4);
        p5 = d4_two_prod(a->x[2], b->x[0], q5);

        d4_three_sum(p1, p2, q0);

        d4_three_sum(p2, q1, q2);
        d4_three_sum(p3, p4, p5);

        s0 = d4_two_sum(p2, p3, t0);
        s1 = d4_two_sum(q1, p4, t1);
        s2 = q2 + p5;
        s1 = d4_two_sum(s1, t0, t0);
        s2 += (t0 + t1);

        p6 = d4_two_prod(a->x[0], b->x[3], q6);
        p7 = d4_two_prod(a->x[1], b->x[2], q7);
        p8 = d4_two_prod(a->x[2], b->x[1], q8);
        p9 = d4_two_prod(a->x[3], b->x[0], q9);

        q0 = d4_two_sum(q0, q3, q3);
        q4 = d4_two_sum(q4, q5, q5);
        p6 = d4_two_sum(p6, p7, p7);
        p8 = d4_two_sum(p8, p9, p9);

        t0 = d4_two_sum(q0, q4, t1);
        t1 += (q3 + q5);

        r0 = d4_two_sum(p6, p8, r1);
        r1 += (p7 + p9);

        q3 = d4_two_sum(t0, r0, q4);
        q4 += (t1 + r1);

        t0 = d4_two_sum(q3, s1, t1);
        t1 += q4;

        t1 += a->x[1] * b->x[3] + a->x[2] * b->x[2] + a->x[3] * b->x[1] + q6 + q7 + q8 + q9 + s2;

        d4_renorm(p0, p1, s0, t0, t1);
        return new_d4(p0, p1, s0, t0);
    }


    solType *d4_minus(solType *a, solType *b) {
        solType *c = new solType;
        d4_assign(c, d4_neg(b));
        solType *d = new solType;
        d4_assign(d, d4_add(a, c));
        return new_d4(d);
    }

    void d4_three_sum2(double &a, double &b, double &c) {
        double t1, t2, t3;
        t1 = d4_two_sum(a, b, t2);
        a = d4_two_sum(c, t1, t3);
        b = t2 + t3;
    }

    solType *d4_mul_qd_d(solType *a, double b) {
        double p0, p1, p2, p3;
        double q0, q1, q2;
        double s0, s1, s2, s3, s4;

        p0 = d4_two_prod(a->x[0], b, q0);
        p1 = d4_two_prod(a->x[1], b, q1);
        p2 = d4_two_prod(a->x[2], b, q2);
        p3 = a->x[3] * b;

        s0 = p0;

        s1 = d4_two_sum(q0, p1, s2);

        d4_three_sum(s2, q1, p2);

        d4_three_sum2(q1, q2, p3);
        s3 = q1;

        s4 = q2 + p2;

        d4_renorm(s0, s1, s2, s3, s4);
        return new_d4(s0, s1, s2, s3);

    }

    solType *d4_div(solType *a, solType *b) {
        double q0, q1, q2, q3;

        solType *r = new solType;

        q0 = a->x[0] / b->x[0];
        d4_assign(r, d4_minus(a, d4_mul_qd_d(b, q0)));

        q1 = r->x[0] / b->x[0];
        d4_assign(r, d4_minus(r, d4_mul_qd_d(b, q1)));

        q2 = r->x[0] / b->x[0];
        d4_assign(r, d4_minus(r, d4_mul_qd_d(b, q2)));

        q3 = r->x[0] / b->x[0];
        d4_assign(r, d4_minus(r, d4_mul_qd_d(b, q3)));

        double q4 = r->x[0] / b->x[0];

        d4_renorm(q0, q1, q2, q3, q4);

        return new_d4(q0, q1, q2, q3);
    }

    void d4_assign(solType *a, solType *b) {
        a->x[0] = b->x[0];
        a->x[1] = b->x[1];
        a->x[2] = b->x[2];
        a->x[3] = b->x[3];
    }

    solType *d4_ldexp(solType *a, int n) {
        return new_d4(std::ldexp(a->x[0], n), std::ldexp(a->x[1], n),
                      std::ldexp(a->x[2], n), std::ldexp(a->x[3], n));
    }

    solType *d4_exp(solType *a) {

        const double k = ldexp(1.0, 16);
        const double inv_k = 1.0 / k;

        if (a->x[0] <= -709.0)
            return new_d4(&d4_zero);

        if (a->x[0] >= 709.0)
            return new_d4(&d4_inf);

        if (d4_is_zero(a))
            return new_d4(&d4_one);

        if (d4_is_one(a))
            return new_d4(&d4_e);

        double m = std::floor(a->x[0] / d4_log2.x[0] + 0.5);
        solType *r = new_d4(d4_mul_pwr2(d4_minus(a, d4_mul(&d4_log2, to_d4(m))), inv_k));
        solType *s=new solType, *p=new solType, *t=new solType;
        double thresh = inv_k * d4_eps;

        d4_assign(p, d4_sqr(r));
        d4_assign(s, d4_add(r, d4_mul_pwr2(p, 0.5)));
        int i = 0;
        do {
            d4_assign(p, d4_mul(p, r));
            d4_assign(t, d4_mul(p, &d4_inv_fact[i++]));
            d4_assign(s, d4_add(s, t));
        } while (std::abs(t->x[0]) > thresh && i < 9);

        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(d4_mul_pwr2(s, 2.0), d4_sqr(s)));
        d4_assign(s, d4_add(s, &d4_one));

        solType *asdf;
        d4_assign(asdf, d4_ldexp(s, static_cast<int>(m)));
        return new_d4(asdf);
    }

    bool d4_is_zero(solType *x) {
        return (x->x[0] == 0.0);
    }

    bool d4_is_one(solType *x) {
        return (x->x[0] == 1.0 && x->x[1] == 0.0 && x->x[2] == 0.0 && x->x[3] == 0.0);
    }

    solType *d4_mul_pwr2(solType *a, double b) {
        return new_d4(a->x[0] * b, a->x[1] * b, a->x[2] * b, a->x[3] * b);
    }

    double d4_two_sqr(double a, double &err) {
        double hi, lo;
        double q = a * a;
        d4_split(a, hi, lo);
        err = ((hi * hi - q) + 2.0 * hi * lo) + lo * lo;
        return q;
    }

    solType *d4_sqr(solType *a) {
        double p0, p1, p2, p3, p4, p5;
        double q0, q1, q2, q3;
        double s0, s1;
        double t0, t1;

        p0 = d4_two_sqr(a->x[0], q0);
        p1 = d4_two_prod(2.0 * a->x[0], a->x[1], q1);
        p2 = d4_two_prod(2.0 * a->x[0], a->x[2], q2);
        p3 = d4_two_sqr(a->x[1], q3);

        p1 = d4_two_sum(q0, p1, q0);

        q0 = d4_two_sum(q0, q1, q1);
        p2 = d4_two_sum(p2, p3, p3);

        s0 = d4_two_sum(q0, p2, t0);
        s1 = d4_two_sum(q1, p3, t1);

        s1 = d4_two_sum(s1, t0, t0);
        t0 += t1;

        s1 = d4_quick_two_sum(s1, t0, t0);
        p2 = d4_quick_two_sum(s0, s1, t1);
        p3 = d4_quick_two_sum(t1, t0, q0);

        p4 = 2.0 * a->x[0] * a->x[3];
        p5 = 2.0 * a->x[1] * a->x[2];

        p4 = d4_two_sum(p4, p5, p5);
        q2 = d4_two_sum(q2, q3, q3);

        t0 = d4_two_sum(p4, q2, t1);
        t1 = t1 + p5 + q3;

        p3 = d4_two_sum(p3, t0, p4);
        p4 = p4 + q0 + t1;

        d4_renorm(p0, p1, p2, p3, p4);
        return new_d4(p0, p1, p2, p3);

    }

    void d4_to_digits(solType *x, char *s, int &expn, int precision) {
        int D = precision + 1;

        solType *r = d4_abs(x);
        int e;
        int i, d;

        if (x->x[0] == 0.0) {
            expn = 0;
            for (i = 0; i < precision; i++) s[i] = '0';
            return;
        }

        e = static_cast<int>(std::floor(std::log10(std::abs(x->x[0]))));

        if (e < -300) {
            solType *tmp=new solType;
            tmp->x[0] = 10.0;
            tmp->x[1] = 0.0;
            tmp->x[2] = 0.0;
            tmp->x[3] = 0.0;
            d4_assign(r, d4_mul(r, d4_pow(tmp, 300)));
            d4_assign(r, d4_div(r, d4_pow(tmp, (e + 300))));
        } else if (e > 300) {
            solType *tmp=new solType;
            tmp->x[0] = 10.0;
            tmp->x[1] = 0.0;
            tmp->x[2] = 0.0;
            tmp->x[3] = 0.0;
            d4_assign(r, d4_ldexp(r, -53));
            d4_assign(r, d4_div(r, d4_pow(tmp, e)));
            d4_assign(r, d4_ldexp(r, 53));
        } else {
            solType *tmp=new solType;
            tmp->x[0] = 10.0;
            tmp->x[1] = 0.0;
            tmp->x[2] = 0.0;
            tmp->x[3] = 0.0;
            d4_assign(r, d4_div(r, d4_pow(tmp, e)));
        }

        if (d4_ge(r, to_d4(10.0))) {
            d4_assign(r, d4_div(r, to_d4(10.0)));
            e++;
        } else if (d4_l(r, to_d4(1.0))) {
            d4_assign(r, d4_mul(r, to_d4(10.0)));
            e--;
        }

        if (d4_ge(r, to_d4(10.0)) || d4_l(r, to_d4(1.0))) {
            return;
        }

        for (i = 0; i < D; i++) {
            d = static_cast<int>(r->x[0]);
            d4_assign(r, d4_minus(r, to_d4(d)));
            d4_assign(r, d4_mul(r, to_d4(10.0)));

            s[i] = static_cast<char>(d + '0');
        }

        for (i = D - 1; i > 0; i--) {
            if (s[i] < '0') {
                s[i - 1]--;
                s[i] += 10;
            } else if (s[i] > '9') {
                s[i - 1]++;
                s[i] -= 10;
            }
        }

        if (s[0] <= '0') {
            return;
        }

        if (s[D - 1] >= '5') {
            s[D - 2]++;

            i = D - 2;
            while (i > 0 && s[i] > '9') {
                s[i] -= 10;
                s[--i]++;
            }
        }

        if (s[0] > '9') {
            e++;
            for (i = precision; i >= 2; i--) s[i] = s[i - 1];
            s[0] = '1';
            s[1] = '0';
        }

        s[precision] = 0;
        expn = e;
    }


    solType *d4_pow(solType *a, int n) {
        if (n == 0)
            return to_d4(1.0);

        solType *r=new solType;
        d4_assign(r, a);
        solType *s=new solType;
        d4_assign(s, to_d4(1.0));
        int N = std::abs(n);

        if (N > 1) {

            while (N > 0) {
                if (N % 2 == 1) {
                    d4_assign(s, d4_mul(s, r));
                }
                N /= 2;
                if (N > 0)
                    d4_assign(r, d4_sqr(r));
            }

        } else {
            d4_assign(s, r);
        }

        if (n < 0)
            return new_d4(d4_div(to_d4(1.0), s));

        return new_d4(s);
    }

    void d4_round_string_qd(char *s, int precision, int *offset) {

        int i;
        int D = precision;

        if (s[D - 1] >= '5') {
            s[D - 2]++;

            i = D - 2;
            while (i > 0 && s[i] > '9') {
                s[i] -= 10;
                s[--i]++;
            }
        }

        if (s[0] > '9') {
            for (i = precision; i >= 2; i--) s[i] = s[i - 1];
            s[0] = '1';
            s[1] = '0';

            (*offset)++;
            precision++;
        }

        s[precision] = 0;
    }

    void d4_append_expn(std::string &str, int expn) {
        int k;

        str += (expn < 0 ? '-' : '+');
        expn = std::abs(expn);

        if (expn >= 100) {
            k = (expn / 100);
            str += '0' + k;
            expn -= 100 * k;
        }

        k = (expn / 10);
        str += '0' + k;
        expn -= 10 * k;

        str += '0' + expn;
    }


}