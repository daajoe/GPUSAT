/**
 * adaptation of https://github.com/scibuilder/QD for opencl
 */

#ifndef GPUSAT_D4_UTILS_H
#define GPUSAT_D4_UTILS_H

#include <types.h>
#include <iostream>

#define _QD_SPLITTER 134217729.0               // = 2^27 + 1
#define _QD_SPLIT_THRESH 6.69692879491417e+299 // = 2^996

namespace gpusat {
    /// turn x into string
    std::string d4_to_string(solType *x, int precision = 62, int width = 0,
                             std::ios_base::fmtflags fmt = static_cast<std::ios_base::fmtflags>(0),
                             bool showpos = false, bool uppercase = false, char fill = ' ');

    /// a - b
    solType *d4_minus(solType *a, solType *b);

    /// a + b
    solType *d4_add(solType *a, solType *b);

    /// a * b
    solType *d4_mul(solType *a, solType *b);

    /// a / b
    solType *d4_div(solType *a, solType *b);

    /// a = b
    void d4_assign(solType *a, solType *b);

    /// a ^ n
    solType *d4_pow(solType *a, int n);

    /// -x
    solType *d4_neg(solType *x);

    solType *new_d4(solType *x);

    solType *to_d4(double x);

    solType *new_d4(double d, double d1, double d2, double d3);

    static solType d4_e = {2.718281828459045091e+00, 1.445646891729250158e-16, -2.127717108038176765e-33,
                           1.515630159841218954e-49};
    static solType d4_log10 = {2.302585092994045901e+00, -2.170756223382249351e-16, -9.984262454465776570e-33,
                               -4.023357454450206379e-49};
    static solType d4_neg_inf = {-std::numeric_limits<double>::infinity(),
                                 -std::numeric_limits<double>::infinity(),
                                 -std::numeric_limits<double>::infinity(),
                                 -std::numeric_limits<double>::infinity()};
    static solType d4_inf = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    static solType d4_zero = {0.0, 0.0, 0.0, 0.0};
    static solType d4_one = {1.0, 0.0, 0.0, 0.0};
    static solType d4_log2 = {6.931471805599452862e-01, 2.319046813846299558e-17, 5.707708438416212066e-34,
                              -3.582432210601811423e-50};
    static const int n_inv_fact = 15;
    static solType d4_inv_fact[n_inv_fact] = {
            {1.66666666666666657e-01, 9.25185853854297066e-18,  5.13581318503262866e-34,  2.85094902409834186e-50},
            {4.16666666666666644e-02, 2.31296463463574266e-18,  1.28395329625815716e-34,  7.12737256024585466e-51},
            {8.33333333333333322e-03, 1.15648231731787138e-19,  1.60494162032269652e-36,  2.22730392507682967e-53},
            {1.38888888888888894e-03, -5.30054395437357706e-20, -1.73868675534958776e-36, -1.63335621172300840e-52},
            {1.98412698412698413e-04, 1.72095582934207053e-22,  1.49269123913941271e-40,  1.29470326746002471e-58},
            {2.48015873015873016e-05, 2.15119478667758816e-23,  1.86586404892426588e-41,  1.61837908432503088e-59},
            {2.75573192239858925e-06, -1.85839327404647208e-22, 8.49175460488199287e-39,  -5.72661640789429621e-55},
            {2.75573192239858883e-07, 2.37677146222502973e-23,  -3.26318890334088294e-40, 1.61435111860404415e-56},
            {2.50521083854417202e-08, -1.44881407093591197e-24, 2.04267351467144546e-41,  -8.49632672007163175e-58},
            {2.08767569878681002e-09, -1.20734505911325997e-25, 1.70222792889287100e-42,  1.41609532150396700e-58},
            {1.60590438368216133e-10, 1.25852945887520981e-26,  -5.31334602762985031e-43, 3.54021472597605528e-59},
            {1.14707455977297245e-11, 2.06555127528307454e-28,  6.88907923246664603e-45,  5.72920002655109095e-61},
            {7.64716373181981641e-13, 7.03872877733453001e-30,  -7.82753927716258345e-48, 1.92138649443790242e-64},
            {4.77947733238738525e-14, 4.39920548583408126e-31,  -4.89221204822661465e-49, 1.20086655902368901e-65},
            {2.81145725434552060e-15, 1.65088427308614326e-31,  -2.87777179307447918e-50, 4.27110689256293549e-67}
    };
    static double d4_eps = 1.21543267145725e-63;

    solType *d4_log(solType *a);

    solType *_log10(solType *a);

    double d4_quick_two_sum(double a, double b, double &err);

    double d4_two_sum(double a, double b, double &err);

    double d4_quick_three_accum(double &a, double &b, double c);

    void d4_renorm(double &c0, double &c1, double &c2, double &c3);

    void d4_split(double a, double &hi, double &lo);

    double d4_two_prod(double a, double b, double &err);

    void d4_three_sum(double &a, double &b, double &c);

    void d4_renorm(double &c0, double &c1, double &c2, double &c3, double &c4);

    void d4_three_sum2(double &a, double &b, double &c);;

    solType *d4_mul_qd_d(solType *a, double b);

    solType *d4_ldexp(solType *a, int n);

    solType *d4_exp(solType *a);

    bool d4_is_zero(solType *x);

    bool d4_is_one(solType *x);

    solType *d4_mul_pwr2(solType *a, double b);

    double d4_two_sqr(double a, double &err);

    solType *d4_sqr(solType *a);

    int d4_to_int(solType *a);

    solType *d4_floor(solType *a);

    void d4_to_digits(solType *x, char *s, int &expn, int precision);

    void d4_round_string_qd(char *s, int precision, int *offset);

    void d4_append_expn(std::string &str, int expn);

    ///  a >= b
    inline bool d4_ge(solType *a, solType *b) {
        return (a->x[0] > b->x[0] ||
                (a->x[0] == b->x[0] && (a->x[1] > b->x[1] ||
                                        (a->x[1] == b->x[1] && (a->x[2] > b->x[2] ||
                                                                (a->x[2] == b->x[2] && a->x[3] >= b->x[3]))))));
    }

    /// a < b
    inline bool d4_l(solType *a, solType *b) {
        return (a->x[0] < b->x[0] ||
                (a->x[0] == b->x[0] && (a->x[1] < b->x[1] ||
                                        (a->x[1] == b->x[1] && (a->x[2] < b->x[2] ||
                                                                (a->x[2] == b->x[2] && a->x[3] < b->x[3]))))));
    }

}
#endif //GPUSAT_D4_UTILS_H
