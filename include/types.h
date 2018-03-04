#ifndef GPUSAT_TYPES_H_H
#define GPUSAT_TYPES_H_H

#define alloca __builtin_alloca

#include <CL/cl_platform.h>
#include <cmath>
#include <list>
#include <vector>
#include <set>

struct d4_Type {
    cl_double x[4];

    d4_Type() {
        x[0] = 0.0;
        x[1] = 0.0;
        x[2] = 0.0;
        x[3] = 0.0;
    }

    d4_Type(double a, double b, double c, double d) {
        x[0] = a;
        x[1] = b;
        x[2] = c;
        x[3] = d;
    }

    d4_Type(double a) {
        x[0] = a;
        x[1] = 0.0;
        x[2] = 0.0;
        x[3] = 0.0;
    }

    d4_Type &operator=(double a) {
        x[0] = a;
        x[1] = 0.0;
        x[2] = 0.0;
        x[3] = 0.0;
        return *this;
    }

    d4_Type &operator=(d4_Type a) {
        x[0] = a.x[0];
        x[1] = a.x[1];
        x[2] = a.x[2];
        x[3] = a.x[3];
        return *this;
    }

    d4_Type &operator=(d4_Type *a) {
        x[0] = a->x[0];
        x[1] = a->x[1];
        x[2] = a->x[2];
        x[3] = a->x[3];
        return *this;
    }
};

#ifdef sType_Double
#define solType cl_double
#else
#define solType d4_Type
#endif

namespace dual {
    typedef struct {
        cl_long numC;
        cl_long numCE1;
        cl_long numCE2;
        cl_long minId1;
        cl_long maxId1;
        cl_long minId2;
        cl_long maxId2;
        cl_long startIDNode;
        cl_long startIDEdge1;
        cl_long startIDEdge2;
        cl_long numV;
        cl_long numVE1;
        cl_long numVE2;
    } sJVars;

    typedef struct {
        cl_long numCI;
        cl_long numCE;
        cl_long numVF;
        cl_long numVI;
        cl_long numVE;
        cl_long combinations;
        cl_long minIdE;
        cl_long maxIdE;
        cl_long startIDF;
        cl_long startIDE;
        cl_long numCF;
    } sIFVars;
}
namespace gpusat {

    /// type for a bag in the tree decomposition
    struct bagType {
        cl_long numSol = 0;
        std::vector<cl_long> variables;
        std::vector<cl_long> edges;
        solType **solution = nullptr;
    };

    /// type for saving a tree decomposition
    struct treedecType {
        cl_long numb = 0;
        cl_long numVars = 0;
        bagType *bags = nullptr;
    };

    /// type for preprocessing a tree decomposition
    struct preebagType {
        cl_long id = 0;
        std::vector<cl_long> variables;
        std::vector<preebagType *> edges;
    };

    /// type for saving a tree decomposition
    struct preetreedecType {
        cl_long numb = 0;
        cl_long numVars = 0;
        preebagType *bags = nullptr;
    };

    bool compTreedType(const preebagType *a, const preebagType *b);

    /// type for saving the sat formula
    struct satformulaType {
        cl_long numVars = 0;
        cl_long numWeights = 0;
        bool unsat = false;
        solType *variableWeights = nullptr;
        std::vector<std::vector<cl_long>> clauses;
        std::vector<cl_long> facts;
    };

    enum graphTypes {
        PRIMAL, INCIDENCE, DUAL, NONE
    };

    enum precisionTypes {
        DOUBLE, D4
    };

    bool compVars(const cl_long &a, const cl_long &b);

    bool compVarsEq(const cl_long &a, const cl_long &b);
}

#endif //GPUSAT_TYPES_H_H
