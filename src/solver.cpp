#include <gpusatutils.h>
#include <math.h>
#include <solver.h>

using namespace std;

int solveProblem(treedecType &decomp, satformulaType &formula, bagType &node) {
    for (int i = 0; i < node.nume; i++) {
        int edge = node.edges[i] - 1;
        solveProblem(decomp, formula, decomp.bags[edge]);
    }
    for (long id = 0; id < pow(2, node.numv); id++) {
        solveNode(&decomp, &formula, &node, id);
    }
}

void solveNode(treedecType *decomp, satformulaType *formula, bagType *node, long id) {
    solutionType *sol = &(*node).solution[id];
    long i;
    long bagnum = (*node).id;
    if ((*node).nume > 1) {
        //join node
        for (i = 0; i < (*node).numv; i++) {
            if ((id & (1 << i)) >> i) {
                (*sol).vars[i] = (*node).vertices[i];
            } else {
                (*sol).vars[i] = (*node).vertices[i] * -1;
            }
        }
        long n1 = (*decomp).bags[(*node).edges[0] - 1].solution[id].n;
        long n2 = (*decomp).bags[(*node).edges[1] - 1].solution[id].n;
        (*sol).n = n1 * n2;
        for (i = 2; i < (*node).nume; i++) {
            long n = (*decomp).bags[(*node).edges[i] - 1].solution[id].n;
            (*sol).n = (*sol).n * n;
        }
    } else if ((*node).nume == 1 && (*decomp).bags[(*node).edges[0] - 1].numv > (*node).numv) {
        //forget node
        for (i = 0; i < (*node).numv; i++) {
            if ((id & (1 << i)) >> i) {
                (*sol).vars[i] = (*node).vertices[i];
            } else {
                (*sol).vars[i] = (*node).vertices[i] * -1;
            }
        }
        (*sol).n = 0;
        for (i = 0; i < (*decomp).bags[(*node).edges[0] - 1].numSol; i++) {
            solutionType *otherSol = &(*decomp).bags[(*node).edges[0] - 1].solution[i];
            if ((*otherSol).n != 0) {
                long a = 0, b = 0;
                int eq = 1;
                for (a = 0; a < (*node).numv; a++) {
                    long var1 = (*sol).vars[a], var2 = (*otherSol).vars[b];
                    while ((((*sol).vars[a] != (*otherSol).vars[b]) && ((*sol).vars[a] != -(*otherSol).vars[b]))) {
                        b++;
                        var2 = (*otherSol).vars[b];
                    }
                    if ((var1 != var2)) {
                        eq = 0;
                        break;
                    }
                    b++;
                }
                if (eq) {
                    (*sol).n = (*sol).n + (*otherSol).n;
                }
            }
        }
    } else {
        int unsat = 0;
        //check formula
        for (i = 0; i < (*formula).numclauses && !unsat; i++) {
            int satC = 0;
            long a;
            //check clause
            for (a = 0; a < (*formula).clauses[i].numVars && !satC; a++) {
                int found = 0;
                long b;
                for (b = 0; b < node->numv && !satC; b++) {
                    if (((*formula).clauses[i].var[a] == (*node).vertices[b]) ||
                        ((*formula).clauses[i].var[a] == -(*node).vertices[b])) {
                        found = 1;
                        if ((*formula).clauses[i].var[a] < 0) {
                            if (((id & (1 << b)) >> b) == 0) {
                                satC = 1;
                            }
                        } else {
                            if (((id & (1 << b)) >> b) == 1) {
                                satC = 1;
                            }
                        }
                    }
                }
                if (!found) {
                    satC = 1;
                }
            }
            if (!satC) {
                unsat = 1;
            }
        }

        if ((*node).nume == 0) {
            //leaf node
            (*sol).n = !unsat;
            for (i = 0; i < (*node).numv; i++) {
                if ((id & (1 << i)) >> i) {
                    (*sol).vars[i] = (*node).vertices[i];
                } else {
                    (*sol).vars[i] = (*node).vertices[i] * -1;
                }
            }
        } else if ((*node).nume == 1 && (*decomp).bags[(*node).edges[0] - 1].numv < (*node).numv) {
            //introduce node
            for (i = 0; i < (*node).numv; i++) {
                if ((id & (1 << i)) >> i) {
                    (*sol).vars[i] = (*node).vertices[i];
                } else {
                    (*sol).vars[i] = (*node).vertices[i] * -1;
                }
            }
            if (unsat) {
                (*sol).n = 0;
            } else {
                (*sol).n = 0;
                for (i = 0; i < (*decomp).bags[(*node).edges[0] - 1].numSol; i++) {
                    solutionType *otherSol = &(*decomp).bags[(*node).edges[0] - 1].solution[i];
                    long a = 0, b = 0;
                    int eq = 1;
                    for (b = 0; b < (*decomp).bags[(*node).edges[0] - 1].numv; b++) {
                        long var1 = (*sol).vars[a], var2 = (*otherSol).vars[b];
                        while (((var1 != var2) && (-var1 != var2))) {
                            a++;
                            var1 = (*sol).vars[a];
                        }
                        if ((var1 != var2)) {
                            eq = 0;
                            break;
                        }
                        a++;
                    }
                    if (eq) {
                        (*sol).n = (*otherSol).n;
                        break;
                    }
                }
            }
        }
    }
}