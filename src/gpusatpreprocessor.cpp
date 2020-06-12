#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <gpusatparser.h>
#include <gpusatpreprocessor.h>

namespace gpusat {
    void Preprocessor::preprocessDecomp(BagType *decomp, cl_long combineWidth) {

        bool changed = true;
        // try to merge child nodes
        while (changed) {
            changed = false;
            for (long a = 0; a < decomp->edges.size() && !changed; a++) {
                for (long b = 0; b < decomp->edges.size() && !changed; b++) {
                    if (a != b && ((decomp->edges[a]->variables.size() < combineWidth && decomp->edges[b]->variables.size() < combineWidth) || decomp->edges[a]->variables.size() == 0 || decomp->edges[b]->variables.size() == 0) && decomp->edges.size() > 1) {
                        std::vector<cl_long> v;
                        std::set_union(decomp->edges[a]->variables.begin(), decomp->edges[a]->variables.end(), decomp->edges[b]->variables.begin(), decomp->edges[b]->variables.end(), back_inserter(v));
                        if (v.size() < combineWidth || decomp->edges[a]->variables.size() == 0 ||
                            decomp->edges[b]->variables.size() == 0) {
                            changed = true;
                            cl_long cid = decomp->edges[b]->id;
                            decomp->edges[a]->variables.assign(v.begin(), v.end());

                            std::vector<BagType *> v_;
                            std::set_union(decomp->edges[a]->edges.begin(), decomp->edges[a]->edges.end(), decomp->edges[b]->edges.begin(), decomp->edges[b]->edges.end(), back_inserter(v_), compTreedType);
                            decomp->edges[a]->edges.assign(v_.begin(), v_.end());
                            if (b < decomp->edges.size()) {
                                decomp->edges.erase(decomp->edges.begin() + b);
                            }
                        }
                    }
                }
            }
        }

        changed = true;
        // try to merge with child nodes
        if (decomp->variables.size() < combineWidth || decomp->variables.size() == 0) {
            while (changed) {
                changed = false;
                for (long i = 0; i < decomp->edges.size(); i++) {
                    std::vector<cl_long> v;
                    std::set_union(decomp->variables.begin(), decomp->variables.end(), decomp->edges[i]->variables.begin(), decomp->edges[i]->variables.end(), back_inserter(v));
                    if (v.size() < combineWidth || decomp->variables.size() == 0 || decomp->edges[i]->variables.size() == 0) {
                        changed = true;
                        cl_long cid = decomp->edges[i]->id;
                        decomp->variables.assign(v.begin(), v.end());

                        std::vector<BagType *> v_;
                        std::set_union(decomp->edges.begin(), decomp->edges.end(), decomp->edges[i]->edges.begin(), decomp->edges[i]->edges.end(), back_inserter(v_), compTreedType);
                        decomp->edges.resize(0);
                        for (long asdf = 0, x = 0; x < v_.size(); asdf++, x++) {
                            BagType *&sdggg = v_[asdf];
                            if (v_[asdf]->id == cid) {
                                x++;
                            }
                            if (x < v_.size()) {
                                decomp->edges.push_back(v_[x]);
                            }
                        }
                    }
                }
            }
        }

        // process child nodes
        for (long i = 0; i < decomp->edges.size(); i++) {
            preprocessDecomp((decomp->edges)[i], combineWidth);
        }
        //std::sort(decomp->variables.begin(), decomp->variables.end());

        for (long i = 0; i < decomp->edges.size(); i++) {
            std::vector<cl_long> fVars;
            std::set_intersection(decomp->variables.begin(), decomp->variables.end(), decomp->edges[i]->variables.begin(), decomp->edges[i]->variables.end(), std::back_inserter(fVars));
            unsigned long long int numForgetVars = (decomp->edges[i]->variables.size() - fVars.size());
            if (numForgetVars > 8) {
                BagType *newEdge = new BagType;
                if (newEdge == NULL || errno == ENOMEM) {
                    std::cerr << "\nOut of Memory\n";
                    exit(0);
                }
                newEdge->variables.insert(newEdge->variables.end(), fVars.begin(), fVars.end());
                fVars.resize(0);
                std::set_difference(decomp->edges[i]->variables.begin(), decomp->edges[i]->variables.end(), decomp->variables.begin(), decomp->variables.end(), std::back_inserter(fVars));
                newEdge->variables.insert(newEdge->variables.end(), fVars.begin(), fVars.begin() + numForgetVars - 8);
                newEdge->edges.push_back(decomp->edges[i]);
                decomp->edges[i] = newEdge;
                std::sort(newEdge->variables.begin(), newEdge->variables.end());
            }
        }

        if (decomp->variables.size() > 61) {
            // can't solve problems with width > 60
            std::cout << "\nERROR: width > 60";
            exit(0);
        }

    }

    void Preprocessor::preprocessFacts(treedecType &decomp, satformulaType &formula, cl_double &defaultWeight) {
        for (cl_long i = 0; i < formula.facts.size(); i++) {
            cl_long fact = formula.facts[i];
            for (cl_long a = 0; a < formula.clauses.size(); a++) {
                std::vector<cl_long>::iterator elem = std::lower_bound(formula.clauses[a].begin(), formula.clauses[a].end(), fact, compVars);
                if (elem != formula.clauses[a].end()) {
                    if (*elem == (fact)) {
                        //remove clause from formula
                        formula.clauses.erase(formula.clauses.begin() + a);
                        a--;
                    } else if (*elem == (-fact)) {
                        if (formula.clauses[a].size() == 1) {
                            //found contradiction
                            formula.unsat = true;
                            return;
                        } else {
                            //erase variable from clause
                            formula.clauses[a].erase(elem);
                            if (formula.clauses[a].size() == 1 && std::find(formula.facts.begin() + i, formula.facts.end(), formula.clauses[a][0]) == formula.facts.end() && std::find(formula.facts.begin() + i, formula.facts.end(), -formula.clauses[a][0]) == formula.facts.end()) {
                                formula.facts.push_back(formula.clauses[a][0]);
                            }
                        }
                    }
                }
            }
            for (long j = i; j < formula.facts.size(); ++j) {
                if (std::abs(formula.facts[j]) > std::abs(fact)) {
                    if (formula.facts[j] > 0) {
                        formula.facts[j]--;
                    } else if (formula.facts[j] < 0) {
                        formula.facts[j]++;
                    }
                }
            }
            relableDecomp(&decomp.bags[0], std::abs(fact));
            decomp.numVars--;
            if (formula.variableWeights != nullptr) {
                //make product of removed variable weights
                if (fact < 0) {
                    defaultWeight = defaultWeight * formula.variableWeights[std::abs(fact) * 2 + 1];
                } else {
                    defaultWeight = defaultWeight * formula.variableWeights[std::abs(fact) * 2];
                }
                formula.numWeights -= 2;
                for (long j = std::abs(fact); j < formula.numVars; ++j) {
                    formula.variableWeights[j * 2] = formula.variableWeights[(j + 1) * 2];
                    formula.variableWeights[j * 2 + 1] = formula.variableWeights[(j + 1) * 2 + 1];
                }
            }
            relableFormula(formula, std::abs(fact));
            formula.numVars--;
        }
    }


    void Preprocessor::relableDecomp(BagType *decomp, cl_long id) {
        for (long i = 0; i < decomp->variables.size(); i++) {
            if (decomp->variables[i] > id) {
                decomp->variables[i]--;
            } else if (decomp->variables[i] == id) {
                decomp->variables.erase(decomp->variables.begin() + i);
                i--;
            }
        }
        for (long j = 0; j < decomp->edges.size(); ++j) {
            relableDecomp(decomp->edges[j], id);
        }
    }

    void Preprocessor::relableFormula(satformulaType &formula, cl_long id) {
        for (long i = 0; i < formula.clauses.size(); i++) {
            for (long j = 0; j < formula.clauses[i].size(); ++j) {
                if (std::abs(formula.clauses[i][j]) > id) {
                    if (formula.clauses[i][j] > 0) {
                        formula.clauses[i][j]--;
                    } else if (formula.clauses[i][j] < 0) {
                        formula.clauses[i][j]++;
                    }
                }
            }
        }
    }
}
