#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <gpusatparser.h>
#include <gpusatpreprocessor.h>

namespace gpusat {
    void Preprocessor::preprocessDecomp(bagType *decomp, cl_long combineWidth) {

        bool changed = true;
        // try to merge child nodes
        while (changed) {
            changed = false;
            for (int a = 0; a < decomp->edges.size() && !changed; a++) {
                for (int b = 0; b < decomp->edges.size() && !changed; b++) {
                    if (a != b && ((decomp->edges[a]->variables.size() < combineWidth &&
                                    decomp->edges[b]->variables.size() < combineWidth) || decomp->edges[a]->variables.size() == 0 ||
                                   decomp->edges[b]->variables.size() == 0) && decomp->edges.size() > 1) {
                        std::vector<cl_long> v(static_cast<unsigned long long int>(decomp->edges[a]->variables.size() + decomp->edges[b]->variables.size()));
                        std::vector<cl_long>::iterator it;
                        it = std::set_union(decomp->edges[a]->variables.begin(), decomp->edges[a]->variables.end(), decomp->edges[b]->variables.begin(),
                                            decomp->edges[b]->variables.end(), v.begin());
                        v.resize(static_cast<unsigned long long int>(it - v.begin()));
                        if (v.size() < combineWidth || decomp->edges[a]->variables.size() == 0 ||
                            decomp->edges[b]->variables.size() == 0) {
                            changed = true;
                            cl_long cid = decomp->edges[b]->id;
                            decomp->edges[a]->variables.assign(v.begin(), v.end());

                            std::vector<bagType *> v_(static_cast<unsigned long long int>(decomp->edges[a]->edges.size() + decomp->edges[b]->edges.size()));
                            std::vector<bagType *>::iterator it_;
                            it_ = std::set_union(decomp->edges[a]->edges.begin(), decomp->edges[a]->edges.end(), decomp->edges[b]->edges.begin(),
                                                 decomp->edges[b]->edges.end(), v_.begin(), compTreedType);
                            v_.resize(static_cast<unsigned long long int>(it_ - v_.begin()));
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
                for (int i = 0; i < decomp->edges.size(); i++) {
                    std::vector<cl_long> v(static_cast<unsigned long long int>(decomp->variables.size() + decomp->edges[i]->variables.size()));
                    std::vector<cl_long>::iterator it;
                    it = std::set_union(decomp->variables.begin(), decomp->variables.end(), decomp->edges[i]->variables.begin(), decomp->edges[i]->variables.end(),
                                        v.begin());
                    v.resize(static_cast<unsigned long long int>(it - v.begin()));
                    if (v.size() < combineWidth || decomp->variables.size() == 0 || decomp->edges[i]->variables.size() == 0) {
                        changed = true;
                        cl_long cid = decomp->edges[i]->id;
                        decomp->variables.assign(v.begin(), v.end());

                        std::vector<bagType *> v_(static_cast<unsigned long long int>(decomp->edges.size() + decomp->edges[i]->edges.size()));
                        std::vector<bagType *>::iterator it_;
                        it_ = std::set_union(decomp->edges.begin(), decomp->edges.end(), decomp->edges[i]->edges.begin(), decomp->edges[i]->edges.end(), v_.begin(),
                                             compTreedType);
                        v_.resize(static_cast<unsigned long long int>(it_ - v_.begin()));
                        decomp->edges.resize(0);
                        for (int asdf = 0, x = 0; x < v_.size(); asdf++, x++) {
                            bagType *&sdggg = v_[asdf];
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
        for (int i = 0; i < decomp->edges.size(); i++) {
            preprocessDecomp((decomp->edges)[i], combineWidth);
        }
        //std::sort(decomp->variables.begin(), decomp->variables.end());

        for (int i = 0; i < decomp->edges.size(); i++) {
            std::vector<cl_long> fVars;
            std::set_intersection(decomp->variables.begin(), decomp->variables.end(), decomp->edges[i]->variables.begin(), decomp->edges[i]->variables.end(),
                                  std::back_inserter(fVars));
            unsigned long long int numForgetVars = (decomp->edges[i]->variables.size() - fVars.size());
            if (numForgetVars > 4) {
                bagType *newEdge = new bagType;
                newEdge->variables.insert(newEdge->variables.end(), fVars.begin(), fVars.end());
                fVars.resize(0);
                std::set_difference(decomp->edges[i]->variables.begin(), decomp->edges[i]->variables.end(), decomp->variables.begin(), decomp->variables.end(),
                                    std::back_inserter(fVars));
                newEdge->variables.insert(newEdge->variables.end(), fVars.begin(), fVars.begin() + numForgetVars - 4);
                newEdge->edges.push_back(decomp->edges[i]);
                decomp->edges[i] = newEdge;
                std::sort(newEdge->variables.begin(), newEdge->variables.end());
            }
        }

        if (decomp->variables.size() > 61) {
            std::cout << "ERROR: width > 60";
            exit(0);
        }

    }

    void Preprocessor::preprocessFacts(treedecType &decomp, satformulaType &formula, graphTypes gType, cl_double &defaultWeight) {
        for (cl_long i = 0; i < formula.facts.size(); i++) {
            cl_long fact = formula.facts[i];
            for (cl_long a = 0; a < formula.clauses.size(); a++) {
                std::vector<cl_long>::iterator elem = std::lower_bound(formula.clauses[a].begin(), formula.clauses[a].end(), fact, compVars);
                if (elem != formula.clauses[a].end()) {
                    if (*elem == (fact)) {
                        //remove clause from formula
                        formula.clauses.erase(formula.clauses.begin() + a);
                        if (gType == INCIDENCE) {
                            relableDecomp(&decomp.bags[0], a + formula.numVars + 1);
                            decomp.numVars--;
                        } else if (gType == DUAL) {
                            relableDecomp(&decomp.bags[0], a);
                            decomp.numVars--;
                        }
                        a--;
                    } else if (*elem == (-fact)) {
                        if (formula.clauses[a].size() == 1) {
                            //found contradiction
                            formula.unsat = true;
                            return;
                        } else {
                            //erase variable from clause
                            formula.clauses[a].erase(elem);
                            if (formula.clauses[a].size() == 1 &&
                                std::find(formula.facts.begin() + i, formula.facts.end(), formula.clauses[a][0]) == formula.facts.end() &&
                                std::find(formula.facts.begin() + i, formula.facts.end(), -formula.clauses[a][0]) == formula.facts.end()) {
                                formula.facts.push_back(formula.clauses[a][0]);
                            }
                        }
                    }
                }
            }
            for (int j = i; j < formula.facts.size(); ++j) {
                if (std::abs(formula.facts[j]) > std::abs(fact)) {
                    if (formula.facts[j] > 0) {
                        formula.facts[j]--;
                    } else if (formula.facts[j] < 0) {
                        formula.facts[j]++;
                    }
                }
            }
            if (gType != DUAL) {
                relableDecomp(&decomp.bags[0], std::abs(fact));
            }
            decomp.numVars--;
            if (formula.variableWeights != nullptr) {
                //make product of removed variable weights
                if (fact < 0) {
                    defaultWeight = defaultWeight * formula.variableWeights[std::abs(fact) * 2 + 1];
                } else {
                    defaultWeight = defaultWeight * formula.variableWeights[std::abs(fact) * 2];
                }
                formula.numWeights -= 2;
                for (int j = std::abs(fact); j < formula.numVars; ++j) {
                    formula.variableWeights[j * 2] = formula.variableWeights[(j + 1) * 2];
                    formula.variableWeights[j * 2 + 1] = formula.variableWeights[(j + 1) * 2 + 1];
                }
            }
            relableFormula(formula, std::abs(fact));
            formula.numVars--;
        }
    }


    void Preprocessor::relableDecomp(bagType *decomp, cl_long id) {
        for (int i = 0; i < decomp->variables.size(); i++) {
            if (decomp->variables[i] > id) {
                decomp->variables[i]--;
            } else if (decomp->variables[i] == id) {
                decomp->variables.erase(decomp->variables.begin() + i);
                i--;
            }
        }
        for (int j = 0; j < decomp->edges.size(); ++j) {
            relableDecomp(decomp->edges[j], id);
        }
    }

    void Preprocessor::relableFormula(satformulaType &formula, cl_long id) {
        for (int i = 0; i < formula.clauses.size(); i++) {
            for (int j = 0; j < formula.clauses[i].size(); ++j) {
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