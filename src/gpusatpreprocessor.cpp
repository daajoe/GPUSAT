#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "gpusatparser.h"
#include "gpusatpreprocessor.h"

namespace gpusat {
    void Preprocessor::preprocessDecomp(BagType& decomp, size_t combineWidth) {

        bool changed = true;
        // try to merge child nodes
        while (changed) {
            changed = false;
            for (size_t a = 0; a < decomp.edges.size() && !changed; a++) {
                for (size_t b = 0; b < decomp.edges.size() && !changed; b++) {
                    auto& edge_a = decomp.edges[a];
                    auto& edge_b = decomp.edges[b];
                    if (a != b &&
                            ((edge_a.variables.size() < combineWidth
                                && edge_b.variables.size() < combineWidth)
                             || edge_a.variables.size() == 0
                             || edge_b.variables.size() == 0
                             ) && decomp.edges.size() > 1) {

                        std::vector<int64_t> v;
                        std::set_union(
                                edge_a.variables.begin(), edge_a.variables.end(),
                                edge_b.variables.begin(), edge_b.variables.end(),
                                back_inserter(v)
                        );
                        if (v.size() < combineWidth || edge_a.variables.size() == 0 ||
                            edge_b.variables.size() == 0) {
                            changed = true;
                            edge_a.variables.assign(v.begin(), v.end());

                            edge_a.edges.insert(
                                    edge_a.edges.end(),
                                    std::make_move_iterator(edge_b.edges.begin()),
                                    std::make_move_iterator(edge_b.edges.end())
                            );
                            decomp.edges.erase(decomp.edges.begin() + b);
                            sort(edge_a.edges.begin(), edge_a.edges.end(), compTreedType);
                        }
                    }
                }
            }
        }

        changed = true;
        // try to merge with child nodes
        if (decomp.variables.size() < combineWidth || decomp.variables.size() == 0) {
            while (changed) {
                changed = false;
                for (size_t i = 0; i < decomp.edges.size(); i++) {
                    std::vector<int64_t> v;
                    auto& edge_i = decomp.edges[i];
                    std::set_union(
                            decomp.variables.begin(), decomp.variables.end(),
                            edge_i.variables.begin(), edge_i.variables.end(),
                            back_inserter(v)
                    );

                    if (v.size() < combineWidth || decomp.variables.size() == 0 || edge_i.variables.size() == 0) {
                        changed = true;
                        decomp.variables.assign(v.begin(), v.end());

                        decomp.edges.insert(
                                decomp.edges.end(),
                                std::make_move_iterator(edge_i.edges.begin()),
                                std::make_move_iterator(edge_i.edges.end())
                        );
                        decomp.edges.erase(decomp.edges.begin() + i);
                        std::sort(decomp.edges.begin(), decomp.edges.end(), compTreedType);
                    }
                }
            }
        }

        // process child nodes
        for (size_t i = 0; i < decomp.edges.size(); i++) {
            preprocessDecomp((decomp.edges)[i], combineWidth);
        }
        //std::sort(decomp.variables.begin(), decomp.variables.end());

        for (size_t i = 0; i < decomp.edges.size(); i++) {
            std::vector<int64_t> fVars;
            auto& edge_i = decomp.edges[i];
            std::set_intersection(
                    decomp.variables.begin(), decomp.variables.end(),
                    edge_i.variables.begin(), edge_i.variables.end(),
                    std::back_inserter(fVars));
            unsigned long long int numForgetVars = (edge_i.variables.size() - fVars.size());
            if (numForgetVars > 8) {
                auto newEdge = BagType();
                newEdge.variables.insert(newEdge.variables.end(), fVars.begin(), fVars.end());
                fVars.resize(0);
                std::set_difference(
                        edge_i.variables.begin(), edge_i.variables.end(),
                        decomp.variables.begin(), decomp.variables.end(),
                        std::back_inserter(fVars));
                newEdge.variables.insert(newEdge.variables.end(), fVars.begin(), fVars.begin() + numForgetVars - 8);
                std::sort(newEdge.variables.begin(), newEdge.variables.end());
                // move existing node to be child of newEdge,
                // wich becomes decomp.edges[i]
                newEdge.edges.push_back(std::move(decomp.edges[i]));
                decomp.edges[i] = std::move(newEdge);
            }
        }

        if (decomp.variables.size() > 61) {
            // can't solve problems with width > 60
            std::cout << "\nERROR: width > 60";
            exit(0);
        }

    }

    void Preprocessor::preprocessFacts(treedecType &decomp, satformulaType &formula, double &defaultWeight) {
        std::vector<int64_t> intersection;
        std::set_intersection(
            formula.facts.begin(),
            formula.facts.end(),
            formula.clause_bag.begin(),
            formula.clause_bag.end(),
            back_inserter(intersection),
            compVars
        );
        if (intersection.size() == 0) {
            std::cerr << "preprocessing unnecessary." << std::endl;
            return;
        }

        std::cerr << "WARNING: fact pp was re-implemented!" << std::endl;
        std::vector<int64_t> new_clause_bag;
        std::vector<size_t> new_offsets;

        if (formula.variableWeights != nullptr) {
            for (auto fact : formula.facts) {
                if (fact < 0) {
                    defaultWeight *= formula.variableWeights[std::abs(fact) * 2 + 1];
                } else {
                    defaultWeight *= formula.variableWeights[std::abs(fact) * 2];
                }
            }
        }

        bool facts_changed = false;

        size_t old_clause_idx = 0;
        size_t old_clause_lits = 0;
        size_t current_clause_lits = 0;
        bool skip_clause = false;
        for (auto lit : formula.clause_bag) {
            if (lit == 0) {
                // non-empty clause
                if (current_clause_lits > 0) {
                    new_clause_bag.push_back(0);
                    new_offsets.push_back(new_clause_bag.size() - current_clause_lits - 1);
                }
                old_clause_lits = 0;
                current_clause_lits = 0;
                skip_clause = false;
                old_clause_idx++;
                continue;
            }

            if (skip_clause) {
                continue;
            }

            old_clause_lits++;

            bool skip_lit = false;
            for (auto fact : intersection) {
                // there is a fact describing this literal
                if (abs(lit) == abs(fact)) {
                    auto size = clause_size(formula, old_clause_idx);
                    size_t remaining = size - old_clause_lits;
                    skip_lit = true;

                    // fact solves clause
                    if (lit == fact) {
                        skip_clause = true;
                        new_clause_bag.erase(new_clause_bag.end() - current_clause_lits , new_clause_bag.end());
                        current_clause_lits = 0;

                    // fact does not solve clause
                    } else if (lit != fact) {
                        // this is a contradiction
                        if (current_clause_lits == 0 && remaining == 0) {
                            formula.unsat = true;
                            return;
                        }
                        // the other literal must be true
                        // } else if (current_clause_lits == 1 && remaining == 0) {
                        //     formula.facts.push_back(new_clause_bag.back());
                        //     facts_changed = true;
                        //     new_clause_bag.pop_back();
                        //     skip_lit = true;
                        // literal cannot satisfy clause -> leave out
                    }
                    break;
                }
            }

            if (!skip_lit) {
                new_clause_bag.push_back(lit);
                current_clause_lits++;
            }
        }

        formula.clause_bag = std::move(new_clause_bag);
        formula.clause_offsets = std::move(new_offsets);

            /*
        for (size_t i = 0; i < formula.facts.size(); i++) {
            int64_t fact = formula.facts[i];
            for (size_t a = 0; a < formula.clauses.size(); a++) {
                std::vector<int64_t>::iterator elem = std::lower_bound(formula.clauses[a].begin(), formula.clauses[a].end(), fact, compVars);
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
            // FIXME: Relabel unnötig.
            for (size_t j = i; j < formula.facts.size(); ++j) {
                if (std::abs(formula.facts[j]) > std::abs(fact)) {
                    if (formula.facts[j] > 0) {
                        formula.facts[j]--;
                    } else if (formula.facts[j] < 0) {
                        formula.facts[j]++;
                    }
                }
            }
            relabelDecomp(decomp.root, std::abs(fact));
            decomp.numVars--;
            if (formula.variableWeights != nullptr) {
                //make product of removed variable weights
                if (fact < 0) {
                    defaultWeight = defaultWeight * formula.variableWeights[std::abs(fact) * 2 + 1];
                } else {
                    defaultWeight = defaultWeight * formula.variableWeights[std::abs(fact) * 2];
                }
                formula.numWeights -= 2;
                for (size_t j = std::abs(fact); j < formula.numVars; ++j) {
                    formula.variableWeights[j * 2] = formula.variableWeights[(j + 1) * 2];
                    formula.variableWeights[j * 2 + 1] = formula.variableWeights[(j + 1) * 2 + 1];
                }
            }
            relabelFormula(formula, std::abs(fact));
            formula.numVars--;
        }
        */
    }


    void Preprocessor::relabelDecomp(BagType& decomp, int64_t id) {
        for (size_t i = 0; i < decomp.variables.size(); i++) {
            if (decomp.variables[i] > id) {
                //decomp.variables[i]--;
            } else if (decomp.variables[i] == id) {
                decomp.variables.erase(decomp.variables.begin() + i);
                i--;
            }
        }
        for (size_t j = 0; j < decomp.edges.size(); ++j) {
            relabelDecomp(decomp.edges[j], id);
        }
    }

    void Preprocessor::relabelFormula(satformulaType &formula, int64_t id) {
        /*
        for (size_t i = 0; i < formula.clauses.size(); i++) {
            for (size_t j = 0; j < formula.clauses[i].size(); ++j) {
                if (std::abs(formula.clauses[i][j]) > id) {
                    if (formula.clauses[i][j] > 0) {
                        formula.clauses[i][j]--;
                    } else if (formula.clauses[i][j] < 0) {
                        formula.clauses[i][j]++;
                    }
                }
            }
        }
        */
    }
}
