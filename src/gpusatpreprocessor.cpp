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
        std::vector<int64_t> variable_union;

        // maximal size of joined variables
        while (changed) {
            changed = false;

            for (auto a_it = decomp.edges.begin(); a_it != decomp.edges.end(); a_it++) {
                for (auto b_it = decomp.edges.begin(); b_it != decomp.edges.end(); b_it++) {
                    auto& edge_a = *a_it;
                    auto& edge_b = *b_it;
                    if (a_it != b_it &&
                            ((edge_a.variables.size() < combineWidth
                                && edge_b.variables.size() < combineWidth)
                             || edge_a.variables.size() == 0
                             || edge_b.variables.size() == 0
                             ) && decomp.edges.size() > 1) {

                        auto max_vars = edge_a.variables.size() + edge_b.variables.size();
                        if (max_vars > variable_union.size()) {
                            variable_union.resize(max_vars);
                        }

                        bool definitely_disjunct =
                            edge_a.variables.size() == 0
                         || edge_b.variables.size() == 0
                         || (edge_a.variables.front() > edge_b.variables.back())
                         || (edge_a.variables.back() < edge_b.variables.front());

                        auto union_size = max_vars;
                        if (!definitely_disjunct) {
                            auto union_end = std::set_union(
                                    edge_a.variables.begin(), edge_a.variables.end(),
                                    edge_b.variables.begin(), edge_b.variables.end(),
                                    variable_union.begin()
                            );

                            union_size = union_end - variable_union.begin();
                        }

                        if (union_size < combineWidth || edge_a.variables.size() == 0 ||
                            edge_b.variables.size() == 0) {

                            auto union_end = std::set_union(
                                    edge_a.variables.begin(), edge_a.variables.end(),
                                    edge_b.variables.begin(), edge_b.variables.end(),
                                    variable_union.begin()
                            );
                            changed = true;
                            edge_a.variables.assign(variable_union.begin(), union_end);

                            edge_a.edges.insert(
                                    edge_a.edges.end(),
                                    std::make_move_iterator(edge_b.edges.begin()),
                                    std::make_move_iterator(edge_b.edges.end())
                            );
                            // This is a linked list, so iterators stay valid.
                            decomp.edges.erase(b_it);
                            edge_a.edges.sort(compTreedType);
                            break;
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
                    // FIXME: clean this up
                    auto i_it = decomp.edges.begin();
                    std::advance(i_it, i);
                    auto& edge_i = *i_it;
                    std::set_union(
                            decomp.variables.begin(), decomp.variables.end(),
                            edge_i.variables.begin(), edge_i.variables.end(),
                            back_inserter(v),
                            compVars
                    );


                    if (v.size() < combineWidth || decomp.variables.size() == 0 || edge_i.variables.size() == 0) {
                        changed = true;
                        decomp.variables.assign(v.begin(), v.end());

                        decomp.edges.insert(
                                decomp.edges.end(),
                                std::make_move_iterator(edge_i.edges.begin()),
                                std::make_move_iterator(edge_i.edges.end())
                        );
                        decomp.edges.erase(i_it);
                        decomp.edges.sort(compTreedType);
                    }
                }
            }
        }

        // process child nodes
        for (auto& edge : decomp.edges) {
            preprocessDecomp(edge, combineWidth);
        }
        //std::sort(decomp.variables.begin(), decomp.variables.end());

        for (auto edge_it = decomp.edges.begin(); edge_it != decomp.edges.end(); edge_it++)  {
            std::vector<int64_t> fVars;
            auto& edge_i = *edge_it;
            std::set_intersection(
                    decomp.variables.begin(), decomp.variables.end(),
                    edge_i.variables.begin(), edge_i.variables.end(),
                    std::back_inserter(fVars),
                    compVars);
            unsigned long long int numForgetVars = (edge_i.variables.size() - fVars.size());
            if (numForgetVars > 8) {
                auto newEdge = BagType();
                newEdge.variables.insert(newEdge.variables.end(), fVars.begin(), fVars.end());
                fVars.resize(0);
                std::set_difference(
                        edge_i.variables.begin(), edge_i.variables.end(),
                        decomp.variables.begin(), decomp.variables.end(),
                        std::back_inserter(fVars),
                        compVars);
                newEdge.variables.insert(newEdge.variables.end(), fVars.begin(), fVars.begin() + numForgetVars - 8);
                std::sort(newEdge.variables.begin(), newEdge.variables.end(), compVars);
                // move existing node to be child of newEdge,
                // wich becomes decomp.edges[i]
                newEdge.edges.push_back(std::move(*edge_it));
                *edge_it = std::move(newEdge);
            }
        }

        if (decomp.variables.size() > 61) {
            // can't solve problems with width > 60
            std::cout << "\nERROR: width > 60";
            exit(0);
        }
    }


    void Preprocessor::checkNoFactInDecomp(BagType& decomp, const std::vector<int64_t>& facts) {
        for (auto fact : facts) {
            auto elem = std::lower_bound( decomp.variables.begin(), decomp.variables.end(), fact, compVars );
            assert(elem != decomp.variables.end() && "Fact in decomposition!");
        }
        for (auto& child : decomp.edges) {
            checkNoFactInDecomp(child, facts);
        }
    }

    std::vector<int64_t> Preprocessor::buildRelabelMap(int64_t facts_and_vars, const std::vector<int64_t>& facts) {
        std::vector<int64_t> relabel_map;
        relabel_map.resize(facts_and_vars + 1);

        auto current_fact = facts.begin();
        for (auto it = relabel_map.begin(); it != relabel_map.end(); it++) {
            int64_t index = it - relabel_map.begin();
            while (current_fact != facts.end() && abs(*current_fact) < index) {
                current_fact++;
            }

            int64_t facts_so_far = current_fact - facts.begin();
            // this is the fact itself
            if (abs(*current_fact) == index) {
                *it = -1;
            } else {
                *it = facts_so_far;
            }
        }
        return relabel_map;
    }

    void Preprocessor::relabelFormula(satformulaType &formula) {

        std::sort(formula.facts.begin(), formula.facts.end(), compVars);
        auto relabel_map = buildRelabelMap(formula.facts.size() + formula.numVars, formula.facts);

        for (auto it=formula.clause_bag.begin(); it!=formula.clause_bag.end(); it++) {
            // no facts here
            assert(relabel_map[abs(*it)] > -1);

            if (*it < 0) {
                *it += relabel_map[abs(*it)];
            } else {
                *it -= relabel_map[*it];
            }
        }
        /*
        std::cout << "p cnf " << formula.numVars << " " << formula.clause_offsets.size() << std::endl;
        for (auto lit : formula.facts) {
            std::cout << " " << lit << " " << 0 << std::endl;
        }
        for (auto lit : formula.clause_bag) {
            std::cout << " " << lit;
            if (lit == 0) {
                std::cout << std::endl;
            }
        }
        */
    }

    void Preprocessor::preprocessFacts(satformulaType &formula, double &defaultWeight) {
        std::vector<int64_t> intersection;
        std::vector<int64_t> new_facts;
        std::vector<int64_t> sorted_clausevars = formula.clause_bag;
        std::sort(sorted_clausevars.begin(), sorted_clausevars.end(), compVars);
        std::set_intersection(
            formula.facts.begin(),
            formula.facts.end(),
            sorted_clausevars.begin(),
            sorted_clausevars.end(),
            back_inserter(intersection),
            compVars
        );
        if (intersection.size() == 0) {
            std::cerr << "preprocessing unnecessary." << std::endl;
            return;
        }

        /*
        for (auto fact : formula.facts) {
            std::cout << " " << fact;
        }
        */
        std::cout << std::endl;

        std::cerr << "WARNING: fact pp was re-implemented, WMC untested!" << std::endl;
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
                    if (current_clause_lits == 1) {
                        new_facts.push_back(new_clause_bag.back());
                        new_clause_bag.pop_back();
                    } else {
                        new_clause_bag.push_back(0);
                        new_offsets.push_back(new_clause_bag.size() - current_clause_lits - 1);
                    }
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

        std::cout << "old clauses: " << formula.clause_offsets.size() << " new clauses: " << new_offsets.size() << std::endl;
        formula.clause_bag = std::move(new_clause_bag);
        formula.clause_offsets = std::move(new_offsets);

        if (!new_facts.empty()) {
            for (auto fact : new_facts) {
                auto it = std::lower_bound( formula.facts.begin(), formula.facts.end(), fact, compVars );
                if (abs(*it) != abs(fact)) {
                    formula.facts.insert( it, fact );
                    formula.numVars--;
                } else if (*it == -fact) {
                    formula.unsat = true;
                    return;
                }
            }
            preprocessFacts(formula, defaultWeight);
        }
    }
}
