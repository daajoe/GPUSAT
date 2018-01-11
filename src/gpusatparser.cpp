#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <gpusatparser.h>
#include <d4_utils.h>
#include <gpusautils.h>
#include <unordered_map>

namespace gpusat {

    satformulaType CNFParser::parseSatFormula(std::string formula) {
        satformulaType ret = satformulaType();
        std::stringstream ss(formula);
        std::string item;
        std::unordered_map<cl_long, solType> weights;
        std::vector<cl_long> *clause = new std::vector<cl_long>();
        while (getline(ss, item)) {
            //ignore empty line
            if (item.length() > 0) {
                char type = item.at(0);
                if (type == 'c' || type == '%') {
                    //comment line (ignore)
                } else if (type == 'p') {
                    //start line
                    parseProblemLine(ret, item);
                } else if (type == 'w') {
                    //weight line
                    this->parseWeightLine(item, weights);
                } else {
                    //clause line
                    std::stringstream sline(item);
                    std::string i;

                    cl_long num = 0;
                    while (!sline.eof()) {
                        getline(sline, i, ' ');
                        if (i.size() > 0) {
                            num = std::stol(i);
                            if (num != 0) {
                                clause->push_back(num);
                            }
                        }
                    }
                    if (clause->size() > 1 && num == 0) {
                        std::sort(clause->begin(), clause->end(), compVars);
                        ret.clauses.push_back(*clause);
                        clause->clear();
                    } else if (clause->size() == 1 && num == 0) {
                        if (std::find(ret.facts.begin(), ret.facts.end(), (*clause)[0]) == ret.facts.end())
                            ret.facts.push_back((*clause)[0]);
                        ret.clauses.push_back(*clause);
                        clause->clear();
                    }
                }
            }
        }

        if (wmc) {
            ret.variableWeights = new solType[(ret.numVars + 1) * 2]();
            ret.numWeights = (ret.numVars + 1) * 2;

            for (cl_long i = 0; i <= ret.numVars; i++) {
                std::unordered_map<cl_long, solType>::const_iterator elem = weights.find(i);
                if (elem != weights.end()) {
                    solType we = weights[i];
                    if (we < 0) {
                        ret.variableWeights[i * 2] = 1;
                        ret.variableWeights[i * 2 + 1] = 1;
                    } else {
                        ret.variableWeights[i * 2] = weights[i];
                        ret.variableWeights[i * 2 + 1] = 1 - weights[i];
                    }
                } else {
                    ret.variableWeights[i * 2] = 1;
                    ret.variableWeights[i * 2 + 1] = 1;
                }
            }
        } else {
            ret.variableWeights = nullptr;
        }
        return ret;
    }

    void CNFParser::parseProblemLine(satformulaType &satformula, std::string item) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //p
        while (i.size() == 0) getline(sline, i, ' ');
        getline(sline, i, ' '); //cnf
        while (i.size() == 0) getline(sline, i, ' ');
        getline(sline, i, ' '); //num vars
        while (i.size() == 0) getline(sline, i, ' ');
        satformula.numVars = stoi(i);
        getline(sline, i, ' '); //num clauses
        while (i.size() == 0) getline(sline, i, ' ');
    }

    void CNFParser::parseWeightLine(std::string item, std::unordered_map<cl_long, solType> &weights) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //w
        getline(sline, i, ' '); //variable
        cl_long id = stol(i);
        getline(sline, i, ' '); //weight
        solType val = stod(i);
        weights[id] = (id < 0) ? -val : val;
    }

    CNFParser::CNFParser(bool weighted) {
        wmc = weighted;
    }

    TDParser::TDParser(int i, bool b) {
        combineWidth = i;
        factR = b;
    }

    treedecType TDParser::parseTreeDecomp(std::string graph, satformulaType &formula) {
        preetreedecType ret = preetreedecType();
        std::stringstream ss(graph);
        std::string item;
        std::vector<std::vector<cl_long>> edges;
        while (getline(ss, item)) {
            //ignore empty line
            if (item.length() > 2) {
                char type = item.at(0);
                if (type == 'c') {
                    //comment line (ignore)
                } else if (type == 's') {
                    //start line
                    parseStartLine(ret, item, edges);
                } else if (type == 'b') {
                    //bag line
                    parseBagLine(ret, item);
                } else {
                    //edge line
                    parseEdgeLine(item, edges);
                }
            }
        }

        if (!edges.empty()) {
            for (int a = 0; a < edges.size(); a++) {
                std::sort(edges[a].begin(), edges[a].end());
            }
            removeEdges(edges, 0, 0);
        }

        if (!edges.empty()) {
            for (int a = 0; a < ret.numb; a++) {
                int b = 0;
                while (!edges[a].empty()) {
                    ret.bags[a].edges.push_back(&ret.bags[edges[a].back() - 1]);
                    edges[a].pop_back();
                    b++;
                }
                std::sort(ret.bags[a].edges.begin(), ret.bags[a].edges.end(), compTreedType);
            }
        }

        // remove facts form decomp and formula
        if (factR) {
            preprocessFacts(ret, formula);
            if (formula.unsat) {
                return treedecType();
            }
        }
        // combine small bags
        preprocessDecomp(&ret.bags[0]);
        treedecType ret_ = treedecType();
        ret_.numb = 0;
        std::list<preebagType *> bags;
        bags.push_back(&(ret.bags[0]));
        while (!bags.empty()) {
            preebagType *bag = bags.front();
            bags.pop_front();
            for (int a = 0; a < bag->edges.size(); a++) {
                bags.push_back(bag->edges[a]);
            }
            ret_.numb++;
        }
        ret_.bags = new bagType[ret_.numb];
        cl_long id = 0, cid = 2;
        bags.push_back(&ret.bags[0]);
        while (!bags.empty()) {
            preebagType *bag = bags.front();
            bags.pop_front();
            ret_.bags[id].variables = (*bag).variables;
            ret_.bags[id].numSol = static_cast<cl_long>(pow(2, (cl_long) ret_.bags[id].variables.size()));
            for (int a = 0; a < bag->edges.size(); a++) {
                ret_.bags[id].edges.push_back(cid);
                bags.push_back(bag->edges[a]);
                cid++;
            }
            std::sort(ret_.bags[id].edges.begin(), ret_.bags[id].edges.end());
            id++;
        }
        ret_.numVars = ret.numVars;
        return ret_;
    }

    void TDParser::removeEdges(std::vector<std::vector<cl_long>> &node, cl_long id, cl_long preID) {
        for (int b = 0; b < node[id].size(); b++) {
            if (preID != (node[id][b] - 1)) {
                removeEdges(node, node[id][b] - 1, id);
            }
        }
        std::vector<cl_long>::iterator it = std::find(node[id].begin(), node[id].end(), preID + 1);
        if (it != node[id].end()) {
            node[id].erase(it);
        }
    }

    void TDParser::parseEdgeLine(std::string item, std::vector<std::vector<cl_long>> &edges) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //start
        cl_long start = stoi(i);
        getline(sline, i, ' '); //end
        cl_long end = stoi(i);
        edges[start - 1].push_back(end);
        edges[end - 1].push_back(start);
    }

    void TDParser::parseStartLine(preetreedecType &ret, std::string &item, std::vector<std::vector<cl_long>> &edges) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //s
        getline(sline, i, ' '); //td
        getline(sline, i, ' '); //num bags
        ret.bags = new preebagType[stoi(i)];
        ret.numb = stoi(i);
        for (int a = 0; a < ret.numb; a++) {
            std::vector<cl_long> edge;
            edges.push_back(edge);
        }
        getline(sline, i, ' '); //width
        getline(sline, i, ' '); //num vars
        ret.numVars = stoi(i);
    }

    void TDParser::parseBagLine(preetreedecType &ret, std::string item) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //b
        getline(sline, i, ' '); //bag number
        int bnum = stoi(i);
        int a = 0;
        cl_long match_count = 0;
        std::istringstream ss(item);
        std::string word;
        while (ss >> word) {
            std::istringstream maybenumber(word);
            int number = 0;
            if (maybenumber >> number) {
                match_count++;
            }
        }
        ret.bags[bnum - 1].id = bnum - 1;
        while (getline(sline, i, ' ')) //vertices
        {
            if (i[0] != '\r') {
                ret.bags[bnum - 1].variables.push_back(stoi(i));
                a++;
            }
        }
        std::sort(ret.bags[bnum - 1].variables.begin(), ret.bags[bnum - 1].variables.end());
    }

    void TDParser::preprocessDecomp(preebagType *decomp) {

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

                            std::vector<preebagType *> v_(static_cast<unsigned long long int>(decomp->edges[a]->edges.size() + decomp->edges[b]->edges.size()));
                            std::vector<preebagType *>::iterator it_;
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

                        std::vector<preebagType *> v_(static_cast<unsigned long long int>(decomp->edges.size() + decomp->edges[i]->edges.size()));
                        std::vector<preebagType *>::iterator it_;
                        it_ = std::set_union(decomp->edges.begin(), decomp->edges.end(), decomp->edges[i]->edges.begin(), decomp->edges[i]->edges.end(), v_.begin(),
                                             compTreedType);
                        v_.resize(static_cast<unsigned long long int>(it_ - v_.begin()));
                        decomp->edges.clear();
                        for (int asdf = 0, x = 0; x < v_.size(); asdf++, x++) {
                            preebagType *&sdggg = v_[asdf];
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
            preprocessDecomp((decomp->edges)[i]);
        }
    }

    void TDParser::preprocessFacts(preetreedecType &decomp, satformulaType &formula) {
        for (cl_long i = 0; i < formula.facts.size(); i++) {
            cl_long fact = formula.facts[i];
            for (cl_long a = 0; a < formula.clauses.size(); a++) {
                std::vector<cl_long>::iterator elem = std::lower_bound(formula.clauses[a].begin(), formula.clauses[a].end(), fact, compVars);
                if (elem != formula.clauses[a].end()) {
                    if (*elem == (fact)) {
                        formula.clauses.erase(formula.clauses.begin() + a);
                        if (decomp.numVars > formula.numVars) {
                            relableDecomp(&decomp.bags[0], a + formula.numVars + 1);
                            decomp.numVars--;
                        }
                        a--;
                    } else if (*elem == (-fact)) {
                        if (formula.clauses[a].size() == 1) {
                            formula.unsat = true;
                            return;
                        } else {
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
            relableDecomp(&decomp.bags[0], std::abs(fact));
            decomp.numVars--;
            if (formula.variableWeights != nullptr) {
                if (fact < 0) {
                    defaultWeight = defaultWeight * formula.variableWeights[std::abs(fact) * 2 + 1];
                } else {
                    defaultWeight = defaultWeight * formula.variableWeights[std::abs(fact) * 2];
                }
                formula.numWeights-=2;
                for (int j = std::abs(fact); j < formula.numVars; ++j) {
                    formula.variableWeights[j * 2] = formula.variableWeights[(j + 1) * 2];
                    formula.variableWeights[j * 2 + 1] = formula.variableWeights[(j + 1) * 2 + 1];
                }
            }
            relableFormula(formula, std::abs(fact));
            formula.numVars--;
        }
    }


    void TDParser::relableDecomp(preebagType *decomp, cl_long id) {
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


    void TDParser::relableFormula(satformulaType &formula, cl_long id) {
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