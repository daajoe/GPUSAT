#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <gpusatparser.h>

namespace gpusat {

    satformulaType CNFParser::parseSatFormula(std::string formula) {
        satformulaType ret = satformulaType();
        std::stringstream ss(formula);
        std::string item;
        std::queue<std::queue<cl_long  >> clauses;
        cl_long clauseSize = 0;
        while (getline(ss, item)) {
            //ignore empty line
            if (item.length() > 0) {
                char type = item.at(0);
                if (type == 'c') {
                    //comment line (ignore)
                } else if (type == 'p') {
                    //start line
                    parseProblemLine(ret, item, clauses);
                } else {
                    //clause line
                    parseClauseLine(item, clauses, clauseSize);
                }
            }
        }

        if (!clauses.empty()) {
            int a = 0, s = 0;
            ret.totalNumVar = clauseSize;
            ret.clauses = new cl_long[clauseSize]();
            while (!clauses.empty()) {
                std::queue<cl_long> &clause = clauses.front();
                ret.numVarsC[a] = clause.size();

                int b = 0;
                while (!clause.empty()) {
                    ret.clauses[s + b] = clause.front();
                    clause.pop();
                    b++;
                }
                std::sort(&ret.clauses[s], &ret.clauses[s + b]);
                clauses.pop();
                s += ret.numVarsC[a];
                a++;
            }
        }
        return ret;
    }

    void CNFParser::parseClauseLine(std::string item, std::queue<std::queue<cl_long>> &clauses, cl_long &clauseSize) {
        std::stringstream sline(item);
        std::string i;
        std::queue<cl_long> clause;
        getline(sline, i, ' ');
        cl_long match_count = 1;
        std::istringstream ss(item);
        std::string word;
        while (ss >> word) {
            std::istringstream maybenumber(word);
            int number = 0;
            if (maybenumber >> number) {
                match_count++;
            }
        }
        clauseSize += match_count - 1;
        while (!sline.eof()) {
            if (!i.empty()) {
                if (stoi(i) == 0) {
                    break;
                }
                clause.push(stoi(i));
            }
            getline(sline, i, ' ');
        }
        clauses.push(clause);
    }

    void CNFParser::parseProblemLine(satformulaType &satformula, std::string item, std::queue<std::queue<cl_long>> &clauses) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //p
        getline(sline, i, ' '); //cnf
        getline(sline, i, ' '); //num vars
        satformula.numVars = stoi(i);
        getline(sline, i, ' '); //num clauses
        satformula.numclauses = stoi(i);
        satformula.numVarsC = new cl_long[satformula.numclauses]();
    }

    TDParser::TDParser(int i) {
        combineWidth = i;
    }

    treedecType TDParser::parseTreeDecomp(std::string graph) {
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
            for (int a = 0; a < ret.numb; a++) {
                ret.bags[a].edges = new preebagType *[edges[a].size()]();
                int b = 0;
                while (!edges[a].empty()) {
                    ret.bags[a].edges[b] = &ret.bags[edges[a].back() - 1];
                    edges[a].pop_back();
                    b++;
                }
                ret.bags[a].numEdges = b;
                std::sort(ret.bags[a].edges, ret.bags[a].edges + ret.bags[a].numEdges, compTreedType);
            }
        }

        preprocessDecomp(&ret.bags[0]);
        treedecType ret_ = treedecType();
        ret_.numb = 0;
        std::list<preebagType *> bags;
        bags.push_back(&ret.bags[0]);
        while (!bags.empty()) {
            preebagType *bag = bags.front();
            bags.pop_front();
            for (int a = 0; a < bag->numEdges; a++) {
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
            ret_.bags[id].numVars = bag->numVariables;
            ret_.bags[id].variables = new cl_long[ret_.bags[id].numVars];
            std::copy(bag->variables, bag->variables + bag->numVariables, ret_.bags[id].variables);
            ret_.bags[id].numEdges = bag->numEdges;
            ret_.bags[id].edges = new cl_long[ret_.bags[id].numEdges];
            ret_.bags[id].numSol = static_cast<cl_long>(pow(2, ret_.bags[id].numVars));
            for (int a = 0; a < bag->numEdges; a++) {
                ret_.bags[id].edges[a] = cid;
                bags.push_back(bag->edges[a]);
                cid++;
            }
            std::sort(ret_.bags[id].edges, ret_.bags[id].edges + ret_.bags[id].numEdges);
            id++;
        }
        for (int a = 0; a < ret.numb; a++) {
            delete[] ret.bags[a].edges;
            delete[] ret.bags[a].variables;
        }
        delete[] ret.bags;
        return ret_;
    }

    void TDParser::parseEdgeLine(std::string item, std::vector<std::vector<cl_long>> &edges) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //start
        cl_long start = stoi(i);
        getline(sline, i, ' '); //end
        cl_long end = stoi(i);
        edges[start - 1].push_back(end);
    }

    void TDParser::parseStartLine(preetreedecType &ret, std::string &item, std::vector<std::vector<cl_long>> &edges) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //s
        getline(sline, i, ' '); //tw
        getline(sline, i, ' '); //num bags
        ret.bags = new preebagType[stoi(i)];
        ret.numb = stoi(i);
        for (int a = 0; a < ret.numb; a++) {
            std::vector<cl_long> edge;
            edges.push_back(edge);
        }
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
        ret.bags[bnum - 1].variables = new cl_long[match_count - 1]();
        ret.bags[bnum - 1].numVariables = match_count - 1;
        while (getline(sline, i, ' ')) //vertices
        {
            ret.bags[bnum - 1].variables[a] = stoi(i);
            a++;
        }
        std::sort(ret.bags[bnum - 1].variables, &ret.bags[bnum - 1].variables[ret.bags[bnum - 1].numVariables]);
    }

    void TDParser::preprocessDecomp(preebagType *decomp) {

        bool changed = true;
        // try to merge child nodes
        while (changed) {
            changed = false;
            for (int a = 0; a < decomp->numEdges && !changed; a++) {
                for (int b = 0; b < decomp->numEdges && !changed; b++) {
                    if (a != b && decomp->edges[a]->numVariables < combineWidth &&
                        decomp->edges[b]->numVariables < combineWidth) {
                        std::vector<cl_long> v(static_cast<unsigned long long int>(decomp->edges[a]->numVariables + decomp->edges[b]->numVariables));
                        std::vector<cl_long>::iterator it;
                        it = std::set_union(decomp->edges[a]->variables, decomp->edges[a]->variables + decomp->edges[a]->numVariables,
                                            decomp->edges[b]->variables, decomp->edges[b]->variables + decomp->edges[b]->numVariables, v.begin());
                        v.resize(static_cast<unsigned long long int>(it - v.begin()));
                        if (v.size() < combineWidth) {
                            changed = true;
                            cl_long cid = decomp->edges[b]->id;
                            decomp->edges[a]->numVariables = v.size();
                            decomp->edges[a]->variables = new cl_long[decomp->edges[a]->numVariables];
                            std::copy(&v[0], &v[0] + v.size(), decomp->edges[a]->variables);

                            std::vector<preebagType *> v_(
                                    static_cast<unsigned long long int>(decomp->edges[a]->numEdges + decomp->edges[b]->numEdges));
                            std::vector<preebagType *>::iterator it_;
                            it_ = std::set_union(decomp->edges[a]->edges, decomp->edges[a]->edges + decomp->edges[a]->numEdges,
                                                 decomp->edges[b]->edges, decomp->edges[b]->edges + decomp->edges[b]->numEdges, v_.begin(),
                                                 compTreedType);
                            v_.resize(static_cast<unsigned long long int>(it_ - v_.begin()));
                            decomp->edges[a]->numEdges = v_.size();
                            decomp->edges[a]->edges = new preebagType *[decomp->edges[a]->numEdges];
                            std::copy(&v_[0], &v_[0] + v_.size(), decomp->edges[a]->edges);
                            if (b < decomp->numEdges - 1) {
                                std::copy(decomp->edges + b + 1, decomp->edges + decomp->numEdges, decomp->edges + b);
                            }
                            decomp->numEdges--;
                        }
                    }
                }
            }
        }

        changed = true;
        // try to merge with child nodes
        if (decomp->numVariables < combineWidth) {
            while (changed) {
                changed = false;
                for (int i = 0; i < decomp->numEdges; i++) {
                    std::vector<cl_long> v(static_cast<unsigned long long int>(decomp->numVariables + decomp->edges[i]->numVariables));
                    std::vector<cl_long>::iterator it;
                    it = std::set_union(decomp->variables, decomp->variables + decomp->numVariables, decomp->edges[i]->variables,
                                        decomp->edges[i]->variables + decomp->edges[i]->numVariables, v.begin());
                    v.resize(static_cast<unsigned long long int>(it - v.begin()));
                    if (v.size() < combineWidth) {
                        changed = true;
                        cl_long cid = decomp->edges[i]->id;
                        decomp->numVariables = v.size();
                        decomp->variables = new cl_long[decomp->numVariables];
                        std::copy(&v[0], &v[0] + v.size(), decomp->variables);

                        std::vector<preebagType *> v_(static_cast<unsigned long long int>(decomp->numEdges + decomp->edges[i]->numEdges));
                        std::vector<preebagType *>::iterator it_;
                        it_ = std::set_union(decomp->edges, decomp->edges + decomp->numEdges, decomp->edges[i]->edges,
                                             decomp->edges[i]->edges + decomp->edges[i]->numEdges, v_.begin(), compTreedType);
                        v_.resize(static_cast<unsigned long long int>(it_ - v_.begin()));
                        decomp->numEdges = v_.size() - 1;
                        decomp->edges = new preebagType *[decomp->numEdges];
                        for (int asdf = 0, x = 0; asdf < decomp->numEdges; asdf++, x++) {
                            preebagType *&sdggg = v_[asdf];
                            if (v_[asdf]->id == cid) {
                                x++;
                            }
                            decomp->edges[asdf] = v_[x];
                        }
                    }
                }
            }
        }


        // process child nodes
        for (int i = 0; i < decomp->numEdges; i++) {
            preprocessDecomp((decomp->edges)[i]);
        }
    }
}