#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <gpusatparser.h>
#include <unordered_set>

namespace gpusat {

    satformulaType CNFParser::parseSatFormula(std::string formula) {
        satformulaType ret = satformulaType();
        std::stringstream ss(formula);
        std::string item;
        std::unordered_map<cl_long, cl_double> weights;
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
                    parseClauseLine(ret, item, clause);
                }
            }
        }

        if (wmc) {
            ret.variableWeights = new cl_double[(ret.numVars + 1) * 2]();
            ret.numWeights = (ret.numVars + 1) * 2;

            for (cl_long i = 0; i <= ret.numVars; i++) {
                std::unordered_map<cl_long, cl_double>::const_iterator elem = weights.find(i);
                if (elem != weights.end()) {
                    cl_double we = weights[i];
                    if (we < 0.0) {
                        ret.variableWeights[i * 2] = 1;
                        ret.variableWeights[i * 2 + 1] = 1;
                    } else {
                        ret.variableWeights[i * 2] = weights[i];
                        ret.variableWeights[i * 2 + 1] = 1.0 - weights[i];
                    }
                } else {
                    ret.variableWeights[i * 2] = 0.5;
                    ret.variableWeights[i * 2 + 1] = 0.5;
                }
            }
        } else {
            ret.variableWeights = new cl_double[(ret.numVars + 1) * 2]();
            ret.numWeights = (ret.numVars + 1) * 2;

            for (cl_long i = 0; i <= ret.numVars; i++) {
                ret.variableWeights[i * 2] = 0.78;
                ret.variableWeights[i * 2 + 1] = 0.78;
            }
        }
        return ret;
    }

    void CNFParser::parseClauseLine(satformulaType &ret, std::string &item, std::vector<cl_long> *clause) {
        std::stringstream sline(item);
        std::string i;

        cl_long num = 0;
        while (!sline.eof()) {
            getline(sline, i, ' ');
            if (i.size() > 0) {
                num = stol(i);
                if (num != 0) {
                    clause->push_back(num);
                }
            }
        }
        if (clause->size() > 1 && num == 0) {
            sort(clause->begin(), clause->end(), compVars);
            ret.clauses.push_back(*clause);
            clause->resize(0);
        } else if (clause->size() == 1 && num == 0) {
            if (find(ret.facts.begin(), ret.facts.end(), (*clause)[0]) == ret.facts.end())
                ret.facts.push_back((*clause)[0]);
            ret.clauses.push_back(*clause);
            clause->resize(0);
        }
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

    void CNFParser::parseWeightLine(std::string item, std::unordered_map<cl_long, cl_double> &weights) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //w
        getline(sline, i, ' '); //variable
        cl_long id = stol(i);
        getline(sline, i, ' '); //weight
        cl_double val = stod(i);
        weights[id] = (id < 0) ? -val : val;
    }

    CNFParser::CNFParser(bool weighted) {
        wmc = weighted;
    }

    TDParser::TDParser(int i, bool b, int i1) {
        combineWidth = i;
        factR = b;
    }

    void TDParser::iterateDecompPre(bagType &bag){
        this->preWidth = (bag.variables.size() > this->preWidth) ? bag.variables.size() : this->preWidth;
        std::unordered_set<cl_long> joinSet;

        for (auto &node:bag.edges) {
            iterateDecompPre(*node);
            std::set_intersection(bag.variables.begin(), bag.variables.end(), node->variables.begin(), node->variables.end(), std::inserter(joinSet, joinSet.begin()));
            std::unordered_set<cl_long> cutSet;
            std::set_intersection(bag.variables.begin(), bag.variables.end(), node->variables.begin(), node->variables.end(), std::inserter(cutSet, cutSet.begin()));
            this->preCut = (cutSet.size() > this->preCut) ? cutSet.size() : this->preCut;
        }

        if (bag.edges.size() > 1) {
            this->preJoinSize = (joinSet.size() > this->preJoinSize) ? joinSet.size() : this->preJoinSize;
        }
    }

    void TDParser::iterateDecompPost(bagType &bag){
        this->postWidth = (bag.variables.size() > this->postWidth) ? bag.variables.size() : this->postWidth;
        std::unordered_set<cl_long> joinSet;

        for (auto &node:bag.edges) {
            iterateDecompPost(*node);
            std::set_intersection(bag.variables.begin(), bag.variables.end(), node->variables.begin(), node->variables.end(), std::inserter(joinSet, joinSet.begin()));
            std::unordered_set<cl_long> cutSet;
            std::set_intersection(bag.variables.begin(), bag.variables.end(), node->variables.begin(), node->variables.end(), std::inserter(cutSet, cutSet.begin()));
            this->postCut = (cutSet.size() > this->postCut) ? cutSet.size() : this->postCut;
        }

        if (bag.edges.size() > 1) {
            this->postJoinSize = (joinSet.size() > this->postJoinSize) ? joinSet.size() : this->postJoinSize;
        }
    }

    treedecType TDParser::parseTreeDecomp(std::string graph, satformulaType &formula, graphTypes gType) {
        treedecType ret;
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

        iterateDecompPre(ret.bags[0]);
        this->preNumBags = ret.bags.size();
        return ret;
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

    void TDParser::parseStartLine(treedecType &ret, std::string &item, std::vector<std::vector<cl_long>> &edges) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //s
        getline(sline, i, ' '); //td
        getline(sline, i, ' '); //num bags
        ret.bags.resize(stoi(i));
        ret.numb = stoi(i);
        for (int a = 0; a < ret.numb; a++) {
            std::vector<cl_long> edge;
            edges.push_back(edge);
        }
        getline(sline, i, ' '); //width
        getline(sline, i, ' '); //num vars
        ret.numVars = stoi(i);
    }

    void TDParser::parseBagLine(treedecType &ret, std::string item) {
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
}