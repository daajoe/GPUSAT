#include <cmath>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <iterator>

#include "gpusatparser.h"

namespace gpusat {

    satformulaType CNFParser::parseSatFormula(std::istream& ss) {
        satformulaType ret = satformulaType();
        std::string item;
        std::unordered_map<int64_t, double> weights;
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
                } else if (type == 's') {
                    //weight line
                    this->parseSolutionLine(item);
                } else {
                    //clause line
                    parseClauseLine(ret, item);
                }
            }
            if (item.find("c UNSATISFIABLE") != std::string::npos) {
                std::cout << "    \"pre Width\": " << 0;
                std::cout << "\n    ,\"pre Cut Set Size\": " << 0;
                std::cout << "\n    ,\"pre Join Size\": " << 0;
                std::cout << "\n    ,\"pre Bags\": " << 0;
                std::cout << "\n    ,\"Model Count\": " << 0;
                std::cout << "\n    ,\"Time\":{";
                std::cout << "\n        \"Solving\": " << 0;
                std::cout << "\n        ,\"Total\": " << 0;
                std::cout << "\n    }";
                std::cout << "\n    ,\"Statistics\":{";
                std::cout << "\n        \"Num Join\": " << 0;
                std::cout << "\n        ,\"Num Forget\": " << 0;
                std::cout << "\n        ,\"Num Introduce\": " << 0;
                std::cout << "\n        ,\"Num Leaf\": " << 0;
                std::cout << "\n    }";
                std::cout << "\n}\n";
                exit(20);
            }
        }
        std::sort(ret.facts.begin(), ret.facts.end(), compVars);

        if (wmc) {
            ret.variableWeights = new double[(ret.numVars + 1) * 2]();
            if (ret.variableWeights == NULL || errno == ENOMEM) {
                std::cerr << "\nOut of Memory\n";
                exit(0);
            }
            ret.numWeights = (ret.numVars + 1) * 2;

            for (size_t i = 0; i <= ret.numVars; i++) {
                std::unordered_map<int64_t, double>::const_iterator elem = weights.find(i);
                if (elem != weights.end()) {
                    double we = weights[i];
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
        }
        ret.numVars -= ret.facts.size();
        return ret;
    }

    void CNFParser::parseClauseLine(satformulaType &ret, std::string &item) {
        std::stringstream sline(item);
        std::string i;

        int64_t num = 0;
        int count = 0;
        int64_t first = 0;
        size_t clause_start = ret.clause_bag.size();

        while (!sline.eof()) {
            getline(sline, i, ' ');
            if (i.size() > 0) {
                num = stol(i);
                if (num != 0) {
                    // hold in case this is a fact.
                    if (count == 0) {
                        first = num;
                    // this is at least a binary clause.
                    } else {
                        if (first) {
                            ret.clause_bag.push_back(first);
                            first = 0;
                        }
                        ret.clause_bag.push_back(num);
                    }
                }
                count++;
            }
        }
        // store a fact
        if (first) {
            // we sort this later
            ret.facts.push_back(first);
        // or finish the clause
        } else {
            std::sort(ret.clause_bag.begin() + clause_start, ret.clause_bag.end(), compVars);
            ret.clause_bag.push_back(0);
            ret.clause_offsets.push_back(clause_start);
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

    void CNFParser::parseWeightLine(std::string item, std::unordered_map<int64_t, double> &weights) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //w
        getline(sline, i, ' '); //variable
        int64_t id = stol(i);
        getline(sline, i, ' '); //weight
        double val = stod(i);
        weights[id] = (id < 0) ? -val : val;
    }

    CNFParser::CNFParser(bool weighted) {
        wmc = weighted;
    }

    void CNFParser::parseSolutionLine(std::string item) {
        if (item.find("UNSATISFIABLE") != std::string::npos) {
            std::cout << "\n{\n";
            std::cout << "    \"Num Join\": " << 0;
            std::cout << "\n    ,\"Num Introduce Forget\": " << 0;
            std::cout << "\n    ,\"max Table Size\": " << 0;
            std::cout << "\n    ,\"Model Count\": " << 0;
            std::cout << "\n    ,\"Time\":{";
            std::cout << "\n        \"Decomposing\": " << 0;
            std::cout << "\n        ,\"Solving\": " << 0;
            std::cout << "\n        ,\"Total\": " << 0;
            std::cout << "\n    }";
            std::cout << "\n}\n";
            exit(20);
        }
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //s
        getline(sline, i, ' '); //solutions

        std::cout << "\n{\n";
        std::cout << "    \"Num Join\": " << 0;
        std::cout << "\n    ,\"Num Introduce Forget\": " << 0;
        std::cout << "\n    ,\"max Table Size\": " << 0;
        std::cout << "\n    ,\"Model Count\": " << i;
        std::cout << "\n    ,\"Time\":{";
        std::cout << "\n        \"Decomposing\": " << 0;
        std::cout << "\n        ,\"Solving\": " << 0;
        std::cout << "\n        ,\"Total\": " << 0;
        std::cout << "\n    }";
        std::cout << "\n}\n";
        exit(20);

    }
}
