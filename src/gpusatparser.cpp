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
        std::vector<int64_t> *clause = new std::vector<int64_t>();
        if (clause == NULL || errno == ENOMEM) {
            std::cerr << "\nOut of Memory\n";
            exit(0);
        }
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
                    parseClauseLine(ret, item, clause);
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
        delete clause;
        return ret;
    }

    void CNFParser::parseClauseLine(satformulaType &ret, std::string &item, std::vector<int64_t> *clause) {
        std::stringstream sline(item);
        std::string i;

        int64_t num = 0;
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
