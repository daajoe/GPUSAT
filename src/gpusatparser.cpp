#include <cmath>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <gpusatparser.h>
#include <unordered_set>
#include <iterator>

namespace gpusat {

    satformulaType CNFParser::parseSatFormula(std::string formula) {
        satformulaType ret = satformulaType();
        std::stringstream ss(formula);
        std::string item;
        std::unordered_map<cl_long, cl_double> weights;
        std::vector<cl_long> *clause = new std::vector<cl_long>();
        if (clause == NULL || errno == ENOMEM) {
            std::cerr << "\nOut of Memory\n";
            exit(0);
        }
        while (getline(ss, item)) {
            //ignore empty line
            if (item.length() > 0) {
                //replace tabs with spaces
				std::replace(item.begin(), item.end(), '\t', ' ');
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
            ret.variableWeights = new cl_double[(ret.numVars + 1) * 2]();
            if (ret.variableWeights == NULL || errno == ENOMEM) {
                std::cerr << "\nOut of Memory\n";
                exit(0);
            }
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

    treedecType TDParser::parseTreeDecomp(std::string graph, satformulaType &formula) {
        treedecType ret;
        std::stringstream ss(graph);
        std::string item;
        std::vector<std::vector<cl_long>> edges;
        while (getline(ss, item)) {
            //ignore empty line
            if (item.length() > 2) {
                //replace tabs with spaces
				std::replace(item.begin(), item.end(), '\t', ' ');
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
            for (long a = 0; a < edges.size(); a++) {
                std::sort(edges[a].begin(), edges[a].end());
            }
            std::list<cl_long> backlog;
            backlog.push_back(0);
            while (backlog.size() > 0) {
                cl_long id = backlog.front();
                backlog.pop_front();

                for (cl_long b : edges[id]) {
                    auto &n = edges[b - 1];
                    auto idx = std::find(n.begin(), n.end(), id + 1);
                    n.erase(idx);
                    backlog.push_back(b - 1);
                }
            }
        }

        if (!edges.empty()) {
            for (long a = 0; a < ret.numb; a++) {
                long b = 0;
                while (!edges[a].empty()) {
                    ret.bags[a].edges.push_back(&ret.bags[edges[a].back() - 1]);
                    edges[a].pop_back();
                    b++;
                }
                std::sort(ret.bags[a].edges.begin(), ret.bags[a].edges.end(), compTreedType);
            }
        }

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
        edges.resize(stoi(i));
        getline(sline, i, ' '); //width
        ret.width = stoi(i);
        getline(sline, i, ' '); //num vars

        ret.numVars = stoi(i);
    }

    void TDParser::parseBagLine(treedecType &ret, std::string item) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //b
        getline(sline, i, ' '); //bag number
        long bnum = stoi(i);
        long a = 0;
        cl_long match_count = 0;
        std::istringstream ss(item);
        std::string word;
        while (ss >> word) {
            std::istringstream maybenumber(word);
            long number = 0;
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
}
