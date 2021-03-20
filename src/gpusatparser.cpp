#include <cmath>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <iterator>
#include <deque>

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

    BagType extract_bag(int64_t target_id, std::vector<BagType>& bags) {
        auto it = std::find_if(bags.begin(), bags.end(), [target_id](const BagType& b) { return b.id == target_id; });
        assert(it != bags.end());
        BagType bag;
        std::swap(bag, *it);
        bags.erase(it);
        return std::move(bag);
    }

    treedecType TDParser::parseTreeDecomp(std::istream& ss, std::vector<int64_t>& removed_facts) {
        treedecType ret;
        std::string item;
        std::vector<std::vector<int64_t>> edges;
        std::vector<BagType> bags;
        std::deque<BagType*> backlog;

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
                    bags.push_back(parseBagLine(item));
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
            std::list<int64_t> backlog;
            backlog.push_back(0);
            while (backlog.size() > 0) {
                int64_t id = backlog.front();
                backlog.pop_front();

                // delete edges with improper orientation
                for (int64_t b : edges[id]) {
                    auto &n = edges[b - 1];
                    auto idx = std::find(n.begin(), n.end(), id + 1);
                    n.erase(idx);
                    backlog.push_back(b - 1);
                }
            }
        }

        assert(bags.size() == ret.numb);
        ret.root = extract_bag(0, bags);
        backlog.push_back(&ret.root);
        while (!backlog.empty()) {
            auto current_bag = backlog[0];
            backlog.pop_front();

            // reserve the current size, so we do not have to reallocate
            // and the backlog pointers stay valid
            current_bag->edges.reserve(edges[current_bag->id].size());
            removeFactsFromDecomposition(*current_bag, removed_facts);

            for (auto child_id : edges[current_bag->id]) {
                child_id = child_id - 1;
                auto bag = std::move(extract_bag(child_id, bags));
                current_bag->edges.push_back(std::move(bag));
                backlog.push_back(&current_bag->edges.back());
            }
        }
        assert(bags.size() == 0);

        return ret;
    }

    void TDParser::removeFactsFromDecomposition(BagType& bag, std::vector<int64_t>& to_remove) {
        // to_remove, must be sorted w.r.t. compVars!

        int64_t prev_var = 0;
        std::vector<int64_t> new_variables;
        // vars should be already sorted as well
        for (auto var : bag.variables) {
            assert(var > 0);
            assert(var > prev_var);
            prev_var = var;

            bool is_fact = std::binary_search(to_remove.begin(), to_remove.end(), var, compVars);
            if (is_fact) {
                continue;
            } else {
                auto offset = std::lower_bound(to_remove.begin(), to_remove.end(), var, compVars) - to_remove.begin();

                assert(var - offset <= 974);
                new_variables.push_back(var - offset);
            }
        }
        bag.variables = std::move(new_variables);
    }

    void TDParser::parseEdgeLine(std::string item, std::vector<std::vector<int64_t>> &edges) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //start
        int64_t start = stoi(i);
        getline(sline, i, ' '); //end
        int64_t end = stoi(i);
        edges[start - 1].push_back(end);
        edges[end - 1].push_back(start);
        std::sort(edges[start-1].begin(), edges[start-1].end());
        std::sort(edges[end-1].begin(), edges[end-1].end());
    }

    void TDParser::parseStartLine(treedecType &ret, std::string &item, std::vector<std::vector<int64_t>> &edges) {
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //s
        getline(sline, i, ' '); //td
        getline(sline, i, ' '); //num bags
        //bags.resize(stoi(i));
        ret.numb = stoi(i);
        edges.resize(stoi(i));
        getline(sline, i, ' '); //width
        ret.width = stoi(i);
        getline(sline, i, ' '); //num vars

        ret.numVars = stoi(i);
    }

    BagType TDParser::parseBagLine(std::string item) {
        BagType bag;
        std::stringstream sline(item);
        std::string i;
        getline(sline, i, ' '); //b
        getline(sline, i, ' '); //bag number
        long bnum = stoi(i);
        long a = 0;
        int64_t match_count = 0;
        std::istringstream ss(item);
        std::string word;
        while (ss >> word) {
            std::istringstream maybenumber(word);
            long number = 0;
            if (maybenumber >> number) {
                match_count++;
            }
        }
        bag.id = bnum - 1;
        while (getline(sline, i, ' ')) //vertices
        {
            if (i[0] != '\r') {
                bag.variables.push_back(stoi(i));
                a++;
            }
        }
        std::sort(bag.variables.begin(), bag.variables.end());
        return std::move(bag);
    }
}
