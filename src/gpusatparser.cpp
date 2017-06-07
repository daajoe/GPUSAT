#include <math.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <main.h>
#include <gpusatparser.h>
#include <algorithm>

satformulaType parseSatFormula(std::string formula) {
    satformulaType ret = satformulaType();
    std::stringstream ss(formula);
    std::string item;
    std::queue<std::queue<cl_int >> *clauses = NULL;
    cl_int clauseSize = 0;
    while (getline(ss, item)) {
        char type = item.at(0);
        if (type == 'c') {
            //comment (ignore)
        } else if (type == 'p') {
            //start
            parseProblemLine(ret, item, clauses);
        } else {
            //clause
            parseClauseLine(item, clauses, clauseSize);
        }
    }

    if (clauses != NULL) {
        int a = 0, s = 0;
        ret.totalNumVar = clauseSize;
        ret.clauses = new cl_int[clauseSize]();
        while (!clauses->empty()) {
            std::queue<cl_int> &clause = clauses->front();
            ret.numVarsC[a] = clause.size();

            int b = 0;
            while (!clause.empty()) {
                ret.clauses[s + b] = clause.front();
                clause.pop();
                b++;
            }
            std::sort(&ret.clauses[s], &ret.clauses[s + b]);
            clauses->pop();
            s += ret.numVarsC[a];
            a++;
        }
    }
    return ret;
}

void parseClauseLine(std::string item, std::queue<std::queue<cl_int>> *clauses, cl_int &clauseSize) {
    std::stringstream sline(item);
    std::string i;
    std::queue<cl_int> clause;
    getline(sline, i, ' ');
    cl_int match_count = 1;
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
        if (i.size() > 0) {
            if (stoi(i) == 0) {
                break;
            } else {
                clause.push(stoi(i));
            }
        }
        getline(sline, i, ' ');
    }
    clauses->push(clause);
}

void parseProblemLine(satformulaType &satformula, std::string item, std::queue<std::queue<cl_int>> *&clauses) {
    std::stringstream sline(item);
    std::string i;
    getline(sline, i, ' '); //p
    getline(sline, i, ' '); //cnf
    getline(sline, i, ' '); //num vars
    satformula.numVar = stoi(i);
    getline(sline, i, ' '); //num clauses
    satformula.numclauses = stoi(i);
    satformula.numVarsC = new cl_int[satformula.numclauses]();
    clauses = new std::queue<std::queue<cl_int>>();
}

treedecType parseTreeDecomp(std::string graph) {
    treedecType ret = treedecType();
    std::stringstream ss(graph);
    std::string item;
    std::queue<cl_int> **edges = NULL;
    while (getline(ss, item)) {
        char type = item.at(0);
        if (type == 'c') {
            //comment (ignore)
        } else if (type == 's') {
            //start
            parseStartLine(ret, item, edges);
        } else if (type == 'b') {
            //bag
            parseBagLine(ret, item);
        } else {
            //edge
            parseEdgeLine(item, edges);
        }
    }

    if (edges != NULL) {
        for (int a = 0; a < ret.numb; a++) {
            ret.bags[a].edges = new cl_int[edges[a]->size()]();
            ret.bags[a].numEdges = edges[a]->size();
            int b = 0;
            while (!edges[a]->empty()) {
                ret.bags[a].edges[b] = edges[a]->front();
                edges[a]->pop();
                b++;
            }
        }
    }
    return ret;
}

void parseEdgeLine(std::string item, std::queue<cl_int> **edges) {
    std::stringstream sline(item);
    std::string i;
    getline(sline, i, ' '); //start
    cl_int start = stoi(i);
    getline(sline, i, ' '); //end
    cl_int end = stoi(i);
    edges[start - 1]->push(end);
}

void parseStartLine(treedecType &ret, std::string &item, std::queue<cl_int> **&edges) {
    std::stringstream sline(item);
    std::string i;
    getline(sline, i, ' '); //s
    getline(sline, i, ' '); //tw
    getline(sline, i, ' '); //num bags
    ret.bags = new bagType[stoi(i)];
    ret.numb = stoi(i);
    edges = new std::queue<cl_int> *[stoi(i)];
    for (int a = 0; a < stoi(i); a++) {
        edges[a] = new std::queue<cl_int>();
    }
}

void parseBagLine(treedecType ret, std::string item) {
    std::stringstream sline(item);
    std::string i;
    getline(sline, i, ' '); //b
    getline(sline, i, ' '); //bag number
    int bnum = stoi(i);
    int a = 0;
    cl_int match_count = 0;
    std::istringstream ss(item);
    std::string word;
    while (ss >> word) {
        std::istringstream maybenumber(word);
        int number = 0;
        if (maybenumber >> number) {
            match_count++;
        }
    }


    ret.bags[bnum - 1].variables = new cl_int[match_count - 1]();
    ret.bags[bnum - 1].numSol = (long) pow(2, match_count - 1);
    ret.bags[bnum - 1].solution = new cl_int[sizeof(cl_int) * (ret.bags[bnum - 1].numSol)]();
    ret.bags[bnum - 1].numVars = match_count - 1;
    while (getline(sline, i, ' ')) //vertices
    {
        ret.bags[bnum - 1].variables[a] = stoi(i);
        a++;
    }
    std::sort(ret.bags[bnum - 1].variables, &ret.bags[bnum - 1].variables[ret.bags[bnum - 1].numVars]);
}