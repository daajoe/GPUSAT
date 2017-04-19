#include <satparser.h>
#include <regex>

satformulaType parseSatFormula(std::string formula) {
    satformulaType ret = satformulaType();
    std::stringstream ss(formula);
    std::string item;
    std::queue<std::queue<cl_long >> *clauses = NULL;
    cl_long clauseSize = 0;
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
        ret.clauses = new cl_long[clauseSize];
        while (!clauses->empty()) {
            std::queue<cl_long> &clause = clauses->front();
            ret.numVarsC[a] = clause.size();

            int b = 0;
            while (!clause.empty()) {
                ret.clauses[s + b] = clause.front();
                clause.pop();
                b++;
            }
            clauses->pop();
            s += ret.numVarsC[a];
            a++;
        }
    }
    return ret;
}

void parseClauseLine(std::string item, std::queue<std::queue<cl_long>> *clauses, cl_long &clauseSize) {
    std::stringstream sline(item);
    std::string i;
    std::queue<cl_long> clause;
    getline(sline, i, ' ');
    std::regex const expression("[0123456789]+");
    std::ptrdiff_t const match_count(std::distance(
            std::sregex_iterator(i.begin(), i.end(), expression),
            std::sregex_iterator()));
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

void parseProblemLine(satformulaType &satformula, std::string item, std::queue<std::queue<cl_long>> *&clauses) {
    std::stringstream sline(item);
    std::string i;
    getline(sline, i, ' '); //p
    getline(sline, i, ' '); //cnf
    getline(sline, i, ' '); //num vars
    getline(sline, i, ' '); //num clauses
    satformula.numclauses = stoi(i);
    satformula.numVarsC = new cl_long[satformula.numclauses];
    clauses = new std::queue<std::queue<cl_long>>();
}