#include <satparser.h>
#include <regex>

satformulaType parseSatFormula(string formula) {
    satformulaType ret = satformulaType();
    stringstream ss(formula);
    string item;
    queue<queue<varIdType>> *clauses = NULL;
    clauseIdType clauseSize = 0;
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
        ret.clauses = new varIdType[clauseSize];
        while (!clauses->empty()) {
            queue<varIdType> &clause = clauses->front();
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

void parseClauseLine(string item, queue<queue<varIdType>> *clauses, clauseIdType &clauseSize) {
    stringstream sline(item);
    string i;
    queue<varIdType> clause;
    getline(sline, i, ' ');
    regex const expression("[0123456789]+");
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

void parseProblemLine(satformulaType &satformula, string item, queue<queue<varIdType>> *&clauses) {
    stringstream sline(item);
    string i;
    getline(sline, i, ' '); //p
    getline(sline, i, ' '); //cnf
    getline(sline, i, ' '); //num vars
    satformula.numVars = stoi(i);
    getline(sline, i, ' '); //num clauses
    satformula.numclauses = stoi(i);
    satformula.numVarsC = new varIdType[satformula.numclauses];
    clauses = new queue<queue<varIdType>>();
}