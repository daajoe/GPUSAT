#include <satparser.h>

satformulaType parseSatFormula(string formula) {
    satformulaType ret = satformulaType();
    stringstream ss(formula);
    string item;
    queue<queue<varIdType>> *clauses = NULL;
    while (getline(ss, item)) {
        char type = item.at(0);
        if (type == 'c') {
            //comment (ignore)
        } else if (type == 'p') {
            //start
            parseProblemLine(ret, item, clauses);
        } else {
            //clause
            parseClauseLine(item, clauses);
        }
    }

    if (clauses != NULL) {
        int a = 0;
        while (!clauses->empty()) {
            queue<varIdType> &clause = clauses->front();
            ret.clauses[a].var = new varIdType[clause.size()];
            ret.clauses[a].numVars = clause.size();

            int b = 0;
            while (!clause.empty()) {
                ret.clauses[a].var[b] = clause.front();
                clause.pop();
                b++;
            }
            clauses->pop();
            a++;
        }
    }
    return ret;
}

void parseClauseLine(string item, queue<queue<varIdType>> *clauses) {
    stringstream sline(item);
    string i;
    queue<varIdType> clause;
    getline(sline, i, ' ');
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
    satformula.clauses = new clauseType[stoi(i)];
    clauses = new queue<queue<varIdType>>();
}