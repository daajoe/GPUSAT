#include <gpusatutils.h>

using namespace std;

int numCharOcc(string str, char c) {
    int numOcc = 0;
    for (unsigned long long int i = 0; i < str.length(); i++) {
        if (str.at(i) == c) {
            numOcc++;
        }
    }
    return numOcc;
}

void printTreeD(treedecType decomp) {
    int size = decomp.numb;
    for (int i = 0; i < size; i++) {
        cout << "\nbagnum: " << i + 1 << "\n";
        int vsize = decomp.bags[i].numv;
        cout << "vertices: ";
        for (int a = 0; a < vsize; a++) {
            cout << decomp.bags[i].vertices[a] << " ";
        }
        cout << "\n";
        unsigned long long int esize = decomp.bags[i].nume;
        cout << "edges: ";
        for (int a = 0; a < esize; a++) {
            cout << decomp.bags[i].edges[a] + 1 << " ";
        }
        cout << "\n";
    }
}

void printFormula(satformulaType formula) {
    int size = formula.numclauses;
    for (int i = 0; i < size; i++) {
        cout << "\nclause: " << i + 1 << "\n";
        int vsize = formula.clauses[i].numVars;
        cout << "variables: ";
        for (int a = 0; a < vsize; a++) {
            cout << formula.clauses[i].var[a] << " ";
        }
        cout << "\n";
    }
}

string readFile(string path) {
    stringbuf treeD;
    string inputLine;
    ifstream fileIn(path);
    while (getline(fileIn, inputLine)) {
        treeD.sputn(inputLine.c_str(), inputLine.size());
        treeD.sputn("\n", 1);
    }
    return treeD.str();
}