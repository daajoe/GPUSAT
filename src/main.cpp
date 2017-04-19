#include <gpusatutils.h>
#include <satparser.h>
#include <treeparser.h>
#include <solver.h>
#include <CL/cl.hpp>

using namespace std;
using namespace cl;

int main(int argc, char *argv[]) {
    stringbuf treeD, sat;
    string inputLine;
    bool file = false, formula = false;
    int opt;
    while ((opt = getopt(argc, argv, "f:s:")) != -1) {
        switch (opt) {
            case 'f': {
                // input tree decomposition file
                file = true;
                ifstream fileIn(optarg);
                while (getline(fileIn, inputLine)) {
                    treeD.sputn(inputLine.c_str(), inputLine.size());
                    treeD.sputn("\n", 1);
                }
                break;
            }
            case 's': {
                // input sat formula
                formula = true;
                ifstream fileIn(optarg);
                while (getline(fileIn, inputLine)) {
                    sat.sputn(inputLine.c_str(), inputLine.size());
                    sat.sputn("\n", 1);
                }
                break;
            }
            default:
                fprintf(stderr, "Usage: %s [-f treedecomp] -s formula \n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    // no file flag
    if (!file) {
        while (getline(cin, inputLine)) {
            treeD.sputn(inputLine.c_str(), inputLine.size());
            treeD.sputn("\n", 1);
        }
    }

    // error no sat formula given
    if (!formula) {
        fprintf(stderr, "Usage: %s [-f treedecomp] -s formula \n", argv[0]);
        exit(EXIT_FAILURE);
    }

    treedecType treeDecomp = parseTreeDecomp(treeD.str());
    satformulaType satFormula = parseSatFormula(sat.str());
    printTreeD(treeDecomp);
    printFormula(satFormula);
    solveProblem(treeDecomp, satFormula, treeDecomp.bags[0]);
    int solutions = 0;
    for (int i = 0; i < treeDecomp.bags[0].numSol; i++) {
        solutions += treeDecomp.bags[0].solution[i].n;
    }
    cout << "Solutions: " << solutions;
    printSolutions(treeDecomp);
}
