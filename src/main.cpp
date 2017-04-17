#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else

#include <CL/cl.hpp>

#endif

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <fstream>
#include <utils.h>
#include <parser.h>

using namespace std;
using namespace cl;

int main(int argc, char *argv[]) {
    stringbuf treeD;
    string inputLine;
    bool file = false;
    int opt;
    while ((opt = getopt(argc, argv, "f:")) != -1) {
        switch (opt) {
            case 'f': {
                file = true;
                ifstream fileIn(optarg);
                while (getline(fileIn, inputLine)) {
                    treeD.sputn(inputLine.c_str(), inputLine.size());
                    treeD.sputn("\n", 1);
                }
                break;
            }
            default:
                fprintf(stderr, "Usage: %s [-f file] \n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    if (!file) {
        while (getline(cin, inputLine)) {
            treeD.sputn(inputLine.c_str(), inputLine.size());
            treeD.sputn("\n", 1);
        }
    }
    treedec treeDecomp = parseTreeDecomp(treeD.str());
    printTreeD(treeDecomp);
}
