#ifndef LOGICSEM_UTILS_H
#define LOGICSEM_UTILS_H

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <fstream>
#include <utils.h>

using namespace std;

typedef long int vertexId;
typedef long int bagId;

struct bag {
    int numv = 0;
    unsigned long long int nume = 0;
    vertexId *vertices;
    bagId *edges;
};

struct treedec {
    int numb;
    bag *bags;
};

int numCharOcc(string str, char c);

void printTreeD(treedec decomp);

string readFile(string path);

#endif //LOGICSEM_UTILS_H
