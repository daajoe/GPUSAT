#ifndef LOGICSEM_MAIN_H
#define LOGICSEM_MAIN_H

#include <list>
#include <queue>

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

void parseBag(treedec ret, string item);

void parseStart(treedec &ret, string &item, queue<bagId> **&edges);

void parseEdge(string item, queue<bagId> **edges);

treedec parseTreeDecomp(string graph);

void printTreeD(treedec decomp);

#endif //LOGICSEM_MAIN_H
