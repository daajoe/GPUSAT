#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else

#include <CL/cl.hpp>

#endif

#include <iostream>
#include <sstream>
#include <main.h>
#include <unistd.h>
#include <fstream>

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

treedec parseTreeDecomp(string graph) {
    treedec ret = treedec();
    stringstream ss(graph);
    string item;
    queue<bagId> **edges = NULL;
    while (getline(ss, item)) {
        char type = item.at(0);
        if (type == 'c') {
            //comment (ignore)
        } else if (type == 's') {
            //start
            parseStart(ret, item, edges);
        } else if (type == 'b') {
            //bag
            parseBag(ret, item);
        } else {
            //edge
            parseEdge(item, edges);
        }
    }

    if (edges != NULL) {
        for (int a = 0; a < ret.numb; a++) {
            ret.bags[a].edges = new bagId[edges[a]->size()];
            ret.bags[a].nume = edges[a]->size();
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

void parseEdge(string item, queue<bagId> **edges) {
    stringstream sline(item);
    string i;
    getline(sline, i, ' '); //start
    bagId start = stoi(i);
    getline(sline, i, ' '); //end
    bagId end = stoi(i);
    edges[start - 1]->push(end - 1);
    edges[end - 1]->push(start - 1);
}

void parseStart(treedec &ret, string &item, queue<bagId> **&edges) {
    stringstream sline(item);
    string i;
    getline(sline, i, ' '); //s
    getline(sline, i, ' '); //tw
    getline(sline, i, ' '); //num bags
    ret.bags = new bag[stoi(i)];
    ret.numb = stoi(i);
    edges = new queue<bagId> *[stoi(i)];
    for (int a = 0; a < stoi(i); a++) {
        edges[a] = new queue<bagId>();
    }
}

void parseBag(treedec ret, string item) {
    stringstream sline(item);
    string i;
    getline(sline, i, ' '); //b
    getline(sline, i, ' '); //bag number
    int bnum = stoi(i);
    int a = 0;
    ret.bags[bnum - 1].vertices = new vertexId[numCharOcc(item, ' ') - 1];
    while (getline(sline, i, ' ')) //vertices
    {
        ret.bags[bnum - 1].vertices[a] = stoi(i);
        a++;
        ret.bags[bnum - 1].numv = a;
    }
}

int numCharOcc(string str, char c) {
    int numOcc = 0;
    for (unsigned long long int i = 0; i < str.length(); i++) {
        if (str.at(i) == c) {
            numOcc++;
        }
    }
    return numOcc;
}

void printTreeD(treedec decomp) {
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