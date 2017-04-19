#include <treeparser.h>
#include <math.h>
#include <regex>

treedecType parseTreeDecomp(string graph) {
    treedecType ret = treedecType();
    stringstream ss(graph);
    string item;
    queue<bagIdType> **edges = NULL;
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
            ret.bags[a].edges = new bagIdType[edges[a]->size()];
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

void parseEdgeLine(string item, queue<bagIdType> **edges) {
    stringstream sline(item);
    string i;
    getline(sline, i, ' '); //start
    bagIdType start = stoi(i);
    getline(sline, i, ' '); //end
    bagIdType end = stoi(i);
    edges[start - 1]->push(end);
}

void parseStartLine(treedecType &ret, string &item, queue<bagIdType> **&edges) {
    stringstream sline(item);
    string i;
    getline(sline, i, ' '); //s
    getline(sline, i, ' '); //tw
    getline(sline, i, ' '); //num bags
    ret.bags = new bagType[stoi(i)];
    ret.numb = stoi(i);
    edges = new queue<bagIdType> *[stoi(i)];
    for (int a = 0; a < stoi(i); a++) {
        edges[a] = new queue<bagIdType>();
    }
}

void parseBagLine(treedecType ret, string item) {
    stringstream sline(item);
    string i;
    getline(sline, i, ' '); //b
    getline(sline, i, ' '); //bag number
    int bnum = stoi(i);
    int a = 0;
    regex const expression("[0123456789]+");
    std::ptrdiff_t const match_count(std::distance(
            std::sregex_iterator(item.begin(), item.end(), expression),
            std::sregex_iterator()));

    ret.bags[bnum - 1].id = bnum;
    ret.bags[bnum - 1].vertices = new vertexIdType[match_count - 1];
    ret.bags[bnum - 1].solution = new solutionType[(long) pow(2, match_count - 1)];
    while (getline(sline, i, ' ')) //vertices
    {
        ret.bags[bnum - 1].vertices[a] = stoi(i);
        a++;
    }
    ret.bags[bnum - 1].numv = match_count - 1;
    ret.bags[bnum - 1].numSol = (long) pow(2, match_count - 1);
    for (long b = 0; b < (long) pow(2, match_count - 1); b++) {
        ret.bags[bnum - 1].solution[b].vars = new varIdType[match_count - 1];
    }
}