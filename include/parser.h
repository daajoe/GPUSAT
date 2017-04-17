#ifndef LOGICSEM_PARSER_H
#define LOGICSEM_PARSER_H

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <fstream>
#include <parser.h>
#include <utils.h>

using namespace std;

treedec parseTreeDecomp(basic_string<char> graph);

void parseEdge(basic_string<char> item, queue<long> **edges);

void parseStart(treedec &ret, basic_string<char> &item, queue<long> **&edges);

void parseBag(treedec ret, basic_string<char> item);

#include <queue>

#endif //LOGICSEM_PARSER_H
