#ifndef LOGICSEM_PARSER_H
#define LOGICSEM_PARSER_H

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <fstream>
#include <parser.h>
#include <utils.h>

using namespace std;

/**
 * generates a treedec from a given string
 *
 * @param graph the string representation of the tree decomposition
 * @return the tree decomposition
 */
treedec parseTreeDecomp(string graph);

/**
 *
 * @param item
 * @param edges
 */
void parseEdge(string item, queue<long> **edges);

/**
 *
 * @param ret
 * @param item
 * @param edges
 */
void parseStart(treedec &ret, string &item, queue<long> **&edges);

/**
 *
 * @param ret
 * @param item
 */
void parseBag(treedec ret, string item);

#include <queue>

#endif //LOGICSEM_PARSER_H
