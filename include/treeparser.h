#ifndef GPUSAT_TREEPARSER_H
#define GPUSAT_TREEPARSER_H

#include <gpusatutils.h>
#include <queue>

/**
 * generates a treedec from a given string
 *
 * @param graph the string representation of the tree decomposition
 * @return the tree decomposition
 */
treedecType parseTreeDecomp(std::string graph);

/**
 *
 * @param item
 * @param edges
 */
void parseEdgeLine(std::string item, std::queue<cl_long> **edges);

/**
 *
 * @param ret
 * @param item
 * @param edges
 */
void parseStartLine(treedecType &ret, std::string &item, std::queue<cl_long> **&edges);

/**
 *
 * @param ret
 * @param item
 */
void parseBagLine(treedecType ret, std::string item);

#endif //GPUSAT_TREEPARSER_H
