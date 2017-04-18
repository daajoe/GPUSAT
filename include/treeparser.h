#ifndef GPUSAT_TREEPARSER_H
#define GPUSAT_TREEPARSER_H

#include <gpusatutils.h>

/**
 * generates a treedec from a given string
 *
 * @param graph the string representation of the tree decomposition
 * @return the tree decomposition
 */
treedecType parseTreeDecomp(basic_string<char> graph);

/**
 *
 * @param item
 * @param edges
 */
void parseEdgeLine(basic_string<char> item, queue<long> **edges);

/**
 *
 * @param ret
 * @param item
 * @param edges
 */
void parseStartLine(treedecType &ret, basic_string<char> &item, queue<long> **&edges);

/**
 *
 * @param ret
 * @param item
 */
void parseBagLine(treedecType ret, basic_string<char> item);

#endif //GPUSAT_TREEPARSER_H
