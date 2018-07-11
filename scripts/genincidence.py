#!/usr/bin/env python2.7
from sets import Set
import sys

with open(sys.argv[1]) as formulaFile:
    formula = formulaFile.read()
    graph = set()
    numRules = 0
    numVariables = 0
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] == 'p':
                numVariables = int(line.split()[2])
    clauses = []
    clause = []
    for line in formula.splitlines():
        if len(line) > 0 and line[0] != 'p' and line[0] != 'c' and line[0] != 's' and line[0] != 'w' and line[0] != '%':
            for lit in [int(x) for x in line.split() if len(x) != 0]:
                if lit == 0 and len(clause) > 0:
                    clauses += [clause]
                    clause = []
                else:
                    clause += [lit]
    for lits in clauses:
        numRules += 1
        for node in lits:
            ainode = abs(int(node))
            if ainode != 0:
                graph |= {(ainode, numRules + numVariables)}
                if len(graph) > 50000000:
                    raise Exception('Incidence Oversize')

    graphString = ("p tw " + str(numVariables + numRules) + " " + str(len(graph)))
    for node in graph:
        graphString += ("\n" + str(node[0]) + " " + str(node[1]))
    print(graphString)