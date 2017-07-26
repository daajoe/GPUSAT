from sets import Set
import sys

with open(sys.argv[1]) as formulaFile:
    formula = formulaFile.read()
    graphEdges = ""
    graph = {}
    for line in formula.splitlines():
        if line[0] == 'p':
            numVariables = int(line.split()[2])
    for i in range(1, numVariables + 1):
        graph[i] = Set()
    for line in formula.splitlines():
        if line[0] != 'p' and line[0] != 'c':
            for node in line.split():
                if int(node) != 0:
                    for node2 in line.split():
                        if int(node2) != 0:
                            graph[abs(int(node))].add(abs(int(node2)))
                            graph[abs(int(node2))].add(abs(int(node)))

    for key in graph.keys():
        for node in graph[key]:
            if node > key:
                graphEdges += str(key) + " " + str(node) + "\n"

    graphString = "p tw " + str(numVariables) + " " + str(len(graphEdges.split('\n')) - 1) + "\n" + graphEdges[:-1]
    print(graphString)
