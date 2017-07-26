#!/usr/bin/env python
import subprocess
from os.path import join, isdir, isfile
from os import makedirs, listdir, remove
from subprocess import check_output

maxWidth = 20
maxNumModels = pow(2, 62)

dirRaw = "./dynasp/raw"
dirFormula = "./dynasp/formula"
dirDecomp = "./dynasp/decomposition"
dirResults = "./dynasp/results"
dirReference = "./dynasp/reference"
dirGraphs = "./dynasp/graph"

if not isdir(dirRaw):
    makedirs(dirRaw)
if not isdir(dirFormula):
    makedirs(dirFormula)
if not isdir(dirGraphs):
    makedirs(dirGraphs)
if not isdir(dirDecomp):
    makedirs(dirDecomp)
if not isdir(dirReference):
    makedirs(dirReference)
if not isdir(dirResults):
    makedirs(dirResults)


# check formula
def checkFormula(formula, resultFile):
    subprocess.call(["./sharpSAT", "-t", "900", formula], stdout=resultFile)


# generate tree decomposition
def genTreeDecomp(graph, decompFile):
    decompFile.write(check_output(["./htd_main", "--opt", "width", "-s", "1234"], stdin=graph, timeout=120).decode('ascii'))


def genPrimalGraph(formulaFile, graphFile):
    formula = formulaFile.read()
    graphEdges = ""
    graph = {}
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] == 'p':
                numVariables = int(line.split()[2])
    for i in range(1, numVariables + 1):
        graph[i] = set()
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] != 'p' and line[0] != 'c':
                for node in line.split():
                    if int(node) != 0:
                        for node2 in line.split():
                            if int(node2) != 0 and node!=node2:
                                graph[abs(int(node))] |= {abs(int(node2))}
                                graph[abs(int(node2))] |= {abs(int(node))}

    for key in graph.keys():
        for node in graph[key]:
            if node > key:
                graphEdges += str(key) + " " + str(node) + "\n"

    graphString = "p tw " + str(numVariables) + " " + str(len(graphEdges.split('\n')) - 1) + "\n" + graphEdges[:-1]
    graphFile.write(graphString)


for testcase in listdir(dirRaw):
    try:
        case = testcase[:-8]
        with open(join(dirRaw, testcase), "r") as formula:
            with open(join(dirFormula, case + ".cnf"), "w") as f:
                f.write(formula.read())

        print("formula: " + testcase)

        print("    primal graph")
        # generate the primal graph of the cnf formula
        with open(join(dirGraphs, case + ".gr"), "w") as graph:
            with open(join(dirFormula, case + ".cnf"), "r") as formula:
                genPrimalGraph(formula, graph)

        print("    gen decomp")
        # generate the tree decomposition
        with open(join(dirDecomp, case + ".td"), "w") as decomp:
            with open(join(dirGraphs, case + ".gr"), "r") as graph:
                genTreeDecomp(graph, decomp)

        # check decomposition
        with open(join(dirDecomp, case + ".td"), "r") as decomp:
            line = decomp.readline()
            if int(line.split(" ")[3]) > maxWidth:
                print("width: " + line.split(" ")[3])
                remove(join(dirFormula, case + ".cnf"))
                remove(join(dirGraphs, case + ".gr"))
                remove(join(dirDecomp, case + ".td"))
                continue
    except:
        print("Error")
        if isfile(join(dirFormula, case + ".cnf")):
            remove(join(dirFormula, case + ".cnf"))
        if isfile(join(dirGraphs, case + ".gr")):
            remove(join(dirGraphs, case + ".gr"))
        if isfile(join(dirDecomp, case + ".td")):
            remove(join(dirDecomp, case + ".td"))
