#!/usr/bin/env python2.7
import random
import subprocess
from os.path import join, isdir, isfile
import json
from os import makedirs, listdir, remove
from sets import Set

maxWidth = 20
maxNumModels = pow(2, 62)

dirRaw = "./benchmarks/raw"
dirFormula = "./benchmarks/formula"
dirDecomp = "./benchmarks/decomposition"
dirResults = "./benchmarks/results"
dirReference = "./benchmarks/reference"
dirGraphs = "./benchmarks/graph"

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


def getModels(sol):
    if sol[0] == "{":
        return json.loads(sol)['Models']['Number']
    elif sol.startswith("cachet"):
        i = 0
        lines = sol.split("\n")
        while i < len(lines):
            if lines[i].startswith("s "):
                if (lines[i].split()[1] == "inf"):
                    return pow(2, 65)
                else:
                    return int(lines[i].split()[1])
            i += 1
    else:
        i = 0
        lines = sol.split("\n")
        while i < len(lines):
            if "# solutions" in lines[i]:
                return int(lines[i + 1])
            i += 1


# check formula
def checkFormula(formula, resultFile):
    subprocess.call(["./sharpSAT", "-t", "900", formula], stdout=resultFile)


# generate tree decomposition
def genTreeDecomp(graph, decompFile):
    subprocess.call(["./htd_main", "--opt", "width", "-s", "1234"], stdout=decompFile, stdin=graph)


# check formula
def preprocessFormula(formula, resultFile):
    subprocess.call(
        ["./preproc", "-no-solve", "-vivification", "-eliminateLit", "-litImplied", "-iterate=1", "-equiv", "-orGate",
         "-affine", formula], stdout=resultFile)


def genPrimalGraph(formulaFile, graphFile):
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
    graphFile.write(graphString)


for testcase in listdir(dirFormula):
    case = testcase[:-4]

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
            print ("width: " + line.split(" ")[3])
            remove(join(dirFormula, case + ".cnf"))
            remove(join(dirGraphs, case + ".gr"))
            remove(join(dirDecomp, case + ".td"))
            continue

#    print("    check formula")
#    # check the formula
#    with open(join(dirReference, case), "w") as reference:
#        checkFormula(join(dirFormula, case + ".cnf"), reference)
#
#    # check number of solutions
#    with open(join(dirReference, case), "r") as reference:
#        ref = reference.read()
#        if getModels(ref) >= maxNumModels:
#            print("    num Models: " + str(getModels(ref)))
#            remove(join(dirFormula, case + ".cnf"))
#            remove(join(dirGraphs, case + ".gr"))
#            remove(join(dirDecomp, case + ".td"))
#            remove(join(dirReference, case))
#            continue
