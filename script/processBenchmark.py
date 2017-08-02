#!/usr/bin/env python
import csv
import subprocess
from os.path import join, isdir, isfile
from os import makedirs, listdir, remove
from shutil import copyfile
from subprocess import check_output

maxWidth = 22
saveWidth = 60

dirRaw = "./benchmarks/raw"
dirFormula = "./benchmarks/formula"
dirDecomp = "./benchmarks/decomposition"
dirResults = "./benchmarks/results"
dirReference = "./benchmarks/reference"
dirGraphs = "./benchmarks/graph"
dirSaveFromula = "./benchmarks/save/formula"
dirSaveGraph = "./benchmarks/save/graph"

fieldnames = ['file_name', 'width']

summaryFile = 'Summary_Benchmark_Width.csv'

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
if not isdir(dirSaveFromula):
    makedirs(dirSaveFromula)
if not isdir(dirSaveGraph):
    makedirs(dirSaveGraph)
if not isfile(summaryFile):
    with open(summaryFile, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


# check formula
def checkFormula(formula, resultFile):
    subprocess.call(["./sharpSAT", "-t", "900", formula], stdout=resultFile)


# generate tree decomposition
def genTreeDecomp(graph, decompFile):
    try:
        decompFile.write(
            check_output(["./htd_main", "--opt", "width", "-s", "1234"], stdin=graph, timeout=120).decode('ascii'))
    except subprocess.TimeoutExpired:
        return


def genPrimalGraph(formulaFile, graphFile):
    formula = formulaFile.read()
    graphEdges = ""
    graph = set()
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] == 'p':
                numVariables = int(line.split()[2])
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] != 'p' and line[0] != 'c':
                for node in line.split():
                    ainode = abs(int(node))
                    if ainode != 0:
                        for node2 in line.split():
                            ainode2 = abs(int(node2))
                            if ainode2 != 0 and ainode != node2:
                                if (ainode < ainode2):
                                    graph |= {(ainode, ainode2)}
                                elif (ainode > ainode2):
                                    graph |= {(ainode2, ainode)}
                                if len(graph) > 10000000:
                                    return

    for node in graph:
            graphEdges += str(node[0]) + " " + str(node[1]) + "\n"

    graphString = "p tw " + str(numVariables) + " " + str(len(graphEdges.split('\n')) - 1) + "\n" + graphEdges[:-1]
    graphFile.write(graphString)


for testcase in listdir(dirRaw):
    try:
        case = testcase[:-4]
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
                if int(line.split(" ")[3]) <= saveWidth:
                    copyfile(join(dirFormula, case + ".cnf"), join(dirSaveFromula, case + ".cnf"))
                    copyfile(join(dirGraphs, case + ".gr"), join(dirSaveGraph, case + ".gr"))
                remove(join(dirFormula, case + ".cnf"))
                remove(join(dirGraphs, case + ".gr"))
                remove(join(dirDecomp, case + ".td"))

            with open('Summary_Benchmark_Width.csv', 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'file_name': case, 'width': line.split(" ")[3]})
    except:
        print("Error")
        if isfile(join(dirFormula, case + ".cnf")):
            remove(join(dirFormula, case + ".cnf"))
        if isfile(join(dirGraphs, case + ".gr")):
            remove(join(dirGraphs, case + ".gr"))
        if isfile(join(dirDecomp, case + ".td")):
            remove(join(dirDecomp, case + ".td"))
