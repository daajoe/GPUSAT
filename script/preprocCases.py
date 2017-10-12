#!/usr/bin/env python2.7
from os import listdir, devnull
import subprocess
from os.path import join, isdir
from os import makedirs

dirFormula = "./new_problems/formula"
dirReference = "./new_problems/reference"

dirFormula_New = "./new_problems_preproc/formula"
dirDecomp_New = "./new_problems_preproc/decomposition"
dirGraph_New = "./new_problems_preproc/graph"

testCasesStrings = ["a_00_test", "b_00_test", "c_00_test", "a_01_test", "b_01_test", "c_01_test", "a_02_test", "b_02_test", "c_02_test", "a_03_test",
                    "b_03_test", "c_03_test"]


def genPrimalGraph(formulaFile, graphFile):
    formula = formulaFile.read()
    graphEdges = ""
    graph = set()
    numVariables = 0
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] == 'p':
                numVariables = int(line.split()[2])
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] != 'p' and line[0] != 'c' and line[0]!='s':
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
                                if len(graph) > 50000000:
                                    raise Exception('Primal Oversize')

    graphFile.write("p tw " + str(numVariables) + " " + str(len(graph)))
    for node in graph:
        graphFile.write("\n" + str(node[0]) + " " + str(node[1]))


for case in testCasesStrings:
    if not isdir(join(dirDecomp_New, case)):
        makedirs(join(dirDecomp_New, case))
    if not isdir(join(dirFormula_New, case)):
        makedirs(join(dirFormula_New, case))
    if not isdir(join(dirGraph_New, case)):
        makedirs(join(dirGraph_New, case))
    numElements = len(listdir(join(dirReference, case)))
    currentElement = 0
    for testcase in listdir(join(dirReference, case)):
        print (case + " - " + str(currentElement))
        currentElement += 1
        with open(dirFormula_New + "/" + case + "/" + testcase + ".cnf","w") as formulaFile:
            FNULL = open(devnull, 'w')
            subprocess.call(
                ["./preproc", "-vivification", "-eliminateLit", "-litImplied", "-iterate=10", "-equiv", "-orGate", "-affine",
                 dirFormula + "/" + case + "/" + testcase + ".cnf"], stdout=formulaFile,stderr=FNULL)

        with open(dirFormula_New + "/" + case + "/" + testcase + ".cnf", "r") as formulaFile:
            with open(dirGraph_New + "/" + case + "/" + testcase + ".gr", "w")as graphFile:
                genPrimalGraph(formulaFile, graphFile)

        with open(dirGraph_New + "/" + case + "/" + testcase + ".gr", "r") as graphFile:
            with open(dirDecomp_New + "/" + case + "/" + testcase + ".td", "w") as decompFile:
                subprocess.call(["./htd_main", "--opt", "width", "-s", "1234"], stdout=decompFile, stdin=graphFile)
