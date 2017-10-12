#!/usr/bin/env python2.7
import json
from os import listdir
import subprocess, getpass
from os.path import join, isdir, isfile
from os import makedirs
import datetime

dirFormula = "./new_problems/formula"
dirDecomp = "./new_problems/decomposition"
dirResults = "./new_problems/results"
dirReference = "./new_problems/reference"
dirGraphs = "./new_problems/graph"

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

testCasesStrings = [
    "a_00_test", "b_00_test", "c_00_test",
    "a_01_test", "b_01_test", "c_01_test",
    "a_02_test", "c_02_test", "b_02_test",
    "a_03_test", "c_03_test", "b_03_test"
]


def getModels(sol):
    if sol[0] == "{":
        return json.loads(sol)['Models']['Number']
    elif sol.startswith("cachet"):
        i = 0
        lines = sol.split("\n")
        while i < len(lines):
            if lines[i].startswith("s "):
                return int(lines[i].split()[1])
            i += 1
    else:
        i = 0
        lines = sol.split("\n")
        while i < len(lines):
            if "# solutions" in lines[i]:
                return int(lines[i + 1])
            i += 1


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
            if line[0] != 'p' and line[0] != 'c' and line[0] != 's' and line:
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


def genIncidenceGraph(formulaFile, graphFile):
    formula = formulaFile.read()
    graphEdges = ""
    graph = set()
    numRules = 0
    max_clause_size = 0
    numVariables = 0
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] == 'p':
                numVariables = int(line.split()[2])
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] != 'p' and line[0] != 'c' and line[0] != 's':
                numRules += 1
                lits = line.split()
                max_clause_size = len(lits) - 1 if (len(lits) - 1) > max_clause_size else max_clause_size
                for node in lits:
                    ainode = abs(int(node))
                    if ainode != 0:
                        graph |= {(ainode, numRules + numVariables)}
                        if len(graph) > 50000000:
                            raise Exception('Incidence Oversize')

    graphFile.write("p tw " + str(numVariables + numRules) + " " + str(len(graph)))
    for node in graph:
        graphFile.write("\n" + str(node[0]) + " " + str(node[1]))
    return max_clause_size


# generate tree decomposition
def genTreeDecomp(graph, decompFile):
    decompFile.write(
        subprocess.check_output(["./htd_main", "--opt", "width", "-s", "1234"], stdin=graph, timeout=300).decode('ascii').replace("\r", ""))


def getTime(sol):
    if sol[0] == "{":
        return json.loads(sol)['Time']['Total']
    elif sol.startswith("cachet"):
        i = 0
        lines = sol.split("\n")
        while i < len(lines):
            if lines[i].startswith("Total Run Time "):
                return float(lines[i].split()[3])
            i += 1
    else:
        i = 0
        lines = sol.split("\n")
        while i < len(lines):
            if "time: " in lines[i]:
                return float(lines[i].split(" ")[1][:-1])
            i += 1


def getModels(sol):
    if sol[0] == "{":
        return json.loads(sol)['Models']['Number']
    elif sol.startswith("cachet"):
        i = 0
        lines = sol.split("\n")
        while i < len(lines):
            if lines[i].startswith("s "):
                return int(lines[i].split()[1])
            i += 1
    else:
        i = 0
        lines = sol.split("\n")
        while i < len(lines):
            if "# solutions" in lines[i]:
                return int(lines[i + 1])
            i += 1


def check_model(model, clauses):
    for line in clauses:
        sat = False
        for var in line[:-1]:
            if var in model:
                sat = True
                break
        if not sat:
            return False
    return True


for case in testCasesStrings:
    if not isdir(join(dirResults, case)):
        makedirs(join(dirResults, case))
    if not isdir(join(dirDecomp, case)):
        makedirs(join(dirDecomp, case))
    if not isdir(join(dirGraphs, case)):
        makedirs(join(dirGraphs, case))
    numElements = len(listdir(join(dirReference, case)))
    currentElement = 0
    for testcase in listdir(join(dirReference, case)):
        currentElement += 1
        if not isfile(join(join(dirResults, case), testcase)):
            with open("./Summary.txt", "a") as summaryFile:
                summaryFile.write("Testcase (" + str(currentElement) + "/" + str(numElements) + "): " + case + "/" + testcase + "\n")
                summaryFile.flush()
                print("Testcase (" + str(currentElement) + "/" + str(numElements) + "): " + case + "/" + testcase + " " + str(
                    datetime.datetime.now().time()))

                try:
                    # generate output
                    with open(join(join(dirResults, case), testcase), "w") as resultFile:
                        subprocess.call(
                            ["./gpusat.exe", "-f", dirDecomp + "/" + case + "/" + testcase + "_p.td", "-s", dirFormula + "/" + case + "/" + testcase + ".cnf"],
                            timeout=30, stdout=resultFile, stderr=resultFile)

                    # check results
                    with open(join(dirResults, case) + "/" + testcase, "r") as resultFile:
                        with open(join(dirReference, case) + "/" + testcase, "r") as referenceFile:
                            try:
                                data = resultFile.read()
                                d = json.loads(data)
                                ref = str(referenceFile.read())
                                summaryFile.write("    ModelCount: ")
                                numModels = getModels(ref)
                                if d['Model Count'] == getModels(ref):
                                    summaryFile.write("OK\n")
                                    print("    ModelCount: OK")
                                else:
                                    summaryFile.write("Failure: " + str(d['Model Count']) + "/" + str(getModels(ref)) + "\n")
                                    print("    ModelCount: Failure: " + str(d['Model Count']) + "/" + str(getModels(ref)) + " ")
                            except ValueError:
                                summaryFile.write("    Error\n")
                                print("    Error")
                except subprocess.TimeoutExpired:
                    summaryFile.write("    Timeout\n")
                    print("    Timeout")
