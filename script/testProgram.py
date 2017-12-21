#!/usr/bin/env python3
import csv
from os import listdir
from os.path import join, isdir, isfile
from os import makedirs
import datetime
from CommonMethods import *

import math

import itertools

dirFormula = "./new_problems/formula"
dirDecomp = "./new_problems/decomposition"
dirResults = "./new_problems/results"
dirReference = "./new_problems/reference"
dirGraphs = "./new_problems/graph"

summaryFile = './Summary.csv'
pathToExeD4 = "../build_mingw_d4"
pathToExeDouble = "../build_mingw_double"
dirKernel = "../kernel/"

# width before splitting bags
splitWidths = [18]
# 0 - incidence
# 1 - primal
testGraphs = [1]
# widths to combine
combineWidth = [0]
# 0 - d4
# 1 - double
precision = [1]

testCasesStrings = [
    # "Problems",
    # "a_00_test", "c_00_test", "e_00_test", #"d_00_test", "e_00_test",
    # "a_01_test", "c_01_test", "e_01_test", #"d_01_test", "e_01_test",
    "a_02_test", "c_02_test", "e_02_test",  # "e_02_test",
    "a_03_test", "c_03_test", "e_03_test",  # "e_03_test",
    # "a_04_test", "b_04_test", "c_04_test", "d_04_test", "e_04_test",
    # "Tests"
]
fieldnames = ['Case', "Success", 'Graph', "Precision", "Combine Width", "Model Count", 'Model Count Reference', "Total Time", "Init Time", "Command"]

if not isfile(summaryFile):
    with open(summaryFile, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
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


def genPrimalGraph(formulaFile, graphFile):
    formula = formulaFile.read()
    graphEdges = ""
    graph = set()
    numVariables = 0
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] == 'p':
                numVariables = int(line.split()[2])
    clauses = []
    clause = []
    for line in formula.splitlines():
        if len(line) > 0 and line[0] != 'p' and line[0] != 'c' and line[0] != 's' and line[0] != 'w' and line[0] != '%':
            for lit in [int(x) for x in line.split(" ") if len(x) != 0]:
                if lit == 0 and len(clause) > 0:
                    clauses += [clause]
                    clause = []
                else:
                    clause += [lit]
    for line in clauses:
        for node in line:
            ainode = abs(int(node))
            if ainode != 0:
                for node2 in line:
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
    clauses = []
    clause = []
    for line in formula.splitlines():
        if len(line) > 0 and line[0] != 'p' and line[0] != 'c' and line[0] != 's' and line[0] != 'w' and line[0] != '%':
            for lit in [int(x) for x in line.split(" ") if len(x) != 0]:
                if lit == 0 and len(clause) > 0:
                    clauses += [clause]
                    clause = []
                else:
                    clause += [lit]
    for lits in clauses:
        numRules += 1
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


# generate tree decomposition
def genTreeDecomp(graph, decompFile):
    decompFile.write(subprocess.check_output(["./htd_main", "-s", "1234", "--opt", "width"], stdin=graph, timeout=300).decode('ascii').replace("\r", ""))


def getModels(sol):
    if len(sol) > 0:
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


def getModelsW(sol):
    i = 0
    lines = sol.split("\n")
    while i < len(lines):
        test = lines[i].split()
        if lines[i].startswith("Satisfying probability"):
            return float(lines[i].split()[2])
        i += 1


def getDecompRef():
    global formulaFile, resultFile
    # generate Graph
    with open(dirFormula + "/" + case + "/" + testcase + ".cnf", 'r')as formulaFile:
        with open(dirGraphs + "/" + case + "/" + testcase + postfix + ".gr", 'w')as graphFile:
            if gr == 0:
                genIncidenceGraph(formulaFile, graphFile)
                print("    Incidence Graph")
            if gr == 1:
                genPrimalGraph(formulaFile, graphFile)
                print("    Primal Graph")

    # generate decomposition
    with open(dirGraphs + "/" + case + "/" + testcase + postfix + ".gr", 'r')as graphFile:
        with open(dirDecomp + "/" + case + "/" + testcase + postfix + ".td", 'w')as decompFile:
            genTreeDecomp(graphFile, decompFile)

            # get reference
            # print("    reference")
            # with open(dirReference + "/" + case + "/" + testcase, 'w')as resultFile:
            #    subprocess.call(["./cachet-wmc", dirFormula + "/" + case + "/" + testcase + ".cnf"], stdout=resultFile)
            # resultFile = open(dirReference + "/" + case + "/" + testcase, 'r')
            # res = resultFile.read()
            # resultFile.close()
            # if (len(res) == 0):
            #    with open(dirReference + "/" + case + "/" + testcase, 'w')as resultFile:
            #        subprocess.call(["./cachet-wmc", dirFormula + "/" + case + "/" + testcase + ".cnf"], stdout=resultFile)


def testProgram():
    global resultFile, formulaFile
    try:
        # generate output
        with open(join(join(dirResults, case), testcase + postfix + postfix_prec + postfix_wi + postfix_sw), "w") as resultFile:
            if prec == 0:
                subprocess.call([pathToExeD4 + "/gpusat.exe", "-f", dirDecomp + "/" + case + "/" + testcase + postfix + ".td", "-s",
                                 dirFormula + "/" + case + "/" + testcase + ".cnf", "-w", str(wi), "-m", str(sw), "-c", dirKernel,
                                 "--noFactRemoval"],
                                stdout=resultFile, stderr=resultFile, timeout=150)
                row[
                    'Command'] = pathToExeD4 + "/gpusat.exe" + " -f " + dirDecomp + "/" + case + "/" + testcase + postfix + ".td -s " + dirFormula + "/" + case + "/" + testcase + ".cnf -c " + dirKernel + " -w " + str(
                    wi) + " -m " + str(sw) + " --noFactRemoval"
            if prec == 1:
                subprocess.call([pathToExeDouble + "/gpusat.exe", "-f", dirDecomp + "/" + case + "/" + testcase + postfix + ".td", "-s",
                                 dirFormula + "/" + case + "/" + testcase + ".cnf", "-w", str(wi), "-m", str(sw), "-c", dirKernel,
                                 "--noFactRemoval"],
                                stdout=resultFile, stderr=resultFile, timeout=150)
                row[
                    'Command'] = pathToExeDouble + "/gpusat.exe" + " -f " + dirDecomp + "/" + case + "/" + testcase + postfix + ".td -s " + dirFormula + "/" + case + "/" + testcase + ".cnf -c " + dirKernel + " -w " + str(
                    wi) + " -m " + str(sw) + " --noFactRemoval"

        weighted = False
        with open(dirFormula + "/" + case + "/" + testcase + ".cnf") as formulaFile:
            weighted = "\nw" in formulaFile.read()
        # check results
        try:
            with open(join(dirResults, case) + "/" + testcase + postfix + postfix_prec + postfix_wi + postfix_sw, "r") as resultFile:
                data = resultFile.read()
            d = json.loads(data)
            with open(join(dirReference, case) + "/" + testcase, "r") as referenceFile:
                ref = str(referenceFile.read())
            row['Model Count'] = d['Model Count']
            if (weighted):
                numModels = getModelsW(ref)
                row['Model Count Reference'] = numModels
                if (numModels == d['Model Count']) or (numModels != 0 and d['Model Count'] != 0 and round(d['Model Count'], int(
                        math.log10(d['Model Count'])) + 5) == round(numModels, int(math.log10(numModels)) + 5)):
                    row['Success'] = "OK"
                    print("    ModelCount: OK")
                else:
                    row['Success'] = "Failure"
                    print("    ModelCount: Failure " + str(numModels) + "/" + str(d['Model Count']))
            else:
                numModels = getModels(ref)
                row['Model Count Reference'] = numModels
                if d['Model Count'] == numModels:
                    row['Success'] = "OK"
                    print("    ModelCount: OK")
                else:
                    row['Success'] = "Failure"
                    print("    ModelCount: Failure " + str(numModels) + "/" + str(d['Model Count']))
            print("    Time Total: " + str(d['Time']['Total']) + " Init: " + str(d['Time']['Init_OpenCL']) + " Time Reference: " + str(
                getTime(ref)))
            row['Total Time'] = d['Time']['Total']
            row['Init Time'] = d['Time']['Init_OpenCL']
        except ValueError:
            row['Success'] = "Error"
            print("    Error")
    except subprocess.TimeoutExpired:
        row['Success'] = "Timeout"
        print("    Timeout")
    with open(summaryFile, 'a', newline='') as csvf:
        wr = csv.DictWriter(csvf, fieldnames=fieldnames)
        wr.writerow(row)
        csvf.flush()


for case in testCasesStrings:
    if not isdir(join(dirResults, case)):
        makedirs(join(dirResults, case))
    if not isdir(join(dirDecomp, case)):
        makedirs(join(dirDecomp, case))
    if not isdir(join(dirGraphs, case)):
        makedirs(join(dirGraphs, case))
    if not isdir(join(dirReference, case)):
        makedirs(join(dirReference, case))
    numElements = len(listdir(join(dirFormula, case)))
    currentElement = 0
    for testcase in listdir(join(dirFormula, case)):
        testcase = testcase[:-4]
        currentElement += 1
        if not isfile(join(join(dirResults, case), testcase)):
            open(join(dirResults, case) + "/" + testcase, "w").close()
            for wi, gr, prec, sw in itertools.product(combineWidth, testGraphs, precision, splitWidths):
                row = {'Case': case + "/" + testcase}
                print("Testcase (" + str(currentElement) + "/" + str(numElements) + "): " + case + "/" + testcase + " " + str(datetime.datetime.now().time()))
                if gr == 0:
                    postfix = "_i"
                    print("    Incidence Graph")
                    row['Graph'] = "incidence"
                if gr == 1:
                    postfix = "_p"
                    print("    Primal Graph")
                    row['Graph'] = "primal"
                if prec == 0:
                    postfix_prec = "_d4"
                    print("    d4")
                    row['Precision'] = "d4"
                if prec == 1:
                    postfix_prec = "_d"
                    print("    double")
                    row['Precision'] = "double"
                postfix_wi = "_" + str(wi)
                postfix_sw = "_" + str(sw)
                print("    wi " + str(wi))
                row['Combine Width'] = str(wi)

                # getDecompRef()

                testProgram()
