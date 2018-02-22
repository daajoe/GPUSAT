#!/usr/bin/env python3
import csv
from os import listdir
from os.path import join, isdir, isfile
from os import makedirs
import datetime
from CommonMethods import *
import decimal

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
splitWidths = [2]
# 0 - incidence
# 1 - primal
# 2 - dual
testGraphs = [2]
# widths to combine
combineWidth = [0]
# 0 - d4
# 1 - double
precision = [1]

flags = ["--noFactRemoval", "--CPU"]

testCasesStrings = [
    "a_00_test", "b_00_test", "c_00_test",  # "e_00_test",  # "d_00_test",
    "a_01_test", "b_01_test", "c_01_test",  # "e_01_test",  # "d_01_test",
    # "a_02_test", "b_02_test", "c_02_test", "e_02_test",  # "d_02_test",
    # "a_03_test", "b_03_test", "c_03_test", "e_03_test",  # "d_03_test",
    # "a_04_test", "b_04_test", "c_04_test", "e_04_test",
    # "Problems",
    # "Test_new",
    # "Tests",
    # "WMC_Tests",
]
fieldnames = ['Case', "Success", 'Graph', "Precision", "Combine Width", "Model Count", 'Model Count Reference',
              "Total Time", "Init Time", "Command"]

if not isfile(summaryFile):
    with open(summaryFile, 'w', newline='') as csvfile:
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


def genDualGraph(formulaFile, graphFile):
    formula = formulaFile.read()
    graph = set()
    numVariables = 0
    variables = {}
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] == 'p':
                numVariables = int(line.split()[2])
                break
    for l in range(1, numVariables + 1):
        variables[l] = []
    clause = 0
    for line in formula.splitlines():
        if len(line) > 2 and line[0] != 'p' and line[0] != 'c' and line[0] != 's' and line[0] != 'w' and line[0] != '%':
            clause += 1
            for lit in [int(x) for x in line.split(" ") if len(x) != 0]:
                if lit != 0:
                    if abs(lit) not in variables:
                        variables[abs(lit)] = []
                    variables[abs(lit)] += [clause]
    for v in variables:
        for c1 in variables[v]:
            for c2 in variables[v]:
                if c1 != c2:
                    graph |= {(min(c1, c2), max(c1, c2))}
                    if len(graph) > 50000000:
                        raise Exception('Dual Oversize')

    graphFile.write("p tw " + str(clause) + " " + str(len(graph)))
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
    decompFile.write(
        subprocess.check_output(["./htd_main", "-s", "1234", "--opt", "width"], stdin=graph, timeout=300).decode(
            'ascii').replace("\r", ""))


def getModels(sol):
    if len(sol) > 0:
        if sol[0] == "{":
            return decimal.Decimal(json.loads(sol)['Models']['Number'])
        elif sol.startswith("cachet"):
            i = 0
            lines = sol.split("\n")
            while i < len(lines):
                if lines[i].startswith("s "):
                    return decimal.Decimal(lines[i].split()[1])
                i += 1
        else:
            i = 0
            lines = sol.split("\n")
            while i < len(lines):
                if "# solutions" in lines[i]:
                    return decimal.Decimal(lines[i + 1])
                i += 1


def getModelsW(sol):
    i = 0
    lines = sol.split("\n")
    while i < len(lines):
        if lines[i].startswith("Satisfying probability"):
            return decimal.Decimal(lines[i].split()[2])
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
            if gr == 2:
                genDualGraph(formulaFile, graphFile)
                print("    Primal Graph")

    # generate decomposition
    with open(dirGraphs + "/" + case + "/" + testcase + postfix + ".gr", 'r')as graphFile:
        with open(dirDecomp + "/" + case + "/" + testcase + postfix + ".td", 'w')as decompFile:
            genTreeDecomp(graphFile, decompFile)

    # get reference
    print("    reference")
    with open(dirReference + "/" + case + "/" + testcase, 'w')as resultFile:
        subprocess.call(["./sharpSAT", dirFormula + "/" + case + "/" + testcase + ".cnf"], stdout=resultFile)
    resultFile = open(dirReference + "/" + case + "/" + testcase, 'r')
    res = resultFile.read()
    resultFile.close()
    if (len(res) == 0):
        with open(dirReference + "/" + case + "/" + testcase, 'w')as resultFile:
            subprocess.call(["./cachet", dirFormula + "/" + case + "/" + testcase + ".cnf"], stdout=resultFile)


def testProgram():
    global resultFile, formulaFile
    try:
        # generate output
        with open(join(join(dirResults, case), testcase + postfix + postfix_prec + postfix_wi + postfix_sw),
                  "w") as resultFile:
            if prec == 0:
                subprocess.call(
                    [pathToExeD4 + "/gpusat.exe", "-f", dirDecomp + "/" + case + "/" + testcase + postfix + ".td", "-s",
                     dirFormula + "/" + case + "/" + testcase + ".cnf", "-w", str(wi), "-m", str(sw), "-c",
                     dirKernel] + flags,
                    stdout=resultFile, stderr=resultFile, timeout=900)
                row[
                    'Command'] = pathToExeD4 + "/gpusat.exe" + " -f " + dirDecomp + "/" + case + "/" + testcase + postfix + ".td -s " + dirFormula + "/" + case + "/" + testcase + ".cnf -c " + dirKernel + " -w " + str(
                    wi) + " -m " + str(sw) + " ".join(flags)
            if prec == 1:
                subprocess.call(
                    [pathToExeDouble + "/gpusat", "-f", dirDecomp + "/" + case + "/" + testcase + postfix + ".td",
                     "-s",
                     dirFormula + "/" + case + "/" + testcase + ".cnf", "-w", str(wi), "-m", str(sw), "-c",
                     dirKernel] + flags,
                    stdout=resultFile, stderr=resultFile, timeout=900)
                row[
                    'Command'] = pathToExeDouble + "/gpusat" + " -f " + dirDecomp + "/" + case + "/" + testcase + postfix + ".td -s " + dirFormula + "/" + case + "/" + testcase + ".cnf -c " + dirKernel + " -w " + str(
                    wi) + " -m " + str(sw) + " ".join(flags)

        weighted = False
        with open(dirFormula + "/" + case + "/" + testcase + ".cnf") as formulaFile:
            weighted = "\nw" in formulaFile.read()
        # check results
        try:
            with open(join(dirResults, case) + "/" + testcase + postfix + postfix_prec + postfix_wi + postfix_sw,
                      "r") as resultFile:
                data = resultFile.read()
            d = json.loads(data)
            with open(join(dirReference, case) + "/" + testcase, "r") as referenceFile:
                ref = str(referenceFile.read())
            row['Model Count'] = decimal.Decimal(d['Model Count'])
            if (weighted):
                numModels = getModelsW(ref)
            else:
                numModels = getModels(ref)
            numModels_ = numModels
            if numModels != 0:
                numModels_ = round(numModels, -int(math.floor(math.log10(abs(numModels))) - (5 - 1)))
            numModelsTest_ = d['Model Count']
            if d['Model Count'] != 0:
                numModelsTest_ = round(d['Model Count'], -int(math.floor(math.log10(abs(d['Model Count']))) - (5 - 1)))
            row['Model Count Reference'] = numModels
            if numModels_ == numModelsTest_:
                row['Success'] = "OK"
                print("    ModelCount: OK")
            else:
                row['Success'] = "Failure"
                print("    ModelCount: Failure " + str(numModels) + "/" + str(d['Model Count']))
            print("    Time Total: " + str(d['Time']['Total']) + " Init: " + str(
                d['Time']['Init_OpenCL']) + " Time Reference: " + str(getTime(ref)))
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
                print("Testcase (" + str(currentElement) + "/" + str(
                    numElements) + "): " + case + "/" + testcase + " " + str(datetime.datetime.now().time()))
                if gr == 0:
                    postfix = "_i"
                    print("    Incidence Graph")
                    row['Graph'] = "incidence"
                if gr == 1:
                    postfix = "_p"
                    print("    Primal Graph")
                    row['Graph'] = "primal"
                if gr == 2:
                    postfix = "_d"
                    print("    Dual Graph")
                    row['Graph'] = "dual"
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
