#!/usr/bin/env python
import csv
import subprocess
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool
from os.path import join, isdir, isfile, devnull
from os import makedirs, listdir
from subprocess import check_output

import time

dirRaw = "./benchmarks/raw"
dirFormula = "./benchmarks/formula"
dirDecomp = "./benchmarks/decomposition"
dirGraphs = "./benchmarks/graph"

fieldnames = ['file_name', 'width_primal_s1234', "width_incidence_s1234", "max_clause_size", "max_var_occ", 'width_primal_simplepreproc_s1234',
              "width_incidence_simplepreproc_s1234", "sat-unsat", "max_clause_size_simplepreproc", "max_var_occ_simplepreproc"]

oldSummaryFile = './benchmarks/Summary_Benchmark_Width.csv'
summaryFile = './benchmarks/Summary_Benchmark_Width.csv_'

total_size = len(listdir(dirRaw))
processing = 0

with open(summaryFile, "r") as csvfile:
    contents = csvfile.read()

lock = Lock()

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


# generate tree decomposition
def genTreeDecomp(graph, decompFile, seed):
    decompFile.write(
        check_output(["./htd_main", "--opt", "width", "-s", str(seed)], stdin=graph, timeout=300).decode(
            'ascii'))


# preprocess formula
def preprocFormula(fDir):
    try:
        FNULL = open(devnull, 'w')
        return (
            check_output(
                ["./preproc", "-vivification", "-eliminateLit", "-litImplied", "-iterate=10", "-equiv", "-orGate",
                 "-affine",
                 fDir], timeout=300, stderr=FNULL).decode('ascii'))
    except subprocess.CalledProcessError as asdf:
        return asdf.output.decode('ascii')


# check if formula is satisfiable
def checkSAT(formula):
    try:
        out = check_output(["./lingeling-bbc"], stdin=formula, timeout=300).decode('ascii')
        if "s UNSATISFIABLE" in out:
            return 'UNSATISFIABLE'
        if "s SATISFIABLE" in out:
            return 'SATISFIABLE'
        return 'UNKNOWN'
    except subprocess.CalledProcessError as asdf:
        if "s UNSATISFIABLE" in asdf.output.decode('ascii'):
            return 'UNSATISFIABLE'
        if "s SATISFIABLE" in asdf.output.decode('ascii'):
            return 'SATISFIABLE'
        return 'UNKNOWN'
    except Exception as ex:
        print("        Error: checkSAT - " + str(type(ex)) + " " + str(ex.args))
        return 'UNKNOWN'


# get maximal occurences of a variable
def getMaxVarOcc(formulaFile):
    formula = formulaFile.read()
    numVariables = 0
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] == 'p':
                numVariables = int(line.split()[2])
    varOcc = [0] * (numVariables + 1)
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] != 'p' and line[0] != 'c' and line[0] != 's':
                lits = line.split()
                for node in lits:
                    ainode = abs(int(node))
                    if ainode != 0:
                        varOcc[ainode] += 1
    maxOcc = 0
    for i in varOcc:
        if i > maxOcc:
            maxOcc = i
    return maxOcc


# generate the primal graph
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


# generate the incidence graph
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


# simple preprocessing
def simplePreproc(formula, timelimit=30):
    start = time.time()
    newFormula = formula.splitlines()
    newFormula = [x for x in newFormula if len(x) > 0 or x[0] == 'p' or x[0] == 'w' or x[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
    allElems = []
    if len([x for x in newFormula if len([y for y in x.split(" ") if (str(int(y) * -1) in x.split(" "))]) > 0]) > 0:
        return "p cnf 1 1\n1 -1 0\n"
    singleElems = [x for x in newFormula if len(x) > 0 and x[0] != 'p' and x[0] != 'c' and x[0] != 's' and len(x.split(" ")) == 1 and not (x in allElems)]
    if timelimit >= (time.time() - start):
        return '\n'.join(newFormula) + '\n'
    while len(singleElems) > 0:
        newFormula = [x for x in newFormula if
                      (len(x) > 0 or x[0] == 'p' or x[0] == 'w') and len([y for y in x.split(" ") if str(int(y) * -1) in singleElems]) == 0]
        if timelimit >= (time.time() - start):
            return '\n'.join(newFormula) + '\n'
        allElems += singleElems
        singleElems = [x for x in newFormula if len(x) > 0 and x[0] != 'p' and x[0] != 'c' and x[0] != 's' and len(x.split(" ")) == 1 and not x in allElems]
        if timelimit >= (time.time() - start):
            return '\n'.join(newFormula) + '\n'
    return '\n'.join(newFormula) + '\n'


def process(testcase):
    case = testcase['file_name']
    global processing
    current = processing
    processing += 1
    if case not in contents:
        iwidth = -1
        pwidth = -1
        with open(join(dirRaw, case + ".cnf"), "r") as formula:
            with open(join(dirFormula, case + ".cnf"), "w") as f:
                f.write(formula.read())

        print(str(total_size) + "/" + str(current) + " formula: " + case + ".cnf")

        if testcase['max_var_occ'] in (None, ""):
            with open(join(dirFormula, case + ".cnf"), "r") as formula:
                testcase['max_var_occ'] = getMaxVarOcc(formula)

        if testcase['sat-unsat'] in (None, ""):
            with open(join(dirFormula, case + ".cnf"), "r") as formula:
                testcase['sat-unsat'] = checkSAT(formula)

        if testcase['width_primal_s1234'] in (None, ""):
            testcase['width_primal_s1234'] = checkPrimal(case, pwidth, "1234", dirFormula)
        if testcase['width_incidence_s1234'] in (None, ""):
            testcase['width_incidence_s1234'] = checkIncidence(case, iwidth, "1234")
        if testcase['max_clause_size'] in (None, ""):
            with open(join(dirGraphs, case + ".i.gr"), "w") as graph:
                with open(join(dirFormula, case + ".cnf"), "r") as formula:
                    testcase['max_clause_size'] = genIncidenceGraph(formula, graph)

        with open(join(dirRaw, case + ".cnf"), "r") as rawFormula:
            preFormulaString = simplePreproc(rawFormula.read())
            with open(join(dirFormula, case + ".cnf"), "w") as preFormula:
                preFormula.write(preFormulaString)
        if testcase['width_primal_simplepreproc_s1234'] in (None, ""):
            testcase['width_primal_simplepreproc_s1234'] = checkPrimal(case, pwidth, "1234", dirFormula)
        if testcase['width_incidence_simplepreproc_s1234'] in (None, ""):
            testcase['width_incidence_simplepreproc_s1234'] = checkIncidence(case, iwidth, "1234")
        if testcase['max_clause_size_simplepreproc'] in (None, ""):
            with open(join(dirGraphs, case + ".i.gr"), "w") as graph:
                with open(join(dirFormula, case + ".cnf"), "r") as formula:
                    testcase['max_clause_size_simplepreproc'] = genIncidenceGraph(formula, graph)
        if testcase['max_var_occ_simplepreproc'] in (None, ""):
            with open(join(dirFormula, case + ".cnf"), "r") as formula:
                testcase['max_var_occ_simplepreproc'] = getMaxVarOcc(formula)
        lock.acquire()
        with open(summaryFile, 'a') as csvf:
            wr = csv.DictWriter(csvf, fieldnames=fieldnames)
            wr.writerow(testcase)
            csvf.flush()
        lock.release()


def checkPrimal(case, ppwidth, seed, dir):
    try:
        with open(join(dirGraphs, case + ".p.gr"), "w") as graph:
            with open(join(dir, case + ".cnf"), "r") as formula:
                genPrimalGraph(formula, graph)

        with open(join(dirDecomp, case + ".p.td"), "w") as decomp:
            with open(join(dirGraphs, case + ".p.gr"), "r") as graph:
                genTreeDecomp(graph, decomp, seed)

        with open(join(dirDecomp, case + ".p.td"), "r") as decomp:
            line = decomp.readline()
            ppwidth = int(line.split(" ")[3])
    except Exception as ex:
        print("        Error: checkPrimal - " + str(type(ex)) + " " + str(ex.args))
    return ppwidth


def checkIncidence(case, ipwidth, seed):
    try:
        with open(join(dirGraphs, case + ".i.gr"), "w") as graph:
            with open(join(dirFormula, case + ".cnf"), "r") as formula:
                genIncidenceGraph(formula, graph)

        with open(join(dirDecomp, case + ".i.td"), "w") as decomp:
            with open(join(dirGraphs, case + ".i.gr"), "r") as graph:
                genTreeDecomp(graph, decomp, seed)

        with open(join(dirDecomp, case + ".i.td"), "r") as decomp:
            line = decomp.readline()
            ipwidth = int(line.split(" ")[3])
    except Exception as ex:
        print("        Error: checkIncidence - " + str(type(ex)) + " " + str(ex.args))
    return ipwidth


with open(oldSummaryFile, 'r') as csvf:
    reader = csv.DictReader(csvf, fieldnames=fieldnames)

    numFiles = listdir(dirRaw)
    pool = ThreadPool(6)
    for i in reader:
        pool.apply_async(process, [i])
    pool.close()
    pool.join()
