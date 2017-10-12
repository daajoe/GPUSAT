import random
import subprocess
from os.path import join, isdir
import json
from os import makedirs

import time

tiny = True
small = True
medium = True
large = True
big = True
random.seed(1234)

maxNumModels = pow(10, 40)
numConnections = 2

numDigits = 4

minSolveTime = 0
maxSolveTime = 300
numGraphs = None
# minimal and maximal number of clauses
minNumClauses = None
maxNumClauses = None
# number of test cases to generate
numTestCases = None
# minimal size of a clause
minClauseSize = None
# maximal size of a clause
maxClauseSize = None
maxNumBags = None

# minimal and maximal tree width
minTreeWidth = None
maxTreeWidth = None

# minimal and maximal number of variables
minNumVariables = None
maxNumVariables = None

# number of test cases to generate
numSATTestCases = None
# number of test cases to generate
numUNSATTestCases = None

prefix = "test"
dirFormula = "./new_problems/formula"
dirGraphs = "./new_problems/graph"
dirDecomp = "./new_problems/decomposition"
dirReference = "./new_problems/reference"
dirResults = "./new_problems/results"

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


def genFolders():
    if not isdir(join(dirFormula, prefix)):
        makedirs(join(dirFormula, prefix))
    if not isdir(join(dirGraphs, prefix)):
        makedirs(join(dirGraphs, prefix))
    if not isdir(join(dirDecomp, prefix)):
        makedirs(join(dirDecomp, prefix))
    if not isdir(join(dirReference, prefix)):
        makedirs(join(dirReference, prefix))
    if not isdir(join(dirResults, prefix)):
        makedirs(join(dirResults, prefix))


def getModels(sol):
    if (len(sol) == 0):
        return pow(10, 400)
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
        i = 0
        while i < len(lines):
            if lines[i].startswith("Number of solutions"):
                if (lines[i].split('\t')[3] == "inf"):
                    return pow(2, 65)
                else:
                    return float(lines[i].split('\t')[3])
            i += 1
    elif sol.startswith("Solving"):
        i = 0
        lines = sol.split("\n")
        while i < len(lines):
            if "# solutions" in lines[i]:
                return int(lines[i + 1])
            i += 1
        return pow(10, 400)
    else:
        return pow(10, 400)


# check formula
def checkFormula(formula, resultFile):
    print("    check Formula")
    solvetime = 0
    start = time.time()
    # subprocess.call(["./clasp", "--outf=2", "-n", "0", "-q", formula], stdout=resultFile)
    # subprocess.call(["./sharpSAT", formula], stdout=resultFile)
    # subprocess.call(["./cachet", formula], stdout=resultFile)
    try:
        resultFile.write(subprocess.check_output(["./cachet", formula], timeout=300).decode('ascii'))
        solvetime = time.time() - start
    except subprocess.TimeoutExpired:
        return solvetime
    except subprocess.CalledProcessError:
        return solvetime
    return solvetime

# check formula
def checkFormulaWeight(formula, resultFile):
    print("    check Formula")
    solvetime = 0
    start = time.time()
    # subprocess.call(["./clasp", "--outf=2", "-n", "0", "-q", formula], stdout=resultFile)
    # subprocess.call(["./sharpSAT", formula], stdout=resultFile)
    # subprocess.call(["./cachet", formula], stdout=resultFile)
    try:
        resultFile.write(subprocess.check_output(["./cachet-wmc", formula], timeout=300).decode('ascii'))
        solvetime = time.time() - start
    except subprocess.TimeoutExpired:
        return solvetime
    except subprocess.CalledProcessError:
        return solvetime
    return solvetime

# generate tree decomposition
def genTreeDecomp(graph, decompFile):
    print("    gen Tree Decomp")
    # print("    command: " + "cat " + graph + " | ./htd_main " + "--opt " + "width " + "--iterations " + "30")
    with open(graph, "r") as infile:
        subprocess.call(["./htd_main", "--opt", "width", "-s", "1234"],
                        stdout=decompFile, stdin=infile)


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
            if line[0] != 'p' and line[0] != 'c' and line[0] != 's' and line[0] != 'w':
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


def genPrimGraph(formulaFile, graphFile):
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
            if line[0] != 'p' and line[0] != 'c' and line[0] != 's' and line[0] != 'w':
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


# generate graph
def genPrimalGraph1(resultFile):
    print("    gen primal graph")
    graphEdges = ""
    graph = {}
    numVars = random.randint(minNumVariables, maxNumVariables)
    for i in range(1, numVars + 1):
        graph[i] = set()
    for i in range(1, numVars + 1):
        clauseSize = random.randint(minClauseSize, numVars if (numVars < maxClauseSize) else maxClauseSize)
        varList = list(range(1, numVars + 1))
        random.shuffle(varList)
        b = 0
        while b < len(varList):
            if len(graph[varList[b]]) >= maxClauseSize or varList[b] == i:
                varList.pop(b)
            else:
                b += 1
        a = len(graph[i])
        while a < clauseSize and a < len(varList):
            idx = varList[a]
            graph[i].add(idx)
            graph[idx].add(i)
            a += 1
    for key in graph.keys():
        for node in graph[key]:
            if node > key:
                graphEdges += str(key) + " " + str(node) + "\n"
    graphString = "p tw " + str(numVars) + " " + str(len(graphEdges.split('\n')) - 1) + "\n" + graphEdges
    resultFile.write(graphString)
    return graph


def allConnected(graph, nodes):
    con = True
    for n in nodes:
        for a in nodes:
            if n != a:
                con = con and (n in graph[a]) and (a in graph[n])
    return con


# generate formula from primal graph
def toFormula(graph, resultFile):
    print("    gen formula")
    clauses = ""
    numClauses = 0
    for i in range(1, len(graph.keys())):
        while len(graph[i]) > 0:
            nodes = [i]
            for elem in graph[i]:
                nodes += [elem]
                if not allConnected(graph, nodes):
                    del nodes[len(nodes) - 1]
            numClauses += 1
            clauses += str(nodes[0] * random.choice([1, -1]))
            graph[i].discard(nodes[0])
            graph[nodes[0]].discard(i)
            for a in range(1, len(nodes)):
                graph[i].discard(nodes[a])
                graph[nodes[a]].discard(i)
                clauses += " " + str(nodes[a] * random.choice([1, -1]))
            clauses += " 0\n"
    clauses = "p cnf " + str(len(graph.keys())) + " " + str(numClauses) + "\n" + clauses
    resultFile.write(clauses)

# generate formula from primal graph
def toFormula_Weight(graph, resultFile):
    print("    gen formula")
    clauses = ""
    numClauses = 0
    for i in range(1, len(graph.keys())):
        while len(graph[i]) > 0:
            nodes = [i]
            for elem in graph[i]:
                nodes += [elem]
                if not allConnected(graph, nodes):
                    del nodes[len(nodes) - 1]
            numClauses += 1
            clauses += str(nodes[0] * random.choice([1, -1]))
            graph[i].discard(nodes[0])
            graph[nodes[0]].discard(i)
            for a in range(1, len(nodes)):
                graph[i].discard(nodes[a])
                graph[nodes[a]].discard(i)
                clauses += " " + str(nodes[a] * random.choice([1, -1]))
            clauses += " 0\n"
    weights = ""
    for i in graph.keys():
        weights+="w " + str(i) + " " + str(random.randint(1,9)/10) + "\n"
    clauses = "p cnf " + str(len(graph.keys())) + " " + str(numClauses) + "\n" + weights + clauses
    resultFile.write(clauses)

# generate formula
def genFormulaFromGraph():
    genFolders()
    numTry = 0
    numSAT = 0
    numUNSAT = 0
    while numUNSAT < numUNSATTestCases or numSAT < numSATTestCases:
        print(
            prefix + " " + str(numTry).zfill(numDigits) + "/" + str(numUNSATTestCases + numSATTestCases) + " SAT: " + str(
                numSAT) + " UNSAT: " + str(numUNSAT))
        with open(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
            graph = genPrimalGraph1(graphFile)
        with open(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), "w") as formulaFile:
            toFormula(graph, formulaFile)
        with open(join(join(dirReference, prefix), str(numTry).zfill(numDigits) + ""), "w") as resultFile:
            solveTime = checkFormula(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), resultFile)
        if (solveTime <= minSolveTime or solveTime >= maxSolveTime):
            print("    Time: " + str(solveTime))
            continue
        with open(join(join(dirReference, prefix), str(numTry).zfill(numDigits) + ""), "r") as resultFile:
            ref = resultFile.read()
            if (getModels(ref) == 0 and numUNSAT >= numUNSATTestCases) or (
                            getModels(ref) > 0 and numSAT >= numSATTestCases) or getModels(ref) >= maxNumModels:
                print("    num Models: " + str(getModels(ref)))
                continue
        with open(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
            with open(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), "r") as formulaFile:
                genIncidenceGraph(formulaFile, graphFile)
        with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "w") as decompFile:
            genTreeDecomp(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), decompFile)
        with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "r") as decompFile:
            line = decompFile.readline()
            if int(line.split(" ")[3]) < minTreeWidth or int(line.split(" ")[3]) > maxTreeWidth:
                print("    width: " + line.split(" ")[3])
                continue
            if maxNumBags != None and int(line.split(" ")[4]) > maxNumBags:
                print("    bags: " + line.split(" ")[4])
                continue
        with open(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
            with open(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), "r") as formulaFile:
                genPrimGraph(formulaFile, graphFile)
        with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "w") as decompFile:
            genTreeDecomp(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), decompFile)
        with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "r") as decompFile:
            line = decompFile.readline()
            if int(line.split(" ")[3]) < minTreeWidth or int(line.split(" ")[3]) > maxTreeWidth:
                print("    width: " + line.split(" ")[3])
                continue
            if maxNumBags != None and int(line.split(" ")[4]) > maxNumBags:
                print("    bags: " + line.split(" ")[4])
                continue

        if getModels(ref) == 0:
            numUNSAT += 1
        else:
            numSAT += 1
        numTry += 1


# generate graph
def genPrimalGraph_(numVars, start):
    graph = {}
    for i in range(1 + start, numVars + 1 + start):
        graph[i] = set()
    for i in range(1 + start, numVars + 1 + start):
        clauseSize = random.randint(minClauseSize, numVars if (numVars < maxClauseSize) else maxClauseSize)
        varList = list(range(1 + start, numVars + 1 + start))
        random.shuffle(varList)
        b = 0
        while b < len(varList):
            if len(graph[varList[b]]) >= maxClauseSize or varList[b] == i:
                varList.pop(b)
            else:
                b += 1
        a = len(graph[i])
        while a < clauseSize and a < len(varList):
            idx = varList[a]
            graph[i].add(idx)
            graph[idx].add(i)
            a += 1
    return graph


# generate formula
def genPrimalGraph2(graphFile):
    print("    gen primal graph")
    graphEdges = ""
    graph = {}
    graphs = []
    totalVars = 0
    for i in range(1, numGraphs):
        numVars = random.randint(minNumVariables, maxNumVariables)
        graphs.append(genPrimalGraph_(numVars, totalVars))
        totalVars += numVars
    last = graphs.pop()
    graph.update(last)
    for c in range(2, numGraphs):
        current = graphs.pop()
        for d in range(1, numConnections):
            a = random.randint(min(last), max(last))
            b = random.randint(min(current), max(current))
            last[a].add(b)
            current[b].add(a)
        last = current
        graph.update(last)
    for key in graph.keys():
        for node in graph[key]:
            if node > key:
                graphEdges += str(key) + " " + str(node) + "\n"
    graphString = "p tw " + str(totalVars) + " " + str(len(graphEdges.split('\n')) - 1) + "\n" + graphEdges
    graphFile.write(graphString)
    return graph


def genFormulaFromGraph2():
    genFolders()
    numTry = 0
    numSAT = 0
    numUNSAT = 0
    while numUNSAT < numUNSATTestCases or numSAT < numSATTestCases:
        with open("./log.txt", "a") as logfile:
            print(
                prefix + " " + str(numTry).zfill(numDigits) + "/" + str(
                    numUNSATTestCases + numSATTestCases) + " SAT: " + str(
                    numSAT) + " UNSAT: " + str(numUNSAT))
            with open(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
                graph = genPrimalGraph2(graphFile)
            with open(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), "w") as formulaFile:
                toFormula(graph, formulaFile)
            with open(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
                with open(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), "r") as formulaFile:
                    genIncidenceGraph(formulaFile, graphFile)
            with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "w") as decompFile:
                genTreeDecomp(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), decompFile)
            with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "r") as decompFile:
                line = decompFile.readline()
                if int(line.split(" ")[3]) < minTreeWidth or int(line.split(" ")[3]) > maxTreeWidth:
                    print("    width: " + line.split(" ")[3])
                    continue
                if maxNumBags != None and int(line.split(" ")[4]) > maxNumBags:
                    print("    bags: " + line.split(" ")[4])
                    continue
            with open(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
                with open(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), "r") as formulaFile:
                    genPrimGraph(formulaFile, graphFile)
            with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "w") as decompFile:
                genTreeDecomp(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), decompFile)
            with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "r") as decompFile:
                line = decompFile.readline()
                if int(line.split(" ")[3]) < minTreeWidth or int(line.split(" ")[3]) > maxTreeWidth:
                    print("    width: " + line.split(" ")[3])
                    continue
                if maxNumBags != None and int(line.split(" ")[4]) > maxNumBags:
                    print("    bags: " + line.split(" ")[4])
                    continue
            with open(join(join(dirReference, prefix), str(numTry).zfill(numDigits) + ""), "w") as resultFile:
                solveTime = checkFormula(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), resultFile)
            if (solveTime <= minSolveTime or solveTime >= maxSolveTime):
                print("    Time: " + str(solveTime))
                continue
            with open(join(join(dirReference, prefix), str(numTry).zfill(numDigits) + ""), "r") as resultFile:
                ref = resultFile.read()
                if (getModels(ref) > 0):
                    logfile.write("\nModels: " + str(getModels(ref)))
                if (getModels(ref) == 0 and numUNSAT >= numUNSATTestCases) or (
                                getModels(ref) > 0 and numSAT >= numSATTestCases) or getModels(ref) >= maxNumModels:
                    print("    Models: " + str(getModels(ref)))
                    continue

            if getModels(ref) == 0:
                numUNSAT += 1
            else:
                numSAT += 1
            numTry += 1

# generate formula
def genFormulaFromGraph_Weight():
    genFolders()
    numTry = 0
    numSAT = 0
    numUNSAT = 0
    while numUNSAT < numUNSATTestCases or numSAT < numSATTestCases:
        print(
            prefix + " " + str(numTry).zfill(numDigits) + "/" + str(numUNSATTestCases + numSATTestCases) + " SAT: " + str(
                numSAT) + " UNSAT: " + str(numUNSAT))
        with open(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
            graph = genPrimalGraph1(graphFile)
        with open(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), "w") as formulaFile:
            toFormula_Weight(graph, formulaFile)
        with open(join(join(dirReference, prefix), str(numTry).zfill(numDigits) + ""), "w") as resultFile:
            solveTime = checkFormulaWeight(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), resultFile)
        if (solveTime <= minSolveTime or solveTime >= maxSolveTime):
            print("    Time: " + str(solveTime))
            continue
        with open(join(join(dirReference, prefix), str(numTry).zfill(numDigits) + ""), "r") as resultFile:
            ref = resultFile.read()
            if (getModels(ref) == 0 and numUNSAT >= numUNSATTestCases) or (
                            getModels(ref) > 0 and numSAT >= numSATTestCases) or getModels(ref) >= maxNumModels:
                print("    num Models: " + str(getModels(ref)))
                continue
        with open(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
            with open(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), "r") as formulaFile:
                genIncidenceGraph(formulaFile, graphFile)
        with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "w") as decompFile:
            genTreeDecomp(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), decompFile)
        with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "r") as decompFile:
            line = decompFile.readline()
            if int(line.split(" ")[3]) < minTreeWidth or int(line.split(" ")[3]) > maxTreeWidth:
                print("    width: " + line.split(" ")[3])
                continue
            if maxNumBags != None and int(line.split(" ")[4]) > maxNumBags:
                print("    bags: " + line.split(" ")[4])
                continue
        with open(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
            with open(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), "r") as formulaFile:
                genPrimGraph(formulaFile, graphFile)
        with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "w") as decompFile:
            genTreeDecomp(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), decompFile)
        with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "r") as decompFile:
            line = decompFile.readline()
            if int(line.split(" ")[3]) < minTreeWidth or int(line.split(" ")[3]) > maxTreeWidth:
                print("    width: " + line.split(" ")[3])
                continue
            if maxNumBags != None and int(line.split(" ")[4]) > maxNumBags:
                print("    bags: " + line.split(" ")[4])
                continue

        if getModels(ref) == 0:
            numUNSAT += 1
        else:
            numSAT += 1
        numTry += 1

# generate graph
def getPrimalGraph(formula, resultFile, numVariables):
    print("    gen primal graph")
    graphEdges = ""
    graph = {}
    for i in range(1, numVariables + 1):
        graph[i] = set()
    for line in formula.splitlines():
        if line[0] != 'p':
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

    graphString = "p tw " + str(numVariables) + " " + str(len(graphEdges.split('\n')) - 1) + "\n" + graphEdges
    resultFile.write(graphString)


# generate formula
def genFormula():
    genFolders()
    numTry = 0
    numSAT = 0
    numUNSAT = 0
    while numUNSAT < numUNSATTestCases or numSAT < numSATTestCases:
        print(
            prefix + " " + str(numTry).zfill(numDigits) + "/" + str(
                numUNSATTestCases + numSATTestCases) + " SAT: " + str(
                numSAT) + " UNSAT: " + str(numUNSAT))
        numClauses = random.randint(minNumClauses, maxNumClauses)
        numVars = random.randint(minNumVariables, maxNumVariables)
        formula = "p cnf " + str(numVars) + " " + str(numClauses - 1) + "\n"
        for clause in range(1, numClauses):
            clauseSize = random.randint(minClauseSize, numVars if (numVars < maxClauseSize) else maxClauseSize)
            varList = list(range(1, numVars + 1))
            random.shuffle(varList)
            formula = formula + str(varList[0] * random.choice([1, -1]))
            for var in range(1, clauseSize):
                formula = formula + " " + str(varList[var] * random.choice([1, -1]))
            formula = formula + " 0\n"
        with open(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), "w") as formulaFile:
            formulaFile.write(formula)
        with open(join(join(dirReference, prefix), str(numTry).zfill(numDigits) + ""), "w") as resultFile:
            solveTime = checkFormula(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), resultFile)
        if (solveTime <= minSolveTime or solveTime >= maxSolveTime):
            print("    Time: " + str(solveTime))
            continue
        with open(join(join(dirReference, prefix), str(numTry).zfill(numDigits) + ""), "r") as resultFile:
            ref = resultFile.read()
            if (getModels(ref) == 0 and numUNSAT >= numUNSATTestCases) or (
                            getModels(ref) > 0 and numSAT >= numSATTestCases) or getModels(ref) >= maxNumModels:
                print("    num Models: " + str(getModels(ref)))
                continue
        with open(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
            getPrimalGraph(formula, graphFile, numVars)
        with open(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
            with open(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), "r") as formulaFile:
                genIncidenceGraph(formulaFile, graphFile)
        with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "w") as decompFile:
            genTreeDecomp(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), decompFile)
        with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "r") as decompFile:
            line = decompFile.readline()
            if int(line.split(" ")[3]) < minTreeWidth or int(line.split(" ")[3]) > maxTreeWidth:
                print("    width: " + line.split(" ")[3])
                continue
            if maxNumBags != None and int(line.split(" ")[4]) > maxNumBags:
                print("    bags: " + line.split(" ")[4])
                continue
        with open(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
            with open(join(join(dirFormula, prefix), str(numTry).zfill(numDigits) + ".cnf"), "r") as formulaFile:
                genPrimGraph(formulaFile, graphFile)
        with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "w") as decompFile:
            genTreeDecomp(join(join(dirGraphs, prefix), str(numTry).zfill(numDigits) + ".gr"), decompFile)
        with open(join(join(dirDecomp, prefix), str(numTry).zfill(numDigits) + ".td"), "r") as decompFile:
            line = decompFile.readline()
            if int(line.split(" ")[3]) < minTreeWidth or int(line.split(" ")[3]) > maxTreeWidth:
                print("    width: " + line.split(" ")[3])
                continue
            if maxNumBags != None and int(line.split(" ")[4]) > maxNumBags:
                print("    bags: " + line.split(" ")[4])
                continue

        if getModels(ref) == 0:
            numUNSAT += 1
        else:
            numSAT += 1
        numTry += 1


######################################################################################
######################################################################################

maxNumBags = 8
##########
# Tiny 0 #
##########
if tiny and False:
    minClauseSize = 1
    maxClauseSize = 4

    minTreeWidth = 1
    maxTreeWidth = 4

    minNumVariables = 1
    maxNumVariables = 10

    numSATTestCases = 500
    numUNSATTestCases = 0

    prefix = "a_00_test"

    genFormulaFromGraph()

##########
# Tiny 1 #
##########
if tiny and False:
    minClauseSize = 0
    maxClauseSize = 4

    minTreeWidth = 1
    maxTreeWidth = 4

    minNumVariables = 1
    maxNumVariables = 10

    numSATTestCases = 450
    numUNSATTestCases = 50

    minNumClauses = 1
    maxNumClauses = 5

    prefix = "b_00_test"

    genFormula()

maxNumBags = 30
##########
# Tiny 2 #
##########
if tiny and False:
    numGraphs = 5

    minClauseSize = 1
    maxClauseSize = 4

    minTreeWidth = 1
    maxTreeWidth = 4

    minNumVariables = 1
    maxNumVariables = 10

    numSATTestCases = 500
    numUNSATTestCases = 0

    prefix = "c_00_test"

    genFormulaFromGraph2()

##########
# Tiny 0 #
##########
if tiny:
    minClauseSize = 1
    maxClauseSize = 4

    minTreeWidth = 1
    maxTreeWidth = 4

    minNumVariables = 1
    maxNumVariables = 10

    numSATTestCases = 500
    numUNSATTestCases = 0

    prefix = "d_00_test"

    genFormulaFromGraph_Weight()

######################################################################################
######################################################################################

maxNumBags = None
###########
# Small 0 #
###########
if small and False:
    minClauseSize = 1
    maxClauseSize = 11

    minTreeWidth = 4
    maxTreeWidth = 11

    minNumVariables = 10
    maxNumVariables = 25

    numSATTestCases = 170
    numUNSATTestCases = 30

    prefix = "a_01_test"

    genFormulaFromGraph()

###########
# Small 1 #
###########
if small and False:
    minClauseSize = 3
    maxClauseSize = 8

    minTreeWidth = 4
    maxTreeWidth = 11

    minNumVariables = 10
    maxNumVariables = 25

    numSATTestCases = 200
    numUNSATTestCases = 0

    minNumClauses = 5
    maxNumClauses = 15

    prefix = "b_01_test"

    genFormula()

###########
# Small 2 #
###########
if small and False:
    numGraphs = 5

    minClauseSize = 1
    maxClauseSize = 11

    minTreeWidth = 4
    maxTreeWidth = 11

    minNumVariables = 10
    maxNumVariables = 25

    numSATTestCases = 100
    numUNSATTestCases = 100

    prefix = "c_01_test"

    genFormulaFromGraph2()

###########
# Small 0 #
###########
if small:
    minClauseSize = 1
    maxClauseSize = 11

    minTreeWidth = 4
    maxTreeWidth = 11

    minNumVariables = 10
    maxNumVariables = 25

    numSATTestCases = 170
    numUNSATTestCases = 30

    prefix = "d_01_test"

    genFormulaFromGraph_Weight()

######################################################################################
######################################################################################

############
# Medium 0 #
############
if medium and False:
    minClauseSize = 8
    maxClauseSize = 15

    minTreeWidth = 12
    maxTreeWidth = 16

    minNumVariables = 20
    maxNumVariables = 50

    numSATTestCases = 130
    numUNSATTestCases = 70

    prefix = "a_02_test"

    genFormulaFromGraph()

############
# Medium 1 #
############
if medium and False:
    minClauseSize = 8
    maxClauseSize = 16

    minTreeWidth = 6
    maxTreeWidth = 16

    minNumVariables = 15
    maxNumVariables = 25

    numSATTestCases = 200
    numUNSATTestCases = 0

    minNumClauses = 5
    maxNumClauses = 15

    prefix = "b_02_test"

    genFormula()

############
# Medium 2 #
############
if medium and False:
    numGraphs = 5

    minClauseSize = 3
    maxClauseSize = 8

    minTreeWidth = 12
    maxTreeWidth = 16

    minNumVariables = 20
    maxNumVariables = 50

    numSATTestCases = 0
    numUNSATTestCases = 200

    prefix = "c_02_test"

    genFormulaFromGraph2()

############
# Medium 0 #
############
if medium:
    minClauseSize = 8
    maxClauseSize = 15

    minTreeWidth = 12
    maxTreeWidth = 16

    minNumVariables = 20
    maxNumVariables = 50

    numSATTestCases = 130
    numUNSATTestCases = 70

    prefix = "d_02_test"

    genFormulaFromGraph_Weight()

######################################################################################
######################################################################################

###########
# Large 0 #
###########
if large and False:
    minClauseSize = 1
    maxClauseSize = 5

    minTreeWidth = 17
    maxTreeWidth = 22

    minNumVariables = 70
    maxNumVariables = 90

    numSATTestCases = 20
    numUNSATTestCases = 180

    prefix = "a_03_test"

    genFormulaFromGraph()

###########
# Large 1 #
###########
if large and False:
    minClauseSize = 6
    maxClauseSize = 10

    minTreeWidth = 6
    maxTreeWidth = 22

    minNumVariables = 40
    maxNumVariables = 60

    numSATTestCases = 200
    numUNSATTestCases = 0

    minNumClauses = 12
    maxNumClauses = 20

    prefix = "b_03_test"

    genFormula()

###########
# Large 2 #
###########
if large and False:
    numGraphs = 3

    minClauseSize = 1
    maxClauseSize = 5

    minTreeWidth = 17
    maxTreeWidth = 22

    minNumVariables = 70
    maxNumVariables = 90

    numSATTestCases = 20
    numUNSATTestCases = 180

    prefix = "c_03_test"

    genFormulaFromGraph2()

###########
# Large 0 #
###########
if large:
    minClauseSize = 1
    maxClauseSize = 5

    minTreeWidth = 17
    maxTreeWidth = 22

    minNumVariables = 70
    maxNumVariables = 90

    numSATTestCases = 20
    numUNSATTestCases = 180

    prefix = "d_03_test"

    genFormulaFromGraph_Weight()

######################################################################################
######################################################################################

###########
# big 0 #
###########
if big and False:
    minClauseSize = 1
    maxClauseSize = 5

    minTreeWidth = 23
    maxTreeWidth = 30

    minNumVariables = 100
    maxNumVariables = 120

    numSATTestCases = 20
    numUNSATTestCases = 180

    prefix = "a_04_test"

    genFormulaFromGraph()

###########
# big 1 #
###########
if big and False:
    minClauseSize = 6
    maxClauseSize = 10

    minTreeWidth = 6
    maxTreeWidth = 30

    minNumVariables = 60
    maxNumVariables = 80

    numSATTestCases = 200
    numUNSATTestCases = 0

    minNumClauses = 12
    maxNumClauses = 20

    prefix = "b_04_test"

    genFormula()

###########
# big 2 #
###########
if big and False:
    numGraphs = 3

    minClauseSize = 1
    maxClauseSize = 5

    minTreeWidth = 23
    maxTreeWidth = 30

    minNumVariables = 100
    maxNumVariables = 120

    numSATTestCases = 20
    numUNSATTestCases = 180

    prefix = "c_04_test"

    genFormulaFromGraph2()

###########
# big 0 #
###########
if big:
    minClauseSize = 1
    maxClauseSize = 5

    minTreeWidth = 23
    maxTreeWidth = 30

    minNumVariables = 100
    maxNumVariables = 120

    numSATTestCases = 20
    numUNSATTestCases = 180

    prefix = "d_04_test"

    genFormulaFromGraph_Weight()

######################################################################################
######################################################################################

