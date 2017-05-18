import random
import subprocess
from sets import Set
from os.path import join, isdir

from os import makedirs

# todo minimal size of a clause
minClauseSize = 0
# todo maximal size of a clause
maxClauseSize = 0

# todo ratio between positive and negative variables
positiveRatio = 0.5

# todo maximal tree widht
maxTreeWidth = 18

# minimal and maximal number of clauses
minNumClauses = 1
maxNumClauses = 12

# minimal and maximal number of variables
minNumVariables = 2
maxNumVariables = 18

# number of test cases to generate
numTestCases = 2000

prefix = "test_"
dirFormula = "./problems/formula"
dirGraphs = "./problems/graph"
dirDecomp = "./problems/decomposition"
dirReference = "./problems/reference"
dirResults = "./problems/results"

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
    subprocess.call(["clasp", "--outf=2", "-n", "0", formula], stdout=resultFile)


# generate graph
def genPrimalGraph(formula, resultFile, numVariables):
    graphEdges = ""
    graph = {}
    for i in range(1, numVariables + 1):
        graph[i] = Set()
    for line in formula.splitlines():
        if (line[0] != 'p'):
            for node in line.split():
                if (int(node) != 0):
                    for node2 in line.split():
                        if (int(node2) != 0):
                            graph[abs(int(node))].add(abs(int(node2)))
                            graph[abs(int(node2))].add(abs(int(node)))
    for key in graph.keys():
        for node in graph[key]:
            if (node > key):
                graphEdges += str(key) + " " + str(node) + "\n"

    graphString = "p tw " + str(numVariables) + " " + str(len(graphEdges.split('\n')) - 1) + "\n" + graphEdges
    resultFile.write(graphString)


# generate tree decomposition
def genTreeDecomp(graph, decompFile):
    subprocess.call(["htd_main", "-s", "1234", "--opt", "width", "--instance", graph], stdout=decompFile)


# generate formula
for numTry in range(0, numTestCases):
    numClauses = random.randint(minNumClauses, maxNumClauses)
    numVars = random.randint(minNumVariables, maxNumVariables)
    formula = "p cnf " + str(numVars) + " " + str(numClauses - 1) + "\n"
    for clause in range(1, numClauses):
        clauseSize = random.randint(1, numVars)
        varList = range(1, numVars + 1)
        random.shuffle(varList)
        formula = formula + str(varList[0] * random.choice([1, -1]))
        for var in range(1, clauseSize):
            formula = formula + " " + str(varList[var] * random.choice([1, -1]))
        formula = formula + " 0\n"
    print "file: " + str(numTry)
    with open(join(dirFormula, prefix + str(numTry) + ".cnf"), "w") as formulaFile:
        formulaFile.write(formula)
    with open(join(dirReference, prefix + str(numTry) + ""), "w") as resultFile:
        checkFormula(join(dirFormula, prefix + str(numTry) + ".cnf"), resultFile)
    with open(join(dirGraphs, prefix + str(numTry) + ".gr"), "w") as graphFile:
        genPrimalGraph(formula, graphFile, numVars)
    with open(join(dirDecomp, prefix + str(numTry) + ".td"), "w") as decompFile:
        genTreeDecomp(join(dirGraphs, prefix + str(numTry) + ".gr"), decompFile)
