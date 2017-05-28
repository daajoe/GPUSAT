import random
import subprocess
from sets import Set
from os.path import join, isdir

from os import makedirs

# minimal size of a clause
minClauseSize = 0
# maximal size of a clause
maxClauseSize = 0

# ratio between positive and negative variables
positiveRatio = 0.5

# minimal and maximal tree width
minTreeWidth = 18
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
    print "    check Formula"
    # subprocess.call(["clasp", "--outf=2", "-n", "0", "-q", formula], stdout=resultFile)
    subprocess.call(["./sharpSAT", formula], stdout=resultFile)


# generate graph
def genPrimalGraph(formula, resultFile, numVariables):
    print "    gen primal graph"
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
    print "    gen Tree Decomp"
    subprocess.call(["htd_main", "-s", "1234", "--opt", "width", "--instance", graph], stdout=decompFile)


# generate formula
def genFormula():
    numTry = 0
    while numTry < numTestCases:
        print prefix + str(1 + numTry) + "/" + str(numTestCases)
        numClauses = random.randint(minNumClauses, maxNumClauses)
        numVars = random.randint(minNumVariables, maxNumVariables)
        formula = "p cnf " + str(numVars) + " " + str(numClauses - 1) + "\n"
        for clause in range(1, numClauses):
            clauseSize = random.randint(minClauseSize, numVars if (numVars < maxClauseSize) else maxClauseSize)
            varList = range(1, numVars + 1)
            random.shuffle(varList)
            formula = formula + str(varList[0] * random.choice([1, -1]))
            for var in range(1, clauseSize):
                formula = formula + " " + str(varList[var] * random.choice([1, -1]))
            formula = formula + " 0\n"
        with open(join(dirFormula, prefix + str(numTry) + ".cnf"), "w") as formulaFile:
            formulaFile.write(formula)
        with open(join(dirGraphs, prefix + str(numTry) + ".gr"), "w") as graphFile:
            genPrimalGraph(formula, graphFile, numVars)
        with open(join(dirDecomp, prefix + str(numTry) + ".td"), "w") as decompFile:
            genTreeDecomp(join(dirGraphs, prefix + str(numTry) + ".gr"), decompFile)
        with open(join(dirDecomp, prefix + str(numTry) + ".td"), "r") as decompFile:
            line = decompFile.readline()
            if int(line.split(" ")[3]) < minTreeWidth or int(line.split(" ")[3]) > maxTreeWidth:
                continue
        with open(join(dirReference, prefix + str(numTry) + ""), "w") as resultFile:
            checkFormula(join(dirFormula, prefix + str(numTry) + ".cnf"), resultFile)
        numTry += 1


#################
##   Tests 1   ##
#################

# minimal size of a clause
minClauseSize = 2
# maximal size of a clause
maxClauseSize = 18

# ratio between positive and negative variables
positiveRatio = 0.5

# minimal and maximal tree widht
minTreeWidth = 1
maxTreeWidth = 20

# minimal and maximal number of clauses
minNumClauses = 4
maxNumClauses = 90

# minimal and maximal number of variables
minNumVariables = 2
maxNumVariables = 30

# number of test cases to generate
numTestCases = 1000

# prefix of the test cases
prefix = "1_test_"

genFormula()

#################
##   Tests 2   ##
#################

# minimal size of a clause
minClauseSize = 3
# maximal size of a clause
maxClauseSize = 10

# ratio between positive and negative variables
positiveRatio = 0.5

# minimal and maximal tree widht
minTreeWidth = 10
maxTreeWidth = 20

# minimal and maximal number of clauses
minNumClauses = 5
maxNumClauses = 15

# minimal and maximal number of variables
minNumVariables = 16
maxNumVariables = 25

# number of test cases to generate
numTestCases = 200

# prefix of the test cases
prefix = "2_test_"

genFormula()

#################
##   Tests 3   ##
#################

# minimal size of a clause
minClauseSize = 5
# maximal size of a clause
maxClauseSize = 15

# ratio between positive and negative variables
positiveRatio = 0.5

# minimal and maximal tree widht
minTreeWidth = 10
maxTreeWidth = 20

# minimal and maximal number of clauses
minNumClauses = 10
maxNumClauses = 20

# minimal and maximal number of variables
minNumVariables = 20
maxNumVariables = 30

# number of test cases to generate
numTestCases = 200

# prefix of the test cases
prefix = "3_test_"

genFormula()

#################
##   Tests 4   ##
#################

# minimal size of a clause
minClauseSize = 5
# maximal size of a clause
maxClauseSize = 15

# ratio between positive and negative variables
positiveRatio = 0.5

# minimal and maximal tree widht
minTreeWidth = 10
maxTreeWidth = 20

# minimal and maximal number of clauses
minNumClauses = 30
maxNumClauses = 80

# minimal and maximal number of variables
minNumVariables = 20
maxNumVariables = 30

# number of test cases to generate
numTestCases = 200

# prefix of the test cases
prefix = "4_test_"

genFormula()
