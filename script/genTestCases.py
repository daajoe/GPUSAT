import random
import subprocess
from os.path import join, isdir
import json
from os import makedirs
from sets import Set

small = True

numDigits = 4

# minimal size of a clause
minClauseSize = None
# maximal size of a clause
maxClauseSize = None

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
    print("    check Formula")
    subprocess.call(["clasp", "--outf=2", "-n", "0", "-q", formula], stdout=resultFile)
    # subprocess.call(["./sharpSAT", formula], stdout=resultFile)


# generate graph
def genPrimalGraph(resultFile):
    print    ("    gen primal graph")
    graphEdges = ""
    graph = {}
    numVars = random.randint(minNumVariables, maxNumVariables)
    for i in range(1, numVars + 1):
        graph[i] = Set()
    for i in range(1, numVars + 1):
        clauseSize = random.randint(minClauseSize, numVars if (numVars < maxClauseSize) else maxClauseSize)
        varList = range(1, numVars + 1)
        random.shuffle(varList)
        b = 0
        while b < len(varList):
            if (len(graph[varList[b]]) >= maxClauseSize or varList[b] == i):
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
            if (node > key):
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
            l = len(graph[i])
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


# generate tree decomposition
def genTreeDecomp(graph, decompFile):
    print("    gen Tree Decomp")
    subprocess.call(["htd_main", "--opt", "width", "--iterations", "30", "--instance", graph],
                    stdout=decompFile)


def getModels(sol):
    if sol[0] == "{":
        return json.loads(sol)['Models']['Number']
    else:
        i = 0
        lines = sol.split("\n")
        while i < len(lines):
            if "# solutions" in lines[i]:
                return int(lines[i + 1])
            i += 1


# generate formula
def genFormula():
    numTry = 0
    numSAT = 0
    numUNSAT = 0
    while numUNSAT < numUNSATTestCases or numSAT < numSATTestCases:
        print(prefix + str(numTry).zfill(numDigits) + "/" + str(numUNSATTestCases + numSATTestCases) + " SAT: " + str(
            numSAT) + " UNSAT: " + str(numUNSAT))
        with open(join(dirGraphs, prefix + str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
            graph = genPrimalGraph(graphFile)
        with open(join(dirDecomp, prefix + str(numTry).zfill(numDigits) + ".td"), "w") as decompFile:
            genTreeDecomp(join(dirGraphs, prefix + str(numTry).zfill(numDigits) + ".gr"), decompFile)
        with open(join(dirDecomp, prefix + str(numTry).zfill(numDigits) + ".td"), "r") as decompFile:
            line = decompFile.readline()
            if int(line.split(" ")[3]) < minTreeWidth or int(line.split(" ")[3]) > maxTreeWidth:
                continue
        with open(join(dirFormula, prefix + str(numTry).zfill(numDigits) + ".cnf"), "w") as formulaFile:
            toFormula(graph, formulaFile)
        with open(join(dirReference, prefix + str(numTry).zfill(numDigits) + ""), "w") as resultFile:
            checkFormula(join(dirFormula, prefix + str(numTry).zfill(numDigits) + ".cnf"), resultFile)
        with open(join(dirReference, prefix + str(numTry).zfill(numDigits) + ""), "r") as resultFile:
            ref = resultFile.read()
            if (getModels(ref) == 0 and numUNSAT >= numUNSATTestCases) or (
                            getModels(ref) > 0 and numSAT >= numSATTestCases):
                continue

            if (getModels(ref) == 0):
                numUNSAT += 1
            else:
                numSAT += 1
        numTry += 1


#################
##   Tests 0   ##
#################

# minimal size of a clause
minClauseSize = 0
# maximal size of a clause
maxClauseSize = 2

# minimal and maximal tree widht
minTreeWidth = 1
maxTreeWidth = 3

# minimal and maximal number of variables
minNumVariables = 1
maxNumVariables = 5

# number of test cases to generate
numSATTestCases = 1000
numUNSATTestCases = 0

# prefix of the test cases
prefix = "a_0_test_"

# genFormula()

#################
##   Tests 1   ##
#################

# minimal size of a clause
minClauseSize = 3
# maximal size of a clause
maxClauseSize = 5

# minimal and maximal tree widht
minTreeWidth = 3
maxTreeWidth = 5

# minimal and maximal number of variables
minNumVariables = 3
maxNumVariables = 10

# number of test cases to generate
numSATTestCases = 900
numUNSATTestCases = 100

# prefix of the test cases
prefix = "a_1_test_"

# genFormula()

#################
##   Tests 2   ##
#################

# minimal size of a clause
minClauseSize = 1
# maximal size of a clause
maxClauseSize = 10

# minimal and maximal tree widht
minTreeWidth = 6
maxTreeWidth = 10

# minimal and maximal number of variables
minNumVariables = 6
maxNumVariables = 30

# number of test cases to generate
numSATTestCases = 800
numUNSATTestCases = 200

# prefix of the test cases
prefix = "a_2_test_"

# genFormula()

#################
##   Tests 3   ##
#################

# minimal size of a clause
minClauseSize = 1
# maximal size of a clause
maxClauseSize = 10

# minimal and maximal tree widht
minTreeWidth = 11
maxTreeWidth = 15

# minimal and maximal number of variables
minNumVariables = 11
maxNumVariables = 40

# number of test cases to generate
numSATTestCases = 800
numUNSATTestCases = 200

# prefix of the test cases
prefix = "a_3_test_"

# genFormula()

#################
##   Tests 4   ##
#################

# minimal size of a clause
minClauseSize = 1
# maximal size of a clause
maxClauseSize = 10

# minimal and maximal tree widht
minTreeWidth = 16
maxTreeWidth = 19

# minimal and maximal number of variables
minNumVariables = 16
maxNumVariables = 50

# number of test cases to generate
numSATTestCases = 800
numUNSATTestCases = 200

# prefix of the test cases
prefix = "a_4_test_"

# genFormula()

#################
##   Tests 5   ##
#################

if not small:
    # minimal size of a clause
    minClauseSize = 3
    # maximal size of a clause
    maxClauseSize = 7

    # minimal and maximal tree widht
    minTreeWidth = 20
    maxTreeWidth = 23

    # minimal and maximal number of variables
    minNumVariables = 40
    maxNumVariables = 90

    # number of test cases to generate
    numSATTestCases = 800
    numUNSATTestCases = 200

    # prefix of the test cases
    prefix = "a_5_test_"

    genFormula()

#################
##   Tests 0   ##
#################

# minimal size of a clause
minClauseSize = 1
# maximal size of a clause
maxClauseSize = 10

# minimal and maximal tree widht
minTreeWidth = 1
maxTreeWidth = 8

# minimal and maximal number of variables
minNumVariables = 2
maxNumVariables = 10

# number of test cases to generate
numSATTestCases = 990
numUNSATTestCases = 10

# prefix of the test cases
prefix = "b_00_test_"

genFormula()

#################
##   Tests 1   ##
#################

# minimal size of a clause
minClauseSize = 1
# maximal size of a clause
maxClauseSize = 18

# minimal and maximal tree widht
minTreeWidth = 1
maxTreeWidth = 19

# minimal and maximal number of variables
minNumVariables = 2
maxNumVariables = 30

# number of test cases to generate
numSATTestCases = 800
numUNSATTestCases = 200

# prefix of the test cases
prefix = "b_01_test_"

genFormula()

#################
##   Tests 2   ##
#################

# minimal size of a clause
minClauseSize = 3
# maximal size of a clause
maxClauseSize = 10

# minimal and maximal tree widht
minTreeWidth = 10
maxTreeWidth = 19

# minimal and maximal number of variables
minNumVariables = 16
maxNumVariables = 25

# number of test cases to generate
numSATTestCases = 180
numUNSATTestCases = 20

# prefix of the test cases
prefix = "b_02_test_"

genFormula()

#################
##   Tests 3   ##
#################

# minimal size of a clause
minClauseSize = 5
# maximal size of a clause
maxClauseSize = 15

# minimal and maximal tree widht
minTreeWidth = 10
maxTreeWidth = 19

# minimal and maximal number of variables
minNumVariables = 20
maxNumVariables = 30

# number of test cases to generate
numSATTestCases = 180
numUNSATTestCases = 20

# prefix of the test cases
prefix = "b_03_test_"

genFormula()

#################
##   Tests 4   ##
#################

# minimal size of a clause
minClauseSize = 5
# maximal size of a clause
maxClauseSize = 15

# minimal and maximal tree widht
minTreeWidth = 10
maxTreeWidth = 19

# minimal and maximal number of variables
minNumVariables = 20
maxNumVariables = 30

# number of test cases to generate
numSATTestCases = 180
numUNSATTestCases = 20

# prefix of the test cases
prefix = "b_04_test_"

genFormula()

#################
##   Tests 5   ##
#################

# minimal size of a clause
minClauseSize = 2
# maximal size of a clause
maxClauseSize = 6

# minimal and maximal tree widht
minTreeWidth = 10
maxTreeWidth = 19

# minimal and maximal number of variables
minNumVariables = 50
maxNumVariables = 90

# number of test cases to generate
numSATTestCases = 80
numUNSATTestCases = 20

# prefix of the test cases
prefix = "b_05_test_"

genFormula()

#################
##   Tests 6   ##
#################

# minimal size of a clause
minClauseSize = 2
# maximal size of a clause
maxClauseSize = 4

# minimal and maximal tree widht
minTreeWidth = 10
maxTreeWidth = 19

# minimal and maximal number of variables
minNumVariables = 90
maxNumVariables = 110

# number of test cases to generate
numSATTestCases = 10
numUNSATTestCases = 5

# prefix of the test cases
prefix = "b_06_test_"

genFormula()

#################
##   Tests 7   ##
#################

if not small:
    # minimal size of a clause
    minClauseSize = 2
    # maximal size of a clause
    maxClauseSize = 6

    # minimal and maximal tree widht
    minTreeWidth = 20
    maxTreeWidth = 20

    # minimal and maximal number of variables
    minNumVariables = 50
    maxNumVariables = 90

    # number of test cases to generate
    numSATTestCases = 80
    numUNSATTestCases = 20

    # prefix of the test cases
    prefix = "b_07_test_"

    genFormula()

#################
##   Tests 8   ##
#################

if not small:
    # minimal size of a clause
    minClauseSize = 2
    # maximal size of a clause
    maxClauseSize = 6

    # minimal and maximal tree widht
    minTreeWidth = 21
    maxTreeWidth = 21

    # minimal and maximal number of variables
    minNumVariables = 50
    maxNumVariables = 90

    # number of test cases to generate
    numSATTestCases = 80
    numUNSATTestCases = 20

    # prefix of the test cases
    prefix = "b_08_test_"

    genFormula()

#################
##   Tests 9   ##
#################

if not small:
    # minimal size of a clause
    minClauseSize = 2
    # maximal size of a clause
    maxClauseSize = 5

    # minimal and maximal tree widht
    minTreeWidth = 22
    maxTreeWidth = 22

    # minimal and maximal number of variables
    minNumVariables = 50
    maxNumVariables = 90

    # number of test cases to generate
    numSATTestCases = 80
    numUNSATTestCases = 20

    # prefix of the test cases
    prefix = "b_09_test_"

    genFormula()

#################
##   Tests 10  ##
#################

if not small:
    # minimal size of a clause
    minClauseSize = 2
    # maximal size of a clause
    maxClauseSize = 5

    # minimal and maximal tree widht
    minTreeWidth = 23
    maxTreeWidth = 23

    # minimal and maximal number of variables
    minNumVariables = 50
    maxNumVariables = 90

    # number of test cases to generate
    numSATTestCases = 80
    numUNSATTestCases = 20

    # prefix of the test cases
    prefix = "b_10_test_"

    genFormula()

###################################################################
###################################################################


# ratio between positive and negative variables
positiveRatio = None
# minimal and maximal number of clauses
minNumClauses = None
maxNumClauses = None
# number of test cases to generate
numTestCases = None


# generate graph
def getPrimalGraph(formula, resultFile, numVariables):
    print ("    gen primal graph")
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


# generate formula
def genFormula():
    numTry = 0
    while numTry < numTestCases:
        print (prefix + str(numTry).zfill(numDigits) + "/" + str(numTestCases))
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
        with open(join(dirFormula, prefix + str(numTry).zfill(numDigits) + ".cnf"), "w") as formulaFile:
            formulaFile.write(formula)
        with open(join(dirGraphs, prefix + str(numTry).zfill(numDigits) + ".gr"), "w") as graphFile:
            getPrimalGraph(formula, graphFile, numVars)
        with open(join(dirDecomp, prefix + str(numTry).zfill(numDigits) + ".td"), "w") as decompFile:
            genTreeDecomp(join(dirGraphs, prefix + str(numTry).zfill(numDigits) + ".gr"), decompFile)
        with open(join(dirDecomp, prefix + str(numTry).zfill(numDigits) + ".td"), "r") as decompFile:
            line = decompFile.readline()
            if int(line.split(" ")[3]) < minTreeWidth or int(line.split(" ")[3]) > maxTreeWidth:
                continue
        with open(join(dirReference, prefix + str(numTry).zfill(numDigits) + ""), "w") as resultFile:
            checkFormula(join(dirFormula, prefix + str(numTry).zfill(numDigits) + ".cnf"), resultFile)
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
maxTreeWidth = 19

# minimal and maximal number of clauses
minNumClauses = 6
maxNumClauses = 30

# minimal and maximal number of variables
minNumVariables = 2
maxNumVariables = 20

# number of test cases to generate
numTestCases = 1000

# prefix of the test cases
prefix = "c_1_test_"

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
maxTreeWidth = 19

# minimal and maximal number of clauses
minNumClauses = 5
maxNumClauses = 15

# minimal and maximal number of variables
minNumVariables = 16
maxNumVariables = 25

# number of test cases to generate
numTestCases = 400

# prefix of the test cases
prefix = "c_2_test_"

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
maxTreeWidth = 19

# minimal and maximal number of clauses
minNumClauses = 10
maxNumClauses = 20

# minimal and maximal number of variables
minNumVariables = 20
maxNumVariables = 30

# number of test cases to generate
numTestCases = 300

# prefix of the test cases
prefix = "c_3_test_"

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
maxTreeWidth = 19

# minimal and maximal number of clauses
minNumClauses = 30
maxNumClauses = 80

# minimal and maximal number of variables
minNumVariables = 20
maxNumVariables = 30

# number of test cases to generate
numTestCases = 300

# prefix of the test cases
prefix = "c_4_test_"

genFormula()
