import json
import re
import subprocess
import time

from subprocess import check_output

from os import remove


def genPrimalGraph(formula):
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
            for lit in [int(x) for x in line.split() if len(x) != 0]:
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
                        if ainode < ainode2:
                            graph |= {(ainode, ainode2)}
                        elif ainode > ainode2:
                            graph |= {(ainode2, ainode)}
                        if len(graph) > 50000000:
                            raise Exception('Primal Oversize')

    graphString = ("p tw " + str(numVariables) + " " + str(len(graph)))
    for node in graph:
        graphString += ("\n" + str(node[0]) + " " + str(node[1]))
    return graphString


# generate tree decomposition
def genTreeDecomp(graphFile, decompFile):
    subprocess.call(["./htd_main", "--opt", "width", "-s", "1234"], stdout=decompFile, stdin=graphFile)


def getBestTreeDecomp(graphPath, decompPath, numTries):
    currentWidth = -1
    try:
        with open(decompPath, "w") as decompFile:
            with open(graphPath, "r")as graphFile:
                subprocess.call(["./htd_main", "--opt", "width"], timeout=300, stdout=decompFile, stdin=graphFile)
        with open(decompPath, "r")as currentDecomp:
            decomp = currentDecomp.read()
            currentWidth = int([x for x in decomp.splitlines() if len(x) > 0 and x[0] == 's'][0].split()[3]) - 1
        for i in range(1, numTries):
            with open(decompPath + str(i), "w") as decompFile:
                with open(graphPath, "r")as graphFile:
                    subprocess.call(["./htd_main", "--opt", "width"], timeout=300, stdout=decompFile, stdin=graphFile)
            with open(decompPath + str(i), "r")as currentDecomp:
                decomp = currentDecomp.read()
            remove(decompPath + str(i))
            width = int([x for x in decomp.splitlines() if len(x) > 0 and x[0] == 's'][0].split()[3]) - 1
            if width < currentWidth:
                currentWidth = width
                with open(decompPath, "w")as decompFile:
                    decompFile.write(decomp)
        return currentWidth
    except subprocess.TimeoutExpired as tim:
        print("        TIMEOUT: htd - " + str(type(tim)) + " " + str(tim.args))
        return currentWidth


# get width
def getTreeWidth(graphPath, decompPath, seed):
    try:
        with open(decompPath, "w") as decompFile:
            with open(graphPath, "r")as graphFile:
                decomp = check_output(["./htd_main", "--opt", "width", "-s", str(seed)], stdin=graphFile, timeout=300).decode('ascii')
                decompFile.write(decomp)
                return int(decomp.splitlines()[0].split(" ")[3]) - 1
    except subprocess.TimeoutExpired as tim:
        print("        TIMEOUT: htd - " + str(type(tim)) + " " + str(tim.args))
        return -1
    except BaseException as ex:
        print("        ERROR: htd - " + str(type(ex)) + " " + str(ex.args))
        return -1


def checkSAT(formula):
    try:
        out = check_output(["./lingeling-bbc", formula], timeout=300).decode('ascii')
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
        return 'ERROR'
    except subprocess.TimeoutExpired as tim:
        print("        TIMEOUT: checkSAT - " + str(type(tim)) + " " + str(tim.args))
        return 'TIMEOUT'
    except BaseException as ex:
        print("        Error: checkSAT - " + str(type(ex)) + " " + str(ex.args))
        return 'ERROR'


# preprocess with SatELite
def preprocessFormula(formualFile, preprocFile):
    decomp = check_output(["./SatELite", "+pre", "+det", formualFile, preprocFile], timeout=300).decode('UTF-8')
    return decomp


def cleanFormula(formula):
    formula = re.sub('\s*%\s*0\s*', ' ', formula.replace("\t", " "))
    formula_ = " ".join([a for a in formula.splitlines() if len(a) > 0 and a[0] != "c" and a[0] != "p" and a[0] != "w"]).replace(" 0 ", " 0\n")
    problemLine = [a for a in formula.splitlines() if len(a) > 0 and a[0] == "p"]
    if len(problemLine) > 0 and len(formula_) > 0:
        problemLine = problemLine[0]
        return re.sub(r' +', ' ', re.sub(r'\n +', '\n', ("p cnf " + str(problemLine.split()[2]) + " " + str(len(formula_.splitlines())) + "\n" + formula_)))
    return ""


def genIncidenceGraph(formula):
    graph = set()
    numRules = 0
    numVariables = 0
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] == 'p':
                numVariables = int(line.split()[2])
    clauses = []
    clause = []
    for line in formula.splitlines():
        if len(line) > 0 and line[0] != 'p' and line[0] != 'c' and line[0] != 's' and line[0] != 'w' and line[0] != '%':
            for lit in [int(x) for x in line.split() if len(x) != 0]:
                if lit == 0 and len(clause) > 0:
                    clauses += [clause]
                    clause = []
                else:
                    clause += [lit]
    for lits in clauses:
        numRules += 1
        for node in lits:
            ainode = abs(int(node))
            if ainode != 0:
                graph |= {(ainode, numRules + numVariables)}
                if len(graph) > 50000000:
                    raise Exception('Incidence Oversize')

    graphString = ("p tw " + str(numVariables + numRules) + " " + str(len(graph)))
    for node in graph:
        graphString += ("\n" + str(node[0]) + " " + str(node[1]))
    return graphString


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


def getModelsW(sol):
    i = 0
    lines = sol.split("\n")
    while i < len(lines):
        if lines[i].startswith("Satisfying probability"):
            return float(lines[i].split()[2])
        i += 1


def getMaxVarOcc(formula):
    occurences = {}
    for i in range(1, int(getNumVariables(formula)) + 1):
        occurences[str(i)] = 0
    newFormula = formula.replace("-", "")
    for line in newFormula.splitlines():
        if len(line) > 0 and line[0] != 'p' and line[0] != 'c' and line[0] != 's' and line[0] != '%':
            for item in line.split():
                if item != "0":
                    occurences[item] += 1
    return max(occurences.items(), key=lambda k: k[1])[1]


def getNumVariables(formula):
    return [x for x in formula.splitlines() if len(x) > 0 and x[0] == 'p'][0].split()[2]


def getNumClauses(formula):
    return [x for x in formula.splitlines() if len(x) > 0 and x[0] == 'p'][0].split()[3]


def getMaxClauseSize(formula):
    return max([len(x.split()) - 1 for x in formula.splitlines() if len(x) > 0 and x[0] != "p"])


def getNumAtoms(formula):
    newFormula = [[y for y in x.split(" ") if y] for x in formula.splitlines() if len(x) > 0 and x[0] != 'p' and x[0] != 'c' and x[0] != 's' and x[0] != '%']
    return len([y for x in newFormula for y in x if len(y) > 0 and y != "0"])


def simplePreproc(formula):
    newFormula = [list(map(int, [y for y in x.split() if y])) for x in formula.splitlines() if len(x) > 0 and x[0] != 'p']
    currentFacts = [x[0] for x in newFormula if len(x) == 2]
    oldFacts = currentFacts
    while len(currentFacts) > 0:
        newFormula = [[y for y in x if not (-y in currentFacts)] for x in newFormula if len(x) > 2 and len([y for y in x if y in currentFacts]) == 0]
        currentFacts = [x[0] for x in newFormula if len(x) == 2]
        oldFacts += currentFacts
    newFormula = [[x, 0] for x in oldFacts] + newFormula
    variables = [x for x in formula.splitlines() if len(x) > 0 and x[0] == 'p'][0].split()[2]
    if len(newFormula) > 0:
        return '\n'.join(["p cnf " + str(variables) + " " + str(len(newFormula))] + [' '.join(map(str, x)) for x in newFormula]) + '\n'
    else:
        return ""
