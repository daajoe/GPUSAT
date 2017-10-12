import json
import subprocess
import time

from subprocess import check_output


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
def getTreeWidth(graphFile, seed):
    try:
        decomp = check_output(["./htd_main", "--opt", "width", "-s", str(seed)], stdin=graphFile, timeout=300).decode('ascii')
        return decomp.splitlines()[0].split(" ")[3]
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
    return '\n'.join([x for x in formula.splitlines() if len(x) > 0 and x[0] != 'c' and x[0] != 'w']).replace("\t", " ").replace("\n 0\n", " 0\n").replace("\n0\n",
                                                                                                                                                           " 0\n")


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
    numVariables = 0
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] == 'p':
                numVariables = int(line.split()[2])
    varOcc = [0] * (numVariables + 1)
    for line in formula.splitlines():
        if len(line) > 0:
            if line[0] != 'p' and line[0] != 'c' and line[0] != 's' and line[0] != '%' and line[0] != 'w':
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


def getMaxClauseSize(formula):
    max_clause_size = 0
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
        max_clause_size = len(lits) - 1 if (len(lits) - 1) > max_clause_size else max_clause_size
    return max_clause_size


def simplePreproc(formula, timelimit=30):
    start = time.time()
    newFormula = formula.replace("\t", " ").splitlines()
    startline = [x for x in newFormula if len(x) > 0 and x[0] == 'p']
    newFormula = [list(map(int, [y for y in x.split(" ") if y])) for x in newFormula if
                  len(x) > 0 and x[0] != 'p' and x[0] != 'c' and x[0] != 's' and x[0] != '%']
    allElems = []
    contradict = [x for x in newFormula if len([y for y in x if y != 0 and (-y) in x]) > 0]
    if len(contradict) > 0:
        return "p cnf 1 1\n1 -1 0\n"
    singleElems = [x[0] for x in newFormula if len(x) == 2]
    if timelimit <= (time.time() - start):
        return '\n'.join(startline + [' '.join(map(str, x)) for x in newFormula]) + '\n'
    while len(singleElems) > 0:
        newFormula = [x for x in newFormula if len([y for y in x if y in singleElems]) == 0]
        for i in range(0, len(newFormula)):
            if len(newFormula[i]) > 2:
                newFormula[i] = [x for x in newFormula[i] if not ((x * -1) in singleElems)]
        if timelimit <= (time.time() - start):
            return '\n'.join(startline + [' '.join(map(str, x)) for x in newFormula]) + '\n'
        allElems += singleElems
        singleElems = [x[0] for x in newFormula if len(x) == 2 and x not in allElems]
        if timelimit <= (time.time() - start):
            return '\n'.join(startline + [' '.join(map(str, x)) for x in newFormula]) + '\n'
    return '\n'.join(startline + [' '.join(map(str, x)) for x in newFormula]) + '\n'
