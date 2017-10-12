import datetime
from os.path import isdir, join, isfile

from os import makedirs, listdir

import time

dirFormula = "./benchmarks/raw"
dirPreproc = "./benchmarks/preproc"

if not isdir(dirFormula):
    makedirs(dirFormula)
if not isdir(dirPreproc):
    makedirs(dirPreproc)


def simplePreproc(formula,timelimit=30):
    start = time.time()
    changed = True
    newFormula = formula.splitlines()
    while changed:
        changed = False
        numLines = len(newFormula)
        splitFormula = []
        for l in newFormula:
            splitFormula += l.split(" ")
        lines = [x for x in newFormula if len(x) > 0 and x[0] != 'p' and x[0] != 'c' and x[0] != 's' and len(x.split(" ")) == 2 and (splitFormula.count(x.split(" ")[0]) + splitFormula.count(str(int(x.split(" ")[0]) * -1))) > 1]
        for line in lines:
            if timelimit>=(time.time() - start ):
                return '\n'.join(newFormula) + '\n'
            parts = line.split(" ")
            var = parts[0]
            a = 0
            while (a < numLines):
                if timelimit>=(time.time() - start ):
                    return '\n'.join(newFormula) + '\n'
                fline = newFormula[a]
                parts = newFormula[a].split(" ")
                if len(fline) > 0 and fline[0] != 'p' and fline[0] != 'c' and fline[0] != 's' and len(
                        parts) > 2 and (str(int(var) * -1) in parts or var in parts):
                    if (var in parts) and (str(int(var) * -1) in parts):
                        return "p cnf 1 1\n1 -1 0\n"
                    elif var in parts:
                        del newFormula[a]
                        numLines -= 1
                        a -= 1
                        changed = True
                    elif int(var) > 0 and str(int(var) * -1) in parts:
                        newFormula[a] = newFormula[a].replace(str(int(var) * -1) + " ", " ")
                        changed = True
                    elif int(var) < 0 and str(int(var) * -1) in parts:
                        test_ = str(int(var) * -1)
                        newFormula[a] = newFormula[a].replace(str(int(var) * -1) + " ", "")
                        test = newFormula[a]
                        changed = True
                a += 1
    return '\n'.join(newFormula) + '\n'


if not isdir(dirPreproc):
    makedirs(dirPreproc)
numElements = len(listdir(dirFormula))
currentElement = 0
for testcase in listdir(dirFormula):
    currentElement += 1
    print("Testcase (" + str(currentElement) + "/" + str(numElements) + "): " + testcase + "  " + str(
                                datetime.datetime.now().time()))
    if not isfile(join(dirPreproc, testcase)):
        with open(join(dirPreproc, testcase), "w") as procfile:
            formula = open(join(dirFormula, testcase), "r")
            procstring = simplePreproc(formula.read())
            formula.close()
            procfile.write(procstring)
