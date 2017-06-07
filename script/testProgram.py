import json
from os import listdir
import subprocess
from os.path import join, isdir
from os import makedirs
import datetime

dirFormula = "./problems/formula"
dirDecomp = "./problems/decomposition"
dirResults = "./problems/results"
dirReference = "./problems/reference"

numElements = len(listdir(dirReference))
currentElement = 0

if not isdir(dirResults):
    makedirs(dirResults)

def getTime(sol):
    if sol[0] == "{":
        return json.loads(sol)['Time']['Total']
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
    else:
        i = 0
        lines = sol.split("\n")
        while i < len(lines):
            if "# solutions" in lines[i]:
                return int(lines[i + 1])
            i += 1


def check_model(model, clauses):
    for line in clauses:
        sat = False
        for var in line[:-1]:
            if var in model:
                sat = True
                break
        if not sat:
            return False
    return True


for testcase in listdir(dirReference):
    currentElement += 1
    if not testcase in listdir(dirResults):
        with open("./Summary.txt", "a") as summaryFile:
            summaryFile.write("Testcase (" + str(currentElement) + "/" + str(numElements) + "): " + testcase + "\n")
            summaryFile.flush()
            print("Testcase (" + str(currentElement) + "/" + str(numElements) + "): " + testcase + " " + str(
                datetime.datetime.now().time()))

            # generate output
            with open(join(dirResults, testcase), "w") as resultFile:
                subprocess.call(
                    ["./gpusat", "-f", dirDecomp + "/" + testcase + ".td", "-s",
                     dirFormula + "/" + testcase + ".cnf"],
                    stdout=resultFile, stderr=resultFile)
            # check results
            with open(dirResults + "/" + testcase, "r") as resultFile:
                with open(dirReference + "/" + testcase, "r") as referenceFile:
                    try:
                        data = resultFile.read()
                        d = json.loads(data)
                        summaryFile.write("    Model: ")
                        mod = str(d['Model'])
                        ref = str(referenceFile.read())
                        with open(dirFormula + "/" + testcase + ".cnf", "r") as formulaFile:
                            if not d['Model'] == 'UNSATISFIABLE':
                                if check_model(mod, formulaFile.read().splitlines()[1:]):
                                    summaryFile.write("OK\n")
                                    print("    Model: OK")
                                else:
                                    summaryFile.write("Failure\n")
                                    print("    Model: Failure")
                        summaryFile.write("    ModelCount: ")
                        numModels = getModels(ref)
                        if d['Model Count'] == getModels(ref):
                            summaryFile.write("OK\n")
                            print("    ModelCount: OK")
                        else:
                            summaryFile.write("Failure\n")
                            print("    ModelCount: Failure")
                        summaryFile.write("    Time Total: " + str(d['Time']['Total']) + " Time Clasp: " + str(
                            getTime(ref)) + "\n")
                        print("    Time Total: " + str(d['Time']['Total']) + " Time Reference: " + str(
                            getTime(ref)))
                    except ValueError:
                        summaryFile.write("    Error\n")
                        print("    Error")
