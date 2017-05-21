import json
from os import listdir
import subprocess
from os.path import join

dirFormula = "./problems/formula"
dirDecomp = "./problems/decomposition"
dirResults = "./problems/results"
dirReference = "./problems/reference"

numElements = len(listdir(dirReference))
currentElement = 1


def check_model(model, clauses):
    for line in clauses:
        sat = False
        for var in line:
            if var in model:
                sat = True
                break
        if not sat:
            return False


for testcase in listdir(dirReference):
    # generate output
    with open(join(dirResults, testcase), "w") as resultFile:
        subprocess.call(
            ["gpusat.exe", "-f", dirDecomp + "/" + testcase + ".td", "-s", dirFormula + "/" + testcase + ".cnf"],
            stdout=resultFile, stderr=resultFile)
    # check results
    with open(dirResults + "/" + testcase, "r") as resultFile:
        with open(dirReference + "/" + testcase, "r") as referenceFile:
            with open("./Summary.txt", "a") as summaryFile:
                summaryFile.write("Testcase (" + str(currentElement) + "/" + str(numElements) + "): " + testcase + "\n")
                print("Testcase (" + str(currentElement) + "/" + str(numElements) + "): " + testcase)
                try:
                    data = resultFile.read()
                    d = json.loads(data)
                    summaryFile.write("    Model: ")
                    mod = str(d['Model'])
                    res = str(referenceFile.read())
                    referenceJSON = json.loads(referenceFile.read())
                    if check_model(mod, res.splitlines()[-1]):
                        summaryFile.write("OK\n")
                        print("    Model: OK")
                    else:
                        summaryFile.write("Failure\n")
                        print("    Model: Failure")
                    summaryFile.write("    ModelCount: ")
                    if d['Model Count'] == referenceJSON['Models']['Number']:
                        summaryFile.write("OK\n")
                        print("    ModelCount: OK")
                    else:
                        summaryFile.write("Failure\n")
                        print("    ModelCount: Failure")
                    print("    Time:" + d['Time_Total'] + " Time Clasp:" + referenceJSON['Time']['Total'])
                except ValueError:
                    summaryFile.write("    Error\n")
                    print("    Error")
    currentElement += 1
