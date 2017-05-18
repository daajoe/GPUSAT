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
                    if (mod in res):
                        summaryFile.write("OK\n")
                        print("    Model: OK")
                    else:
                        summaryFile.write("Failure\n")
                        print("    Model: Failure")
                    summaryFile.write("    ModelCount: ")
                    if (("\"Number\": " + str(d['Model Count'])) in res):
                        summaryFile.write("OK\n")
                        print("    ModelCount: OK")
                    else:
                        summaryFile.write("Failure\n")
                        print("    ModelCount: Failure")
                except ValueError:
                    summaryFile.write("    Error\n")
                    print("    Error")
    currentElement += 1
