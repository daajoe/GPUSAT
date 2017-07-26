#!/usr/bin/env python2.7
import json
from os import listdir
import subprocess, getpass
from os.path import join, isdir
from os import makedirs
import datetime

dirFormula = "./problems/formula"
dirDecomp = "./problems/decomposition"
dirResults = "./problems/results"
dirReference = "./problems/reference"

testCasesStrings = ["a_00_test", "b_00_test", "c_00_test"]


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


for case in testCasesStrings:
    if not isdir(join(dirResults, case)):
        makedirs(join(dirResults, case))
    numElements = len(listdir(join(dirReference, case)))
    currentElement = 0
    for testcase in listdir(join(dirReference, case)):
        currentElement += 1
        if not testcase in listdir(join(dirResults, case)):
            with open("./Summary.txt", "a") as summaryFile:
                summaryFile.write("Testcase (" + str(currentElement) + "/" + str(numElements) + "): " + case + "/" + testcase + "\n")
                summaryFile.flush()
                print("Testcase (" + str(currentElement) + "/" + str(numElements) + "): " + case + "/" + testcase + " " + str(datetime.datetime.now().time()))

                # generate output
                with open(join(join(dirResults, case), testcase), "w") as resultFile:
                    subprocess.call(
                        ["./gpusat", "-f", dirDecomp + "/" + case + "/" + testcase + ".td", "-s", dirFormula + "/" + case + "/" + testcase + ".cnf"],
                        stdout=resultFile, stderr=resultFile)
                # check results
                with open(join(dirResults, case) + "/" + testcase, "r") as resultFile:
                    with open(join(dirReference, case) + "/" + testcase, "r") as referenceFile:
                        try:
                            data = resultFile.read()
                            d = json.loads(data)
                            ref = str(referenceFile.read())
                            summaryFile.write("    ModelCount: ")
                            numModels = getModels(ref)
                            if d['Model Count'] == getModels(ref):
                                summaryFile.write("OK\n")
                                print("    ModelCount: OK")
                            else:
                                summaryFile.write("Failure\n")
                                print("    ModelCount: Failure")
                            summaryFile.write("    Time Total: " + str(d['Time']['Total']) + "\n    Time Solving: " + str(
                                d['Time']['Solving']) + "\n    Time Init_OpenCL: " + str(d['Time']['Init_OpenCL']) + "\n    Time without Init_OpenCL: " + str(
                                d['Time']['Total'] - d['Time']['Init_OpenCL']) + "\n    Time Clasp: " + str(getTime(ref)) + "\n")
                            print("    Time Total: " + str(d['Time']['Total']) + " Init: " + str(
                                d['Time']['Total'] - d['Time']['Init_OpenCL']) + " Time Reference: " + str(getTime(ref)))
                        except ValueError:
                            summaryFile.write("    Error\n")
                            print("    Error")
