#!/usr/bin/env python2.7
import csv
import json
from os import listdir
import subprocess, getpass
from os.path import join, isdir, isfile
from os import makedirs
import datetime
from  CommonMethods import *

import math

import itertools

dirFormula = "./new_problems/formula"
dirDecomp = "./new_problems/decomposition"
dirResults = "./new_problems/results"
dirReference = "./new_problems/reference"
dirGraphs = "./new_problems/graph"

summaryFile = './Summary.csv'

testCasesStrings = [
    # "a_00_test",  "b_00_test", "c_00_test", "d_00_test",
    # "a_01_test",  "b_01_test", "c_01_test", "d_01_test",
    "a_02_test", "c_02_test",  # "b_02_test", "d_02_test",
    "a_03_test", "c_03_test",  # "b_03_test", "d_03_test",
    "a_04_test", "c_04_test", "#b_04_test", "d_04_test",
    # "Tests"
]

fieldnames = ['Case', "Success", 'Graph', "Precision", "Combine Width", "Model Count", 'Model Count Reference', "Total Time", "Solving Time", "Command"]

if not isfile(summaryFile):
    with open(summaryFile, 'w', newline='\n') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
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


# generate tree decomposition
def genTreeDecomp(graph, decompFile):
    decompFile.write(
        subprocess.check_output(["./htd_main", "--opt", "width", "-s", "1234"], stdin=graph, timeout=300).decode('ascii').replace("\r", ""))


for case in testCasesStrings:
    if not isdir(join(dirResults, case)):
        makedirs(join(dirResults, case))
    if not isdir(join(dirDecomp, case)):
        makedirs(join(dirDecomp, case))
    if not isdir(join(dirGraphs, case)):
        makedirs(join(dirGraphs, case))
    numElements = len(listdir(join(dirReference, case)))
    currentElement = 0
    for testcase in listdir(join(dirReference, case)):
        currentElement += 1
        if not isfile(join(join(dirResults, case), testcase)):
            open(join(dirResults, case) + "/" + testcase, "w").close()
            for wi, gr, prec in itertools.product([0], [0, 1], [0, 1]):
                row = {'Case': case + "/" + testcase}
                print("Testcase (" + str(currentElement) + "/" + str(numElements) + "): " + case + "/" + testcase + " " + str(datetime.datetime.now().time()))
                if gr == 0:
                    postfix = "_i"
                    print("    Incidence Graph")
                    row['Graph'] = "incidence"
                if gr == 1:
                    postfix = "_p"
                    print("    Primal Graph")
                    row['Graph'] = "primal"
                if prec == 0:
                    postfix_prec = "_d4"
                    print("    d4")
                    row['Precision'] = "d4"
                if prec == 1:
                    postfix_prec = "_d"
                    print("    double")
                    row['Precision'] = "double"
                print("    wi 0" + str(wi))
                row['Combine Width'] = str(wi)

                if False:
                    # generate Graph
                    with open(dirFormula + "/" + case + "/" + testcase + ".cnf", 'r')as formulaFile:
                        with open(dirGraphs + "/" + case + "/" + testcase + postfix + ".gr", 'w')as graphFile:
                            if gr == 0:
                                graphFile.write(genIncidenceGraph(formulaFile.read()))
                                print("    Incidence Graph")
                            if gr == 1:
                                graphFile.write(genPrimalGraph(formulaFile.read()))
                                print("    Primal Graph")

                    # generate decomposition
                    with open(dirGraphs + "/" + case + "/" + testcase + postfix + ".gr", 'r')as graphFile:
                        with open(dirDecomp + "/" + case + "/" + testcase + postfix + ".td", 'w')as decompFile:
                            genTreeDecomp(graphFile, decompFile)
                if True:
                    try:
                        # generate output
                        with open(join(join(dirResults, case), testcase + postfix + postfix_prec), "w") as resultFile:
                            if prec == 0:
                                subprocess.call(["../build_mingw_d4/gpusat.exe", "-f", dirDecomp + "/" + case + "/" + testcase + postfix + ".td", "-s",
                                                 dirFormula + "/" + case + "/" + testcase + ".cnf", "-c", "../kernel/", "-w", str(wi), "-m", "18"], timeout=30,
                                                stdout=resultFile, stderr=resultFile)
                                row[
                                    'Command'] = "../build_mingw_d4/gpusat.exe -f " + dirDecomp + "/" + case + "/" + testcase + postfix + ".td -s " + dirFormula + "/" + case + "/" + testcase + ".cnf -c ../kernel/ -w " + str(
                                    wi) + " -m 18"
                            if prec == 1:
                                subprocess.call(["../build_mingw_double/gpusat.exe", "-f", dirDecomp + "/" + case + "/" + testcase + postfix + ".td", "-s",
                                                 dirFormula + "/" + case + "/" + testcase + ".cnf", "-c", "../kernel/", "-w", str(wi), "-m", "18"], timeout=30,
                                                stdout=resultFile, stderr=resultFile)
                                row[
                                    'Command'] = "../build_mingw_double/gpusat.exe -f " + dirDecomp + "/" + case + "/" + testcase + postfix + ".td -s " + dirFormula + "/" + case + "/" + testcase + ".cnf -c ../kernel/ -w " + str(
                                    wi) + " -m 18"

                        weighted = False
                        with open(dirFormula + "/" + case + "/" + testcase + ".cnf") as formulaFile:
                            weighted = "\nw" in formulaFile.read()
                        # check results
                        try:
                            with open(join(dirResults, case) + "/" + testcase + postfix + postfix_prec, "r") as resultFile:
                                data = resultFile.read()
                            d = json.loads(data)
                            with open(join(dirReference, case) + "/" + testcase, "r") as referenceFile:
                                ref = str(referenceFile.read())
                            row['Model Count'] = d['Model Count']
                            if (weighted):
                                numModels = getModelsW(ref)
                                row['Model Count Reference'] = numModels
                                if (numModels == d['Model Count']) or (
                                                    numModels != 0 and d['Model Count'] != 0 and round(d['Model Count'],
                                                                                                       int(math.log10(d['Model Count'])) + 5) == round(
                                            numModels, int(math.log10(numModels)) + 5)):
                                    row['Success'] = "OK"
                                    print("    ModelCount: OK")
                                else:
                                    row['Success'] = "Failure"
                                    print("    ModelCount: Failure " + str(numModels) + "/" + str(d['Model Count']))
                            else:
                                numModels = getModels(ref)
                                row['Model Count Reference'] = numModels
                                if d['Model Count'] == numModels:
                                    row['Success'] = "OK"
                                    print("    ModelCount: OK")
                                else:
                                    row['Success'] = "Failure"
                                    print("    ModelCount: Failure " + str(numModels) + "/" + str(d['Model Count']))
                            print("    Time Total: " + str(d['Time']['Total']) + " Init: " + str(
                                d['Time']['Total'] - d['Time']['Init_OpenCL']) + " Time Reference: " + str(getTime(ref)))
                            row['Total Time'] = d['Time']['Total']
                            row['Solving Time'] = d['Time']['Solving']
                        except ValueError as ex:
                            row['Success'] = "Error"
                            print("    Error")
                    except subprocess.TimeoutExpired:
                        row['Success'] = "Timeout"
                        print("    Timeout")
                    with open(summaryFile, 'a', newline='\n') as csvf:
                        wr = csv.DictWriter(csvf, fieldnames=fieldnames)
                        wr.writerow(row)
                        csvf.flush()
