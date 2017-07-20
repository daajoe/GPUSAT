#!/usr/bin/env python2.7
from os import listdir
import subprocess
from os.path import join, isdir
from os import makedirs

dirFormula = "./problems/formula"
dirGraphs = "./problems/graph"
dirDecomp = "./problems/decomposition"
dirReference = "./problems/reference"
dirResults = "./problems/results"

testCasesStrings = ["a_00_test", "b_00_test", "c_00_test", "a_01_test", "b_01_test", "c_01_test",
                    "a_02_test", "b_02_test", "c_02_test", "a_03_test", "b_03_test", "c_03_test",
                    "benchmark", "benchmark_0", "benchmark_1", "benchmark_2",
                    "benchmark_3", "benchmark_4", "benchmark_big", "benchmark_small"]

for case in testCasesStrings:
    if not isdir(dirDecomp):
        makedirs(dirDecomp)
    for testcase in listdir(join(dirReference, case)):
        if not isdir(join(dirDecomp, case)):
            makedirs(join(dirDecomp, case))
        with open(join(join(dirDecomp, case), testcase + ".td"), "w") as decompFile:
            with open(join(join(dirGraphs, case), testcase + ".gr"), "r") as infile:
                print ("decomp: " + join(join(dirReference, case), testcase))
                subprocess.call(["./htd_main", "--opt", "width"], stdout=decompFile, stdin=infile)
                decompFile.flush()
