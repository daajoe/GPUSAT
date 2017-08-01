#!/usr/bin/env python2.7
from os import listdir
import subprocess
from os.path import join, isdir
from os import makedirs

dirFormula = "./problems/formula"
dirReference = "./problems/reference"

testCasesStrings = ["a_00_test", "b_00_test", "c_00_test", "a_01_test", "b_01_test", "c_01_test"]

for case in testCasesStrings:
    for testcase in listdir(join(dirReference, case)):
        with open(join(join(dirReference, case), testcase), "w") as resultFile:
            print ("check: " + join(join(dirReference, case), testcase))
            subprocess.call(["clasp", "--outf=2", "-n", "0", "-q", join(join(dirFormula, case), testcase + ".cnf")],
                            stdout=resultFile)
