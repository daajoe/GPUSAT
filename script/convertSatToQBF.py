#!/usr/bin/env python3
import re
from multiprocessing.pool import ThreadPool
from os import listdir, makedirs
from os.path import join, isdir

dirSatFormulas = "./benchmarks/ready"
dirQBFProgramsa = "./benchmarks/qbfa"
dirQBFProgramse = "./benchmarks/qbfe"

if not isdir(dirQBFProgramsa):
    makedirs(dirQBFProgramsa)
if not isdir(dirQBFProgramse):
    makedirs(dirQBFProgramse)


def process(i):
    print("Instance: " + i)
    with open(join(dirSatFormulas, i), "r") as formulaFile:
        formula = re.sub(' +', ' ', formulaFile.read().replace("\t", " "))
        cleanFormula = "\n".join([a for a in formula.splitlines() if a[0] != "c" and a[0] != "p" and a[0] != "w"])
        problemLine = [a for a in formula.splitlines() if a[0] != "c" and a[0] == "p"][0]
        qbfProgram = problemLine + "\ne "
        numVars = int(problemLine.split()[2])
        for a in range(1, numVars + 1):
            qbfProgram += str(a) + " "
        qbfProgram += "0\n"
        qbfProgram += cleanFormula
        with open(join(dirQBFProgramse, i), "w")as qbfFile:
            qbfFile.write(qbfProgram)

        qbfProgram = problemLine + "\na "
        numVars = int(problemLine.split()[2])
        for a in range(1, numVars + 1):
            qbfProgram += str(a) + " "
        qbfProgram += "0\n"
        qbfProgram += cleanFormula
        with open(join(dirQBFProgramsa, i), "w")as qbfFile:
            qbfFile.write(qbfProgram)


pool = ThreadPool(6)
for i in listdir(dirSatFormulas):
    pool.apply_async(process, [i])
pool.close()
pool.join()
