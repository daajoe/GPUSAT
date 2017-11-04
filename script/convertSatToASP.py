#!/usr/bin/env python3
import re
from multiprocessing.pool import ThreadPool
from os import listdir, makedirs
from subprocess import call

from os.path import join, isdir

dirSatFormulas = "./benchmarks/ready"
dirAspPrograms = "./benchmarks/asp"
dirAspProgramsGround = "./benchmarks/asp_ground"

if not isdir(dirAspPrograms):
    makedirs(dirAspPrograms)
if not isdir(dirAspProgramsGround):
    makedirs(dirAspProgramsGround)


def process(i):
    print("Instance: " + i)
    with open(join(dirSatFormulas, i), "r") as formulaFile:
        aspProgram = ""
        formula = re.sub(' +', ' ', formulaFile.read().replace("\t", " "))
        cleanFormula = " ".join([a for a in formula.splitlines() if a[0] != "c" and a[0] != "p" and a[0] != "w"]).split(" 0 ")
        problemLine = [a for a in formula.splitlines() if a[0] != "c" and a[0] == "p"][0]
        numVars = int(problemLine.split()[2])
        for a in range(1, numVars + 1):
            aspProgram += "{a_" + str(a) + "}.\n"
        for line in cleanFormula:
            if len(line)>1:
                aspProgram += ":-"
                for atom in line.split():
                    at = int(atom)
                    if at < 0:
                        aspProgram += " a_" + str(abs(at)) + ","
                    if at > 0:
                        aspProgram += " not a_" + str(at) + ","
                aspProgram = aspProgram[:-1] + ".\n"

        with open(join(dirAspPrograms, i), "w")as aspFile:
            aspFile.write(aspProgram)

        with open(join(dirAspProgramsGround, i), "w")as groundFile:
            call(["gringo", join(dirAspPrograms, i)], stdout=groundFile)


pool = ThreadPool(6)
for i in listdir(dirSatFormulas):
    pool.apply_async(process, [i])
pool.close()
pool.join()
