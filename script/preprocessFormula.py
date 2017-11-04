#!/usr/bin/env python
from multiprocessing.pool import Pool
from os.path import join, isfile, isdir
import os
from CommonMethods import *

dirFormula = "./benchmarks/clean"
dirPreproc = "./benchmarks/preproc"

if not isdir(dirPreproc):
    os.makedirs(dirPreproc)
if not isdir(dirFormula):
    os.makedirs(dirFormula)


def process(f):
    if not isfile(join(dirPreproc, f)):
        print("formula: " + f)
        with open(join(dirFormula, f), "r") as rawFile:
            formula = rawFile.read()
            preprocessed = simplePreproc(formula)
            with open(join(dirPreproc, f), "w") as selectedFile:
                selectedFile.write(preprocessed)


if __name__ == "__main__":
    pool = Pool(1)
    for i in os.listdir(dirFormula):
        pool.apply_async(process, [i])
    pool.close()
    pool.join()