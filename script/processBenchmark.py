#!/usr/bin/env python
import csv
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool
from os.path import join, isdir, isfile
from os import makedirs, listdir
from CommonMethods import *

dirRaw = "./benchmarks/raw"
dirFormula = "./benchmarks/formula"
dirDecomp = "./benchmarks/decomposition"
dirIncidence = "./benchmarks/incidence"
dirIncidenceDecomp = "./benchmarks/incidenceDecomp"
dirPrimal = "./benchmarks/primal"
dirPrimalDecomp = "./benchmarks/primalDecomp"
preprocessPath = "./benchmarks/preproc"
dirClean = "./benchmarks/clean"

fieldnames = ['file_name', 'width_primal_s1234', "width_incidence_s1234", "max_clause_size", "max_var_occ", 'width_primal_s1234_satelite',
              "width_incidence_s1234_satelite", "max_clause_size_satelite", "max_var_occ_satelite", "sat-unsat"]

summaryFile = './benchmarks/Summary_Benchmark_Width.csv'

lock = Lock()

if not isfile(summaryFile):
    with open(summaryFile, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
if not isdir(dirFormula):
    makedirs(dirFormula)
if not isdir(dirDecomp):
    makedirs(dirDecomp)
if not isdir(dirIncidence):
    makedirs(dirIncidence)
if not isdir(dirPrimal):
    makedirs(dirPrimal)
if not isdir(preprocessPath):
    makedirs(preprocessPath)
if not isdir(dirClean):
    makedirs(dirClean)

with open(summaryFile, "r") as csvfile:
    contents = csvfile.read()
total_size = len(listdir(dirRaw))
processing = 0


def process(testcase):
    global processing
    current = processing
    processing += 1
    case = testcase[:-4]
    row = {'file_name': case}
    row['sat-unsat'] = 'ERROR'
    row['width_primal_s1234'] = -1
    row['width_incidence_s1234'] = -1
    row['max_clause_size'] = -1
    row['max_var_occ'] = -1
    row['width_primal_s1234_satelite'] = -1
    row['width_incidence_s1234_satelite'] = -1
    row['max_clause_size_satelite'] = -1
    row['max_var_occ_satelite'] = -1
    if case not in contents:
        try:
            print("%s/%s: %s" % (str(current), str(total_size), testcase))

            with open(join(dirRaw, testcase), "r") as formulaFile:
                with open(join(dirClean, testcase), "w") as cleanFile:
                    cleanFile.write(cleanFormula(formulaFile.read()))


            with open(join(dirClean, testcase), "r") as formulaFile:
                formula = formulaFile.read()
                with open(join(dirPrimal, case + ".gr"), "w")as graphFile:
                    graphFile.write(genPrimalGraph(formula))
                with open(join(dirPrimal, case + ".gr"), "r")as graphFile:
                    row['width_primal_s1234'] = getTreeWidth(graphFile, 1234)
                with open(join(dirIncidence, case + ".gr"), "w")as graphFile:
                    graphFile.write(genIncidenceGraph(formula))
                with open(join(dirIncidence, case + ".gr"), "r")as graphFile:
                    row['width_incidence_s1234'] = getTreeWidth(graphFile, 1234)
                row['max_clause_size'] = getMaxClauseSize(formula)
                row['max_var_occ'] = getMaxVarOcc(formula)

            row['sat-unsat'] = checkSAT(join(dirClean, testcase))
            preprocessFormula(join(dirClean, testcase), join(preprocessPath, testcase))
            if isfile(join(preprocessPath, testcase)):
                with open(join(preprocessPath, testcase), "r") as formulaFile:
                    formula = formulaFile.read()
                    with open(join(dirPrimal, case + ".gr"), "w")as graphFile:
                        graphFile.write(genPrimalGraph(formula))
                    with open(join(dirPrimal, case + ".gr"), "r")as graphFile:
                        row['width_primal_s1234_satelite'] = getTreeWidth(graphFile, 1234)
                    with open(join(dirIncidence, case + ".gr"), "w")as graphFile:
                        graphFile.write(genIncidenceGraph(formula))
                    with open(join(dirIncidence, case + ".gr"), "r")as graphFile:
                        row['width_incidence_s1234_satelite'] = getTreeWidth(graphFile, 1234)
                    row['max_clause_size_satelite'] = getMaxClauseSize(formula)
                    row['max_var_occ_satelite'] = getMaxVarOcc(formula)

        except BaseException as ex:
            test = 1
        lock.acquire()
        with open(summaryFile, 'a') as csvf:
            wr = csv.DictWriter(csvf, fieldnames=fieldnames)
            wr.writerow(row)
            csvf.flush()
        lock.release()


numFiles = listdir(dirRaw)
pool = ThreadPool(4)
for i in numFiles:
    pool.apply_async(process, [i])
    # process(i)
pool.close()
pool.join()
