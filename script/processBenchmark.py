#!/usr/bin/env python
import csv
import multiprocessing
from os.path import join, isdir, isfile
from os import makedirs, listdir
from CommonMethods import *
from functools import partial

dirRaw = "./benchmarks/raw"
dirFormula = "./benchmarks/formula"
dirDecomp = "./benchmarks/decomposition"
dirIncidence = "./benchmarks/incidence"
dirIncidenceDecomp = "./benchmarks/incidenceDecomp"
dirPrimal = "./benchmarks/primal"
dirPrimalDecomp = "./benchmarks/primalDecomp"
preprocessPath = "./benchmarks/preproc"
dirClean = "./benchmarks/clean"

fieldnames = ['file_name', "num_Variables",
              'width_primal', "width_incidence", "max_clause_size", "max_var_occ", "num_Atoms", "num_Clauses",
              'width_primal_preproc', "width_incidence_preproc", "max_clause_size_preproc", "max_var_occ_preproc", "num_Atoms_preproc", "num_Clauses_preproc",
              "sat-unsat"]

summaryFile = './benchmarks/Summary_Benchmark_Width.csv'

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
if not isdir(dirDecomp + "/primal/"):
    makedirs(dirDecomp + "/primal/")
if not isdir(dirDecomp + "/incidence/"):
    makedirs(dirDecomp + "/incidence/")
if not isdir(dirDecomp + "/preproc/primal/"):
    makedirs(dirDecomp + "/preproc/primal/")
if not isdir(dirDecomp + "/preproc/incidence/"):
    makedirs(dirDecomp + "/preproc/incidence/")

with open(summaryFile, "r") as csvfile:
    contents = csvfile.read()
total_size = len(listdir(dirRaw))
processing = 0


def process(lock,testcase):
    case = testcase[:-4]
    row = {'file_name': case}
    row['sat-unsat'] = 'ERROR'
    row['num_Variables'] = -1
    row['width_primal'] = -1
    row['width_incidence'] = -1
    row['max_clause_size'] = -1
    row['max_var_occ'] = -1
    row['num_Atoms'] = -1
    row['num_Clauses'] = -1
    row['width_primal_preproc'] = -1
    row['width_incidence_preproc'] = -1
    row['max_clause_size_preproc'] = -1
    row['max_var_occ_preproc'] = -1
    row['num_Atoms_preproc'] = -1
    row['num_Clauses_preproc'] = -1
    if case not in contents:
        print("%s" % (testcase))

        try:
            with open(join(dirClean, testcase), "r") as formulaFile:
                formula = formulaFile.read()
            row['max_clause_size'] = getMaxClauseSize(formula)
            row['max_var_occ'] = getMaxVarOcc(formula)
            row['num_Atoms'] = getNumAtoms(formula)
            row['num_Variables'] = getNumVariables(formula)
            row['num_Clauses'] = getNumClauses(formula)
            with open(join(dirPrimal, case + ".gr"), "w")as graphFile:
                graphFile.write(genPrimalGraph(formula))
            row['width_primal'] = getTreeWidth(join(dirPrimal, case + ".gr"), dirDecomp + "/primal/" + testcase + ".td", 1234)
            with open(join(dirIncidence, case + ".gr"), "w")as graphFile:
                graphFile.write(genIncidenceGraph(formula))
            row['width_incidence'] = getTreeWidth(join(dirIncidence, case + ".gr"), dirDecomp + "/incidence/" + testcase + ".td", 1234)
        except BaseException as ex:
            print("        Error: - " + str(type(ex)) + " " + str(ex.args))

        row['sat-unsat'] = checkSAT(join(dirClean, testcase))

        try:
            with open(join(preprocessPath, testcase), "r") as formulaFile:
                formula = formulaFile.read()
            row['max_clause_size_preproc'] = getMaxClauseSize(formula)
            row['max_var_occ_preproc'] = getMaxVarOcc(formula)
            row['num_Atoms_preproc'] = getNumAtoms(formula)
            row['num_Clauses_preproc'] = getNumClauses(formula)
            with open(join(dirPrimal, case + ".gr"), "w")as graphFile:
                graphFile.write(genPrimalGraph(formula))
            row['width_primal_preproc'] = getTreeWidth(join(dirPrimal, case + ".gr"), dirDecomp + "/preproc/primal/" + testcase + ".td", 1234)
            with open(join(dirIncidence, case + ".gr"), "w")as graphFile:
                graphFile.write(genIncidenceGraph(formula))
            row['width_incidence_preproc'] = getTreeWidth(join(dirIncidence, case + ".gr"), dirDecomp + "/preproc/incidence/" + testcase + ".td", 1234)
        except BaseException as ex:
            print("        Error: - " + str(type(ex)) + " " + str(ex.args))

        lock.acquire()
        with open(summaryFile, 'a') as csvf:
            wr = csv.DictWriter(csvf, fieldnames=fieldnames)
            wr.writerow(row)
            csvf.flush()
        lock.release()


if __name__ == "__main__":
    numFiles = listdir(dirRaw)
    pool = multiprocessing.Pool(12)
    m = multiprocessing.Manager()
    l = m.Lock()
    func=partial(process,l)
    pool.map(func, numFiles)
    pool.close()
    pool.join()
