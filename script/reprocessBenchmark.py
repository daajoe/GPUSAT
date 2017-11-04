#!/usr/bin/env python
import csv
from functools import partial
from multiprocessing.pool import Pool
from os.path import join, isdir, isfile
from os import makedirs, listdir

import multiprocessing

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

fieldnames = ['file_name', "num_Variables",
              'width_primal', "width_incidence", "max_clause_size", "max_var_occ", "num_Atoms", "num_Clauses",
              'width_primal_preproc', "width_incidence_preproc", "max_clause_size_preproc", "max_var_occ_preproc", "num_Atoms_preproc", "num_Clauses_preproc",
              "sat-unsat"]

oldSummaryFile = './benchmarks/Summary_Benchmark_Width.csv'
summaryFile = './benchmarks/Summary_Benchmark_Width_new.csv'

total_size = len(listdir(dirRaw))
processing = 0

with open(summaryFile, "r") as csvfile:
    contents = csvfile.read()


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


def process(lock,row):
    testcase = row['file_name'] + ".cnf"
    case = row['file_name']
    if case not in contents:

        try:
            with open(join(dirClean, testcase), "r") as formulaFile:
                formula = formulaFile.read()
            if row['num_Variables'] in (None, ""):
                row['num_Variables'] = getNumVariables(formula)
            if row['max_clause_size'] in (None, ""):
                row['max_clause_size'] = getMaxClauseSize(formula)
            if row['max_var_occ'] in (None, ""):
                row['max_var_occ'] = getMaxVarOcc(formula)
            if row['num_Atoms'] in (None, ""):
                row['num_Atoms'] = getNumAtoms(formula)
            if row['num_Clauses'] in (None, ""):
                row['num_Clauses'] = getNumClauses(formula)
            if row['width_primal'] in (None, ""):
                with open(join(dirPrimal, case + ".gr"), "w")as graphFile:
                    graphFile.write(genPrimalGraph(formula))
                row['width_primal'] = getBestTreeDecomp(join(dirPrimal, case + ".gr"), dirDecomp + "/primal/" + testcase + ".td", 10)
            if row['width_incidence'] in (None, ""):
                with open(join(dirIncidence, case + ".gr"), "w")as graphFile:
                    graphFile.write(genIncidenceGraph(formula))
                row['width_incidence'] = getBestTreeDecomp(join(dirIncidence, case + ".gr"), dirDecomp + "/incidence/" + testcase + ".td", 10)
        except BaseException as ex:
            print("        Error: - " + str(type(ex)) + " " + str(ex.args))

        if row['sat-unsat'] in (None, ""):
            row['sat-unsat'] = checkSAT(join(dirClean, testcase))

        try:
            with open(join(preprocessPath, testcase), "r") as formulaFile:
                formula = formulaFile.read()
            if row['max_clause_size_preproc'] in (None, ""):
                row['max_clause_size_preproc'] = getMaxClauseSize(formula)
            if row['max_var_occ_preproc'] in (None, ""):
                row['max_var_occ_preproc'] = getMaxVarOcc(formula)
            if row['num_Atoms_preproc'] in (None, ""):
                row['num_Atoms_preproc'] = getNumAtoms(formula)
            if row['num_Clauses_preproc'] in (None, ""):
                row['num_Clauses_preproc'] = getNumClauses(formula)
            if row['width_primal_preproc'] in (None, ""):
                with open(join(dirPrimal, case + ".gr"), "w")as graphFile:
                    graphFile.write(genPrimalGraph(formula))
                row['width_primal_preproc'] = getBestTreeDecomp(join(dirPrimal, case + ".gr"), dirDecomp + "/preproc/primal/" + testcase + ".td", 10)
            if row['width_incidence_preproc'] in (None, ""):
                with open(join(dirIncidence, case + ".gr"), "w")as graphFile:
                    graphFile.write(genIncidenceGraph(formula))
                row['width_incidence_preproc'] = getBestTreeDecomp(join(dirIncidence, case + ".gr"), dirDecomp + "/preproc/incidence/" + testcase + ".td", 10)
        except BaseException as ex:
            print("        Error: - " + str(type(ex)) + " " + str(ex.args))

        lock.acquire()
        with open(summaryFile, 'a') as csvf:
            wr = csv.DictWriter(csvf, fieldnames=fieldnames)
            wr.writerow(row)
            csvf.flush()
        lock.release()


with open(oldSummaryFile, 'r') as csvf:
    reader = csv.DictReader(csvf, fieldnames=fieldnames)

    pool = Pool(12)
    m = multiprocessing.Manager()
    l = m.Lock()
    func=partial(process,l)
    pool.map(func, reader)
    pool.close()
    pool.join()
