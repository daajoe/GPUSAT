#!/usr/bin/env python
import csv
from os.path import join, isfile, isdir
import os
from CommonMethods import *
from shutil import copyfile

dirRaw = "./benchmarks/clean"
dirFormula = "./benchmarks/ready"

fieldnames = ['file_name', "num_Variables",
              'width_primal', "width_incidence", "max_clause_size", "max_var_occ", "num_Atoms", "num_Clauses",
              'width_primal_preproc', "width_incidence_preproc", "max_clause_size_preproc", "max_var_occ_preproc", "num_Atoms_preproc", "num_Clauses_preproc",
              "sat-unsat"]

summaryFile = './benchmarks/Summary_Benchmark_Width.csv'
summaryFileReady = './benchmarks/Summary_Benchmark_Width_Ready.csv'

if not isfile(summaryFileReady):
    with open(summaryFileReady, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
if not isdir(dirFormula):
    os.makedirs(dirFormula)

reader = csv.DictReader(open(summaryFile))
for row in reader:
    if isfile(join(dirRaw, row['file_name'] + ".cnf")):
        if ((int(row['width_primal']) <= 40 and int(row['width_primal']) >= 1) or (int(row['width_incidence']) <= 40 and int(row['width_incidence']) >= 1)) and row["sat-unsat"] != "ERROR":
            print("formula: " + row['file_name'])
            if (isfile(join(dirRaw, row['file_name'] + ".cnf"))):
                with open(join(dirRaw, row['file_name'] + ".cnf"), "r") as rawFile:
                    with open(join(dirFormula, row['file_name'] + ".cnf"), "w") as selectedFile:
                        formula = rawFile.read()
                        selectedFile.write(cleanFormula(formula))
                        with open(summaryFileReady, 'a') as csvf:
                            wr = csv.DictWriter(csvf, fieldnames=fieldnames)
                            wr.writerow(row)
                            csvf.flush()
