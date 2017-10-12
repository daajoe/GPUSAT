#!/usr/bin/env python
import csv
from os.path import join, isfile, isdir
import os
from CommonMethods import *
from shutil import copyfile

dirRaw = "./benchmarks/raw"
dirFormula = "./benchmarks/ready"

fieldnames = ['file_name', 'width_primal_s1234', "width_incidence_s1234", "max_clause_size", "max_var_occ", 'width_primal_s1234_satelite',
              "width_incidence_s1234_satelite", "max_clause_size_satelite", "max_var_occ_satelite", "sat-unsat"]

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
    if ((int(row['width_primal_s1234']) <= 40 and int(row['width_primal_s1234']) >= 1) or (
                    int(row['width_incidence_s1234']) <= 40 and int(row['width_incidence_s1234']) >= 1)) and row["sat-unsat"] != "ERROR":
        print("formula: " + row['file_name'] + " width: " + row['width_primal_s1234'])
        if (isfile(join(dirRaw, row['file_name'] + ".cnf"))):
            with open(join(dirRaw, row['file_name'] + ".cnf"), "r") as rawFile:
                with open(join(dirFormula, row['file_name'] + ".cnf"), "w") as selectedFile:
                    formula = rawFile.read()
                    selectedFile.write(cleanFormula(formula))
                    with open(summaryFileReady, 'a') as csvf:
                        wr = csv.DictWriter(csvf, fieldnames=fieldnames)
                        wr.writerow(row)
                        csvf.flush()
