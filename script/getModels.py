#!/usr/bin/env python3
import csv
import decimal
import xml.etree.ElementTree as ET

from os.path import isfile, isdir, basename, join

import os

fieldnames = ['instance', 'width primal', 'width incidence', 'width primal improved', 'width incidence improved',
              'numClauses', 'numVariables', 'instanceSet', 'unique',
              '#models gpusat primal double w 22', 'relative deviation gpusat primal double w 22', 'absolute deviation gpusat primal double w 22',
              '#models gpusat incidence double w 22', 'relative deviation gpusat incidence double w 22', 'absolute deviation gpusat incidence double w 22',
              '#models gpusat primal double w 24', 'relative deviation gpusat primal double w 24', 'absolute deviation gpusat primal double w 24',
              '#models gpusat incidence double w 24', 'relative deviation gpusat incidence double w 24', 'absolute deviation gpusat incidence double w 24',
              '#models sharpSAT']

fieldnamesWidth = ['file_name', 'width_primal_s1234', "width_incidence_s1234", "max_clause_size", "max_var_occ", 'width_primal_simplepreproc_s1234',
                   "width_incidence_simplepreproc_s1234", "sat-unsat", "max_clause_size_simplepreproc", "max_var_occ_simplepreproc"]

summaryFile = './Summary_Benchmarks_models_2.csv'
widthFile = './Summary_Benchmark_Width_Ready.csv'
dirFormula = "./benchmarks/ready"
dirIncidence = "./benchmarks/decomp/incidence"
dirPrimal = "./benchmarks/decomp/primal"

if not isfile(summaryFile):
    with open(summaryFile, 'w', newline='\n') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

tree_cobra = ET.parse('./eval_cobra.xml').getroot()
tree_hermann = ET.parse('./eval_hermann_2.xml').getroot()

instanceset = {}
dirs = ["../../BenchmarksGPUSAT/BenchmarkSets"]
for d in dirs:
    if isdir(d):
        dirs += [d + "/" + i for i in os.listdir(d)]
    elif isfile(d):
        instanceset[basename(d).split(".")[0]] = os.path.dirname(d)[37:]

decimal.getcontext().prec = 70


def getDeviation(solver, row):
    if not '#models sharpSAT' in row:
        row['relative deviation ' + solver] = 0
        row['absolute deviation ' + solver] = 0
    elif '#models sharpSAT' in row and row['#models sharpSAT'] > 0:
        row['relative deviation ' + solver] = abs(1 - (row['#models ' + solver] / row['#models sharpSAT']))
        row['absolute deviation ' + solver] = row['#models ' + solver] - row['#models sharpSAT']
    elif '#models sharpSAT' in row and row['#models sharpSAT'] == 0 and row['#models ' + solver] == 0:
        row['relative deviation ' + solver] = 0
        row['absolute deviation ' + solver] = 0


with open(widthFile, 'r') as csvf_:
    r = csv.DictReader(csvf_, fieldnames=fieldnamesWidth)
    next(r, None)
    for reader in r:
        i = reader['file_name'] + '.cnf'
        solved = 0
        solvedgpusat = 0
        solver = ''
        row = {'instance': i}
        row['width primal'] = int(reader['width_primal_s1234'])-1
        row['width incidence'] = int(reader['width_incidence_s1234'])-1

        with open(join(dirFormula, i), 'r')as formula:
            for n in formula.read().splitlines():
                if n.startswith('p'):
                    row['numClauses'] = n.split()[3]
                    row['numVariables'] = n.split()[2]
                    break
        if (i.split(".")[0] in instanceset):
            row['instanceSet'] = instanceset[i.split(".")[0]]

        with open(join(dirIncidence, i), 'r')as iGraph:
            for n in iGraph.read().splitlines():
                if n.startswith('s'):
                    row['width incidence improved'] = int(n.split()[3])-1
                    break

        with open(join(dirPrimal, i), 'r')as pGraph:
            for n in pGraph.read().splitlines():
                if n.startswith('s'):
                    row['width primal improved'] = int(n.split()[3])-1
                    break

        i_id = tree_cobra.findall('.//benchmark/class/instance[@name="' + i + '"]')[0].get('id')
        m = tree_cobra.findall('.//project/runspec[@system="sharpSAT"]/class/instance[@id="' + i_id + '"]/run/measure[@name="#models"]')
        if len(m) > 0:
            row['#models sharpSAT'] = decimal.Decimal(m[0].get('val'))
            solved += 1
            solver = 'sharpSAT'

        i_id = tree_hermann.findall('.//benchmark/class/instance[@name="' + i + '"]')[0].get('id')
        m = tree_hermann.findall('.//project/runspec[@setting="primal_double_w-14_m-24"]/class/instance[@id="' + i_id + '"]/run/measure[@name="#models"]')
        if len(m) > 0:
            row['#models gpusat primal double w 24'] = decimal.Decimal(m[0].get('val'))
            solvedgpusat = 1
            solver = "gpusat primal double w 24"
            getDeviation(solver, row)
            solver = "gpusat"
            if float(m[0].get('val')) > 0 and float(m[0].get('val')) < 1:
                print("ERROR gpusat primal double: " + row['instance'])
        m = tree_hermann.findall('.//project/runspec[@setting="incidence_double_w-14_m-24"]class/instance[@id="' + i_id + '"]/run/measure[@name="#models"]')
        if len(m) > 0:
            row['#models gpusat incidence double w 24'] = decimal.Decimal(m[0].get('val'))
            solvedgpusat = 1
            solver = "gpusat incidence double w 24"
            getDeviation(solver, row)
            solver = "gpusat"
            if float(m[0].get('val')) > 0 and float(m[0].get('val')) < 1:
                print("ERROR gpusat primal double: " + row['instance'])

        m = tree_hermann.findall('.//project/runspec[@setting="primal_double_w-14_m-22"]/class/instance[@id="' + i_id + '"]/run/measure[@name="#models"]')
        if len(m) > 0:
            row['#models gpusat primal double w 22'] = decimal.Decimal(m[0].get('val'))
            solvedgpusat = 1
            solver = "gpusat primal double w 22"
            getDeviation(solver, row)
            solver = "gpusat"
            if float(m[0].get('val')) > 0 and float(m[0].get('val')) < 1:
                print("ERROR gpusat primal double: " + row['instance'])
        m = tree_hermann.findall('.//project/runspec[@setting="incidence_double_w-14_m-22"]class/instance[@id="' + i_id + '"]/run/measure[@name="#models"]')
        if len(m) > 0:
            row['#models gpusat incidence double w 22'] = decimal.Decimal(m[0].get('val'))
            solvedgpusat = 1
            solver = "gpusat incidence double w 22"
            getDeviation(solver, row)
            solver = "gpusat"
            if float(m[0].get('val')) > 0 and float(m[0].get('val')) < 1:
                print("ERROR gpusat primal double: " + row['instance'])

        if (solved + solvedgpusat) == 1:
            row["unique"] = solver
        with open(summaryFile, 'a', newline='\n') as csvf:
            wr = csv.DictWriter(csvf, fieldnames=fieldnames)
            wr.writerow(row)
            csvf.flush()
