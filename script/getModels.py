#!/usr/bin/env python3
import csv
import decimal
import xml.etree.ElementTree as ET

from os.path import isfile, isdir, basename, join

import os

fieldnames = [
    'instance', 'SAT',
    'widthpg', 'time gpusat pd w14 m24',
    'widthpgi', 'time gpusat pd w14 m24 i',
    'widthpd', 'time dynasp primal',
    'widthig', 'time gpusat id w14 m24',
    'widthigi', 'time gpusat id w14 m24 i',
    'widthid', 'time dynasp incidence',
    'time dynqbfe',
    'time dynqbfa',
    'time clasp_sat',
    'time clasp_asp',

    'time clasp',
    'time lingeling',
    'time minisat',
    'time picosat',
    'time sharpSAT',
    'time cachet',
    'time dsharp',
    'time cryptominisat',
    'time approxmc',

    'numClauses', 'numVariables', 'instanceSet',

    '#models gpusat pd w14 m24', 'relative deviation gpusat pd w14 m24', 'absolute deviation gpusat pd w14 m24',

    '#models gpusat id w14 m24', 'relative deviation gpusat id w14 m24', 'absolute deviation gpusat id w14 m24',

    '#models gpusat pd w14 m24 i', 'relative deviation gpusat pd w14 m24 i',
    'absolute deviation gpusat pd w14 m24 i',

    '#models gpusat id w14 m24 i', 'relative deviation gpusat id w14 m24 i',
    'absolute deviation gpusat id w14 m24 i',

    '#models dynqbfe', 'relative deviation dynqbfe', 'absolute deviation dynqbfe',
    '#models dynqbfa', 'relative deviation dynqbfa', 'absolute deviation dynqbfa',

    '#models dynasp primal', 'relative deviation dynasp primal', 'absolute deviation dynasp primal',

    '#models dynasp incidence', 'relative deviation dynasp incidence', 'absolute deviation dynasp incidence',

    '#models clasp_asp', 'relative deviation clasp_asp', 'absolute deviation clasp_asp',

    '#models clasp_sat', 'relative deviation clasp_sat', 'absolute deviation clasp_sat',

    '#models approxmc', 'relative deviation approxmc', 'absolute deviation approxmc',

    '#models cachet', '#models dsharp', '#models sharpSAT'
]

fieldnamesWidth = ["file_name", "width_primal_s1234", "width_incidence_s1234", "max_clause_size", "max_var_occ",
                   "width_primal_s1234_satelite", "width_incidence_s1234_satelite", "max_clause_size_satelite",
                   "max_var_occ_satelite", "sat-unsat", "#models sharpSAT"]

summaryFile = './Results_all.csv'
widthFile = './Summary_Benchmark_Width_Ready.csv'
dirFormula = "./benchmarks_old/ready"
dirIncidence = "./benchmarks_old/decomp_improved/incidence"
dirPrimal = "./benchmarks_old/decomp_improved/primal"

if not isfile(summaryFile):
    with open(summaryFile, 'w', newline='\n') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

tree_cobra = ET.parse('./eval_cobra.xml').getroot()
tree_hermann = ET.parse('./eval_hermann_2.xml').getroot()
tree_hermann_improved = ET.parse('./eval_hermann_improved.xml').getroot()
tree_clasp_asp = ET.parse('./eval_clasp_asp.xml').getroot()
tree_clasp_sat = ET.parse('./eval_clasp_sat.xml').getroot()
tree_dynasp = ET.parse('./eval_cobra_dynasp.xml').getroot()
tree_dynqbfa = ET.parse('./eval_cobra_dynqbfa.xml').getroot()
tree_dynqbfe = ET.parse('./eval_cobra_dynqbfe.xml').getroot()

instanceset = {}
dirs = ["../../BenchmarksGPUSAT/BenchmarkSets"]
for d in dirs:
    if isdir(d):
        dirs += [d + "/" + i for i in os.listdir(d)]
    elif isfile(d):
        instanceset[basename(d).split(".")[0]] = os.path.dirname(d)[37:]

decimal.getcontext().prec = 70


def getDeviation(solver, row):
    if '#models ' + solver in row:
        if not '#models sharpSAT' in row:
            row['relative deviation ' + solver] = 0
            row['absolute deviation ' + solver] = 0
        elif '#models sharpSAT' in row and row['#models sharpSAT'] > 0:
            row['relative deviation ' + solver] = abs(1 - (row['#models ' + solver] / row['#models sharpSAT']))
            row['absolute deviation ' + solver] = row['#models ' + solver] - row['#models sharpSAT']
        elif '#models sharpSAT' in row and row['#models sharpSAT'] == 0 and row['#models ' + solver] == 0:
            row['relative deviation ' + solver] = 0
            row['absolute deviation ' + solver] = 0


def getStatisticsSystemApproxmc(solver, row, setting, tree, instance, rowTimeout, rowSum):
    i_id = tree_cobra.findall('.//benchmark/class/instance[@name="' + instance + '"]')[0].get('id')
    base = tree.findall(
        './/project/runspec[@system="' + setting + '"]class/instance[@id="' + i_id + '"]/run')
    if len(base) > 0:
        m = base[0].findall('.//measure[@name="#models"]')
        if len(m) > 0:
            mult = int(m[0].get('val').split("x")[0])
            expbase = int(m[0].get('val').split("x")[1].split("^")[0])
            exp = int(m[0].get('val').split("x")[1].split("^")[1])
            row['#models ' + solver] = mult * pow(expbase, exp)
        m = base[0].findall('.//measure[@name="time"]')
        if len(m) > 0:
            row['time ' + solver] = ('%.3f' % (float(m[0].get('val'))))
            row['time ' + solver] = ' ' * (8 - len(row['time ' + solver])) + row['time ' + solver]
            rowSum['time ' + solver] += float(m[0].get('val'))
        m = base[0].findall('.//measure[@name="timeout"]')
        if len(m) > 0:
            rowTimeout['time ' + solver] += int(m[0].get('val'))


def getStatisticsSystem(solver, row, setting, tree, instance, rowTimeout, rowSum):
    i_id = tree_cobra.findall('.//benchmark/class/instance[@name="' + instance + '"]')[0].get('id')
    base = tree.findall(
        './/project/runspec[@system="' + setting + '"]class/instance[@id="' + i_id + '"]/run')
    if len(base) > 0:
        m = base[0].findall('.//measure[@name="#models"]')
        if len(m) > 0:
            row['#models ' + solver] = decimal.Decimal(m[0].get('val'))
        m = base[0].findall('.//measure[@name="time"]')
        if len(m) > 0:
            row['time ' + solver] = ('%.3f' % (float(m[0].get('val'))))
            row['time ' + solver] = ' ' * (8 - len(row['time ' + solver])) + row['time ' + solver]
            rowSum['time ' + solver] += float(m[0].get('val'))
        m = base[0].findall('.//measure[@name="timeout"]')
        if len(m) > 0:
            rowTimeout['time ' + solver] += int(m[0].get('val'))


def getStatisticsSetting(solver, row, setting, tree, instance, rowTimeout, rowSum):
    i_id = tree_cobra.findall('.//benchmark/class/instance[@name="' + instance + '"]')[0].get('id')
    base = tree.findall(
        './/project/runspec[@setting="' + setting + '"]class/instance[@id="' + i_id + '"]/run')
    if len(base) > 0:
        m = base[0].findall('.//measure[@name="#models"]')
        if len(m) > 0:
            row['#models ' + solver] = decimal.Decimal(m[0].get('val'))
        m = base[0].findall('.//measure[@name="time"]')
        if len(m) > 0:
            row['time ' + solver] = ('%.3f' % (float(m[0].get('val'))))
            row['time ' + solver] = ' ' * (8 - len(row['time ' + solver])) + row['time ' + solver]
            rowSum['time ' + solver] += float(m[0].get('val'))
        m = base[0].findall('.//measure[@name="timeout"]')
        if len(m) > 0:
            rowTimeout['time ' + solver] += int(m[0].get('val'))


def process(rowTimeout, rowSum, reader):
    i = reader['file_name'] + '.cnf'
    row = {'instance': i}
    row['widthpg'] = int(reader['width_primal_s1234']) - 1
    row['widthig'] = int(reader['width_incidence_s1234']) - 1
    row['SAT'] = reader['sat-unsat']
    if row['SAT'] == 'UNSATISFIABLE':
        row['SAT'] = 'UNSAT'
    elif row['SAT'] == 'SATISFIABLE':
        row['SAT'] = 'SAT'
    else:
        row['SAT'] = ''
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
                row['widthigi'] = int(n.split()[3]) - 1
                break

    with open(join(dirPrimal, i), 'r')as pGraph:
        for n in pGraph.read().splitlines():
            if n.startswith('s'):
                row['widthpgi'] = int(n.split()[3]) - 1
                break

    getStatisticsSystem("lingeling", row, "lingeling", tree_cobra, i, rowTimeout, rowSum)
    getStatisticsSystem("minisat", row, "minisat", tree_cobra, i, rowTimeout, rowSum)
    getStatisticsSystem("picosat", row, "picosat", tree_cobra, i, rowTimeout, rowSum)
    getStatisticsSystem("sharpSAT", row, "sharpSAT", tree_cobra, i, rowTimeout, rowSum)
    getStatisticsSystem("cachet", row, "cachet", tree_cobra, i, rowTimeout, rowSum)
    getStatisticsSystem("dsharp", row, "dsharp", tree_cobra, i, rowTimeout, rowSum)
    getStatisticsSystem("cryptominisat", row, "cryptominisat", tree_cobra, i, rowTimeout, rowSum)
    getStatisticsSystem("clasp", row, "clasp", tree_cobra, i, rowTimeout, rowSum)
    getStatisticsSystemApproxmc("approxmc", row, "approxmc", tree_cobra, i, rowTimeout, rowSum)
    getDeviation("approxmc", row)

    getStatisticsSetting("gpusat pd w14 m24", row, "primal_double_w-14_m-24", tree_hermann, i, rowTimeout, rowSum)
    getDeviation("gpusat pd w14 m24", row)
    getStatisticsSetting("gpusat id w14 m24", row, "incidence_double_w-14_m-24", tree_hermann, i, rowTimeout, rowSum)
    getDeviation("gpusat id w14 m24", row)

    getStatisticsSetting("gpusat pd w14 m24 i", row, "primal_double_w-14_m-24", tree_hermann_improved, i, rowTimeout, rowSum)
    getDeviation("gpusat pd w14 m24 i", row)
    getStatisticsSetting("gpusat id w14 m24 i", row, "incidence_double_w-14_m-24", tree_hermann_improved, i, rowTimeout, rowSum)
    getDeviation("gpusat id w14 m24 i", row)

    getStatisticsSetting("clasp_asp", row, "count_asp", tree_clasp_asp, i, rowTimeout, rowSum)
    getDeviation("clasp_asp", row)

    getStatisticsSetting("clasp_sat", row, "count_sat", tree_clasp_sat, i, rowTimeout, rowSum)
    getDeviation("clasp_sat", row)

    getStatisticsSetting("dynqbfa", row, "dynqbf_all_s1234", tree_dynqbfa, i, rowTimeout, rowSum)
    getDeviation("dynqbfa", row)

    getStatisticsSetting("dynqbfe", row, "dynqbf_existential_s1234", tree_dynqbfe, i, rowTimeout, rowSum)
    getDeviation("dynqbfe", row)

    getStatisticsSetting("dynasp incidence", row, "dynasp_incidence", tree_dynasp, i, rowTimeout, rowSum)
    getDeviation("dynasp incidence", row)
    getStatisticsSetting("dynasp primal", row, "dynasp_primal", tree_dynasp, i, rowTimeout, rowSum)
    getDeviation("dynasp primal", row)

    i_id = tree_dynasp.findall('.//benchmark/class/instance[@name="' + i + '"]')[0].get('id')
    m = tree_dynasp.findall('.//project/runspec[@setting="dynasp_incidence"]/class/instance[@id="' + i_id + '"]/run/measure[@name="width"]')
    if len(m) > 0:
        row['widthid'] = m[0].get('val')
    m = tree_dynasp.findall('.//project/runspec[@setting="dynasp_primal"]/class/instance[@id="' + i_id + '"]/run/measure[@name="width"]')
    if len(m) > 0:
        row['widthpd'] = m[0].get('val')

    with open(summaryFile, 'a', newline='\n') as csvf:
        wr = csv.DictWriter(csvf, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerow(row)
        csvf.flush()


with open(widthFile, 'r') as csvf_:
    r = csv.DictReader(csvf_, fieldnames=fieldnamesWidth)
    next(r, None)
    rowTimeout = {}
    rowSum = {}
    rowTimeout['instance'] = "Timeouts"
    rowSum['instance'] = "Sum"
    for a in ['time gpusat pd w14 m24', 'time gpusat pd w14 m24 i', 'time dynasp primal', 'time gpusat id w14 m24', 'time gpusat id w14 m24 i',
              'time dynasp incidence', 'time clasp_asp', 'time clasp_sat', 'time clasp', 'time lingeling', 'time minisat', 'time picosat', 'time sharpSAT',
              'time cachet', 'time dsharp', 'time cryptominisat', 'time approxmc', 'time dynqbfe', 'time dynqbfa']:
        rowSum[a] = 0
        rowTimeout[a] = 0
    for a in r:
        process(rowTimeout, rowSum, a)
    for a in ['time gpusat pd w14 m24', 'time gpusat pd w14 m24 i', 'time dynasp primal', 'time gpusat id w14 m24', 'time gpusat id w14 m24 i',
              'time dynasp incidence', 'time clasp_asp', 'time clasp_sat', 'time clasp', 'time lingeling', 'time minisat', 'time picosat', 'time sharpSAT',
              'time cachet', 'time dsharp', 'time cryptominisat', 'time approxmc', 'time dynqbfe', 'time dynqbfa']:
        rowSum[a] = str(rowSum[a])
        rowTimeout[a] = str(rowTimeout[a])
    with open(summaryFile, 'a', newline='\n') as csvf:
        wr = csv.DictWriter(csvf, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerow({})
        wr.writerow(rowSum)
        wr.writerow(rowTimeout)
        csvf.flush()
