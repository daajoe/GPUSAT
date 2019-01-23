import csv
import decimal
import os
from os import listdir
from os.path import isfile, isdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


def sortSecond(val):
    return val[1]


plot_dir = "./plots/"
eval_dir = "./evals/"

refFile = './results/eval_cobra_All_Count.xml'
# summaryFiles = ['./results/eval_cobra_All_Count_pre.xml', './results/eval_gpusat_pre.xml', './results/all_Count_pre.xml', './results/eval_gpusat_map_new.xml']
summaryFiles = []

for a in listdir(eval_dir):
    summaryFiles += [eval_dir + a]

decompFile = './results/eval_htd_all_Count_pre.xml'

fieldnames = ['solver', 'solver_version', 'setting', 'error', 'instance', '#models', 'time', 'timeout', 'SAT']
field_names_models = set()

# width:
instanceWidth = {}
tree = ET.parse(decompFile)
root = tree.getroot()

specs = root.findall('.//project/runspec/')
for sp in specs:
    runs = sp.findall('.//run')
    for run in runs:
        measures = run.findall('.//measure')
        row = {}
        for measure in measures:
            if measure.get('name') in ['solver', 'instance', 'width']:
                row[measure.get('name')] = measure.get('val')
        if 'width' in row:
            instanceWidth[row['instance']] = float(row['width'])

# Reference counts:
modelCounts = {}
tree = ET.parse(refFile)
root = tree.getroot()

compSolvers = ['c2d', 'd4', 'countAntom']
specs = root.findall('.//project/runspec/')
for sp in specs:
    runs = sp.findall('.//run')
    for run in runs:
        measures = run.findall('.//measure')
        row = {}
        for measure in measures:
            if measure.get('name') in ['solver', 'instance', '#models']:
                row[measure.get('name')] = measure.get('val')
        if '#models' in row and row['instance'] not in modelCounts and row['solver'] in compSolvers:
            modelCounts[row['instance']] = decimal.Decimal(row['#models'])

# Iterate over all solvers
allData = {}
for summaryFile in summaryFiles:
    tree = ET.parse(summaryFile)
    root = tree.getroot()
    systems = root.findall('.//system')
    for system in systems:
        settings = system.findall('.//setting')
        for setting in settings:
            allData[system.get('name') + "-" + system.get('version') + "-" + setting.get('name')] = {}

    specs = root.findall('.//project/runspec/')
    for sp in specs:
        runs = sp.findall('.//run')
        for run in runs:
            measures = run.findall('.//measure')
            row = {}
            for measure in measures:
                if measure.get('name') in fieldnames:
                    row[measure.get('name')] = measure.get('val')
            allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']] = {}
            if 'time' in row:
                allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']]['time'] = float(row['time'])
            else:
                allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']]['time'] = 901
            if 'timeout' in row:
                allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']]['timeout'] = row['timeout']
            else:
                allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']]['timeout'] = 1
            if '#models' in row and float(row['time']) < 900:
                if 'x' in row['#models']:
                    mult = decimal.Decimal(row['#models'].split('x')[0])
                    if '^' in row['#models']:
                        base = decimal.Decimal(row['#models'].split('x')[1].split('^')[0])
                        exponent = decimal.Decimal(row['#models'].split('x')[1].split('^')[1])
                    allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']]['#models'] = mult * base ** exponent
                else:
                    models = row['#models'].replace(",", "")
                    allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']]['#models'] = decimal.Decimal(models)
                if row['instance'] in modelCounts:
                    test = modelCounts[row['instance']]
                    test_ = (allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']]['#models'])
                    test2 = test - test_
                    if modelCounts[row['instance']] > 0:
                        allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']]['deviation'] = abs((allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']]['#models'] - modelCounts[row['instance']]) / modelCounts[row['instance']])
                    elif allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']]['#models'] == modelCounts[row['instance']]:
                        allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']]['deviation'] = 0
                    else:
                        allData[row['solver'] + "-" + row['solver_version'] + "-" + row['setting']][row['instance']]['deviation'] = -1

for i in allData:
    field_names_models.add(i)
field_names_models = ['instance'] + list(field_names_models)

with open('../plots_/deviations.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names_models)
    writer.writeheader()

for a in modelCounts:
    row = {'instance': a}
    for i in allData:
        if 'deviation' in allData[i][a]:
            row[i] = allData[i][a]['deviation']
    with open(plot_dir + 'deviations.csv', 'a', newline='') as csvf:
        wr = csv.DictWriter(csvf, fieldnames=field_names_models)
        wr.writerow(row)
        csvf.flush()

for i in allData:
    for a in allData[i]:
        if 'BE_linux' not in i and 'pmc_linux' not in i and 'run_gpusat_B-E' not in i:
            if 'BE_linux-1-BE_linux' in allData:
                allData[i][a]['time'] += allData['BE_linux-1-BE_linux'][a]['time']
            elif 'pmc_linux-1-pmc_linux' in allData:
                allData[i][a]['time'] += allData['pmc_linux-1-pmc_linux'][a]['time']
        allData[i][a]['time'] = min(allData[i][a]['time'], 900)

for y in [30, 35, 40, 45, 50]:
    types = []
    for i in allData:
        if not 'BE_linux-1-BE_linux' in i and 'pmc_linux' not in i:  # and not 'sharpCDCL' in i:  # and not 'approxmc' in i and not 'cachet' in i and not 'cnf2eadt' in i and not 'dsharp' in i :
            frame = sorted([allData[i][x]['time'] for x in allData[i] if x in instanceWidth and instanceWidth[x] <= y and instanceWidth[x] > 0])
            types += [(i, len([a for a in allData[i] if int(allData[i][a]['time']) < 900 and a in instanceWidth and instanceWidth[a] <= y and instanceWidth[a] > 0]), pd.DataFrame(frame))]

    types.sort(key=sortSecond, reverse=True)
    frame = pd.concat([a[2] for a in types], ignore_index=True, axis=1)
    ax = frame.plot(figsize=(16, 9))
    ax.legend([a[0] for a in types], loc='upper left')
    ax.set_xlim(700, 1300)
    plt.savefig(plot_dir + 'plot_' + str(y) + '.pdf')

types = []
for i in allData:
    if not 'BE_linux-1-BE_linux' in i:
        frame = sorted([allData[i][x]['time'] for x in allData[i]])
        types += [(i, len([a for a in allData[i] if int(allData[i][a]['time']) < 900]), pd.DataFrame(frame))]

types.sort(key=sortSecond, reverse=True)

for a in range(0, len(types)):
    print('' + str(types[a][0]) + ' ' + str(types[a][1]))

frame = pd.concat([a[2] for a in types], ignore_index=True, axis=1)
ax = frame.plot(figsize=(16, 9))
ax.legend([a[0] for a in types], loc='upper left')
ax.set_xlim(700, 1300)
plt.savefig(plot_dir + 'plot.pdf')
plt.show()
