#!/usr/bin/env python3
import csv
import xml.etree.ElementTree as ET

fieldnames = ['solver', 'solver_version', 'setting', 'error', 'instance', '#models', 'time', 'timeout', 'SAT']

summaryFile = './Summary_Benchmarks.csv'

with open(summaryFile, 'w', newline='\n') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

for treeString in ['./eval_cobra_dynasp.xml', './eval_cobra.xml', './eval_hermann_2.xml',"./eval_cobra_dynqbfa.xml"]:
    tree = ET.parse(treeString)
    root = tree.getroot()
    specs = root.findall('.//project/runspec/')
    for sp in specs:
        runs = sp.findall('.//run')
        for run in runs:
            measures = run.findall('.//measure')
            row = {}
            for measure in measures:
                if measure.get('name') in fieldnames:
                    row[measure.get('name')] = measure.get('val')
            with open(summaryFile, 'a', newline='\n') as csvf:
                wr = csv.DictWriter(csvf, fieldnames=fieldnames)
                wr.writerow(row)
                csvf.flush()
