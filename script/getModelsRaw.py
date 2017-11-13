import csv
import decimal
import xml.etree.ElementTree as ET
from os.path import isfile

fieldnames = ['instance', 'SAT', 'width_primal', 'width_incidence', "num_Atoms", "num_Clauses", "num_Variables",
              '#models approxmc clean', 'relative deviation approxmc clean', 'absolute deviation approxmc clean',
              '#models sharpSAT clean', 'relative deviation sharpSAT clean', 'absolute deviation sharpSAT clean',
              '#models sharpSAT preproc', 'relative deviation sharpSAT preproc', 'absolute deviation sharpSAT preproc',
              '#models sharpSAT']

fieldnamesWidth = ['file_name', "num_Variables",
                   'width_primal', "width_incidence", "max_clause_size", "max_var_occ", "num_Atoms", "num_Clauses",
                   'width_primal_preproc', "width_incidence_preproc", "max_clause_size_preproc", "max_var_occ_preproc", "num_Atoms_preproc",
                   "num_Clauses_preproc",
                   "sat-unsat"]

summaryFile = './Summary_Benchmarks_models_all.csv'
widthFile = './benchmarks/Summary_Benchmark_Width.csv'
dirRaw = "./benchmarks_old/raw"

if not isfile(summaryFile):
    with open(summaryFile, 'w', newline='\n') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

tree_cobra_raw = ET.parse('./eval_cobra_raw.xml').getroot()
tree_cobra_clean = ET.parse('./eval_cobra_clean.xml').getroot()
tree_cobra_preproc = ET.parse('./eval_cobra_preproc.xml').getroot()


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
        i = reader['file_name']
        row = {'instance': i, 'SAT': reader['sat-unsat'], 'width_primal': reader['width_primal'], 'width_incidence': reader['width_incidence'],
               'num_Atoms': reader['num_Atoms'], 'num_Clauses': reader['num_Clauses'], 'num_Variables': reader['num_Variables']}

        i_id = tree_cobra_raw.findall('.//benchmark/class/instance[@name="' + i + '"]')[0].get('id')
        m = tree_cobra_raw.findall('.//project/runspec[@system="sharpSAT"]/class/instance[@id="' + i_id + '"]/run/measure[@name="#models"]')
        if len(m) > 0:
            row['#models sharpSAT'] = decimal.Decimal(m[0].get('val'))
            solver = 'sharpSAT'

        i_id = tree_cobra_clean.findall('.//benchmark/class/instance[@name="' + i + '"]')[0].get('id')
        m = tree_cobra_clean.findall('.//project/runspec[@system="sharpSAT"]/class/instance[@id="' + i_id + '"]/run/measure[@name="#models"]')
        if len(m) > 0:
            row['#models sharpSAT clean'] = decimal.Decimal(m[0].get('val'))
            solver = 'sharpSAT clean'
            getDeviation(solver, row)
        m = tree_cobra_clean.findall('.//project/runspec[@system="approxmc"]/class/instance[@id="' + i_id + '"]/run/measure[@name="#models"]')
        if len(m) > 0:
            models = m[0].get('val')
            pre = int(models.split('x')[0])
            exp = int(models.split('^')[1])
            row['#models approxmc clean'] = pre * 2 ^ exp
            solver = "approxmc clean"
            getDeviation(solver, row)

        i_id = tree_cobra_preproc.findall('.//benchmark/class/instance[@name="' + i + '"]')[0].get('id')
        m = tree_cobra_preproc.findall('.//project/runspec[@system="sharpSAT"]/class/instance[@id="' + i_id + '"]/run/measure[@name="#models"]')
        if len(m) > 0:
            row['#models sharpSAT preproc'] = decimal.Decimal(m[0].get('val'))
            solver = 'sharpSAT preproc'
            getDeviation(solver, row)

        with open(summaryFile, 'a', newline='\n') as csvf:
            wr = csv.DictWriter(csvf, fieldnames=fieldnames)
            wr.writerow(row)
            csvf.flush()
