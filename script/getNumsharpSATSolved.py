import csv

fieldnames = ['instance', 'SAT', 'width_primal', 'width_incidence', "num_Atoms", "num_Clauses", "num_Variables",
              '#models approxmc clean', 'relative deviation approxmc clean', 'absolute deviation approxmc clean',
              '#models sharpSAT clean', 'relative deviation sharpSAT clean', 'absolute deviation sharpSAT clean',
              '#models sharpSAT preproc', 'relative deviation sharpSAT preproc', 'absolute deviation sharpSAT preproc',
              '#models sharpSAT']

summaryFile = './Summary_Benchmarks_models_all.csv'

numSolvedPrimal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
numSolvedIncidence = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
numSolvedModels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

with open(summaryFile, 'r') as csvf_:
    r = csv.DictReader(csvf_, fieldnames=fieldnames)
    next(r, None)
    for reader in r:
        if reader['#models sharpSAT clean'] is not '':
            if int(reader['width_primal']) <= 0:
                numSolvedPrimal[0] += 1
            if 0 < int(reader['width_primal']) <= 30:
                numSolvedPrimal[1] += 1
            if 30 < int(reader['width_primal']) <= 40:
                numSolvedPrimal[2] += 1
            if 40 < int(reader['width_primal']) <= 50:
                numSolvedPrimal[3] += 1
            if 50 < int(reader['width_primal']) <= 80:
                numSolvedPrimal[4] += 1
            if 80 < int(reader['width_primal']) <= 100:
                numSolvedPrimal[5] += 1
            if 100 < int(reader['width_primal']) <= 150:
                numSolvedPrimal[6] += 1
            if 150 < int(reader['width_primal']) <= 200:
                numSolvedPrimal[7] += 1
            if 200 < int(reader['width_primal']) <= 300:
                numSolvedPrimal[8] += 1
            if 300 < int(reader['width_primal']) <= 500:
                numSolvedPrimal[9] += 1
            if 500 < int(reader['width_primal']) <= 1000:
                numSolvedPrimal[10] += 1
            if 1000 < int(reader['width_primal']):
                numSolvedPrimal[11] += 1

            if int(reader['width_incidence']) <= 0:
                numSolvedIncidence[0] += 1
            if 0 < int(reader['width_incidence']) <= 30:
                numSolvedIncidence[1] += 1
            if 30 < int(reader['width_incidence']) <= 40:
                numSolvedIncidence[2] += 1
            if 40 < int(reader['width_incidence']) <= 50:
                numSolvedIncidence[3] += 1
            if 50 < int(reader['width_incidence']) <= 80:
                numSolvedIncidence[4] += 1
            if 80 < int(reader['width_incidence']) <= 100:
                numSolvedIncidence[5] += 1
            if 100 < int(reader['width_incidence']) <= 150:
                numSolvedIncidence[6] += 1
            if 150 < int(reader['width_incidence']) <= 200:
                numSolvedIncidence[7] += 1
            if 200 < int(reader['width_incidence']) <= 300:
                numSolvedIncidence[8] += 1
            if 300 < int(reader['width_incidence']) <= 500:
                numSolvedIncidence[9] += 1
            if 500 < int(reader['width_incidence']) <= 1000:
                numSolvedIncidence[10] += 1
            if 1000 < int(reader['width_incidence']):
                numSolvedIncidence[11] += 1

            if int(reader['#models sharpSAT clean']) <= 0:
                numSolvedModels[0] += 1
            if 0 < int(reader['#models sharpSAT clean']) <= 1000:
                numSolvedModels[1] += 1
            if 1000 < int(reader['#models sharpSAT clean']) <= 2000:
                numSolvedModels[2] += 1
            if 2000 < int(reader['#models sharpSAT clean']) <= 5000:
                numSolvedModels[3] += 1
            if 5000 < int(reader['#models sharpSAT clean']) <= 10000:
                numSolvedModels[4] += 1
            if 10000 < int(reader['#models sharpSAT clean']) <= 50000:
                numSolvedModels[5] += 1
            if 50000 < int(reader['#models sharpSAT clean']) <= 100000:
                numSolvedModels[6] += 1
            if 100000 < int(reader['#models sharpSAT clean']) <= 1000000:
                numSolvedModels[7] += 1
            if 1000000 < int(reader['#models sharpSAT clean']) <= 10000000:
                numSolvedModels[8] += 1
            if 10000000 < int(reader['#models sharpSAT clean']) <= 100000000:
                numSolvedModels[9] += 1
            if 100000000 < int(reader['#models sharpSAT clean']):
                numSolvedModels[10] += 1

print("Solved by SharpSAT primal width:")
print("???: " + str(numSolvedPrimal[0]))
print("30: " + str(numSolvedPrimal[1]))
print("40: " + str(numSolvedPrimal[2]))
print("50: " + str(numSolvedPrimal[3]))
print("80: " + str(numSolvedPrimal[4]))
print("100: " + str(numSolvedPrimal[5]))
print("150: " + str(numSolvedPrimal[6]))
print("200: " + str(numSolvedPrimal[7]))
print("300: " + str(numSolvedPrimal[8]))
print("500: " + str(numSolvedPrimal[9]))
print("1000: " + str(numSolvedPrimal[10]))
print("1000+: " + str(numSolvedPrimal[11]))
print("\nSum: " + str(sum(numSolvedPrimal)) + "\n")

print("Solved by SharpSAT incidence width:")
print("???: " + str(numSolvedIncidence[0]))
print("30: " + str(numSolvedIncidence[1]))
print("40: " + str(numSolvedIncidence[2]))
print("50: " + str(numSolvedIncidence[3]))
print("80: " + str(numSolvedIncidence[4]))
print("100: " + str(numSolvedIncidence[5]))
print("150: " + str(numSolvedIncidence[6]))
print("200: " + str(numSolvedIncidence[7]))
print("300: " + str(numSolvedIncidence[8]))
print("500: " + str(numSolvedIncidence[9]))
print("1000: " + str(numSolvedIncidence[10]))
print("1000+: " + str(numSolvedIncidence[11]))
print("\nSum: " + str(sum(numSolvedIncidence)) + "\n")

print("Solved by SharpSAT models:")
print("0: " + str(numSolvedIncidence[0]))
print("1000: " + str(numSolvedIncidence[1]))
print("2000: " + str(numSolvedIncidence[2]))
print("5000: " + str(numSolvedIncidence[3]))
print("10000: " + str(numSolvedIncidence[4]))
print("50000: " + str(numSolvedIncidence[5]))
print("100000: " + str(numSolvedIncidence[6]))
print("1000000: " + str(numSolvedIncidence[7]))
print("10000000: " + str(numSolvedIncidence[8]))
print("100000000: " + str(numSolvedIncidence[9]))
print("100000000+: " + str(numSolvedIncidence[10]))
print("\nSum: " + str(sum(numSolvedIncidence)) + "\n")
