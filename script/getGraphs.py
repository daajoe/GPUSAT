from multiprocessing.pool import ThreadPool
from os import makedirs, listdir
from os.path import isdir, join, isfile

import CommonMethods

dirRaw = "./benchmarks/raw"
dirIncidence = "./benchmarks/incidence"
dirPrimal = "./benchmarks/primal"

if not isdir(dirRaw):
    makedirs(dirRaw)
if not isdir(dirIncidence):
    makedirs(dirIncidence)
if not isdir(dirPrimal):
    makedirs(dirPrimal)

numFiles = len(listdir(dirRaw))
currentFile = 1


def process(i):
    global currentFile
    print("(%s/%s): %s" % (str(currentFile), str(numFiles), i))
    currentFile += 1
    with open(join(dirRaw, i), "r") as formulaFile:
        case = i[:-4]
        formula = formulaFile.read()
        formula = '\n'.join([x for x in formula.splitlines() if len(x) > 0 and x[0] != 'c' and x[0] != 'w'])
        if not isfile(join(dirIncidence, case + ".gr")):
            with open(join(dirIncidence, case + ".gr"), "w")as graphFile:
                graphFile.write(CommonMethods.genIncidenceGraph(formula))
        if not isfile(join(dirPrimal, case + ".gr")):
            with open(join(dirPrimal, case + ".gr"), "w")as graphFile:
                graphFile.write(CommonMethods.genPrimalGraph(formula))


pool = ThreadPool(6)
for i in listdir(dirRaw):
    pool.apply_async(process, [i])
    # process(i)
pool.close()
pool.join()
