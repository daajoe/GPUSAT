INSTANCES_DIR=~/instances/
GPUSAT_ORIG=~/GPUSAT_orig/build/gpusat
GPUSAT_NEW=~/GPUSAT/build/gpusat

INSTANCES=$(shell find ~/instances/counting-benchmarks/basic/application -name "*.cnf.bz2")
SHELL=/bin/bash

check/%: %
	echo "checking $*"
	diff <(bzcat $* | $(GPUSAT_ORIG) --dataStructure tree) <(bzcat $* | $(GPUSAT_NEW) --dataStructure tree)

check: $(foreach instance,$(INSTANCES),check/$(instance))
	echo "all checked."
    

.PHONY: check
