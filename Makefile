INSTANCES_DIR=~/instances/
GPUSAT_ORIG=~/GPUSAT_orig/build/gpusat
GPUSAT_NEW=~/GPUSAT/build/gpusat

#INSTANCES=$(shell find ~/instances/counting-benchmarks/basic/application -name "*.cnf.bz2")
INSTANCES=$(shell ls -d /home/vroland/instances/cachet/DQMR/* --sort=size -r | grep ".wcnf")
SHELL=/bin/bash
CHECK_DIR=/tmp/check
CHECK_ARGS ?=

space :=
space +=

all: build_Release

$(info args: $(CHECK_ARGS))
.SECONDEXPANSION:
run/orig/%: /$$(subst $$(space),/,$$(wordlist 2,1000,$$(subst /, ,$$*))) $(if $(CHECK_ARGS),,guard)
	echo "orig: $(notdir $@)"
	$(GPUSAT_ORIG) $(CHECK_ARGS) < $< > $(CHECK_DIR)/orig/$(subst $<,,$*)/$(notdir $*) || echo 'failed!'
run/new/%: /$$(subst $$(space),/,$$(wordlist 2,1000,$$(subst /, ,$$*))) $(if $(CHECK_ARGS),,guard)
	echo "new: $(notdir $@)"
	$(GPUSAT_NEW) $(CHECK_ARGS) < $< > $(CHECK_DIR)/new/$(subst $<,,$*)/$(notdir $*) || echo 'failed!'

tree/%: %
	CHECK_ARGS='--dataStructure tree $(if $(subst .wcnf,,$(suffix $*)),,--weighted)' $(MAKE) run/new/tree/$*
	CHECK_ARGS='--dataStructure tree $(if $(subst .wcnf,,$(suffix $*)),,--weighted)' $(MAKE) run/orig/tree/$*
array/%: %
	CHECK_ARGS='--dataStructure array $(if $(subst .wcnf,,$(suffix $*)),,--weighted)' $(MAKE) run/new/array/$*
	CHECK_ARGS='--dataStructure array $(if $(subst .wcnf,,$(suffix $*)),,--weighted)' $(MAKE) run/orig/array/$*

check_dirs:
	mkdir -p $(CHECK_DIR)/new/tree
	mkdir -p $(CHECK_DIR)/new/array
	mkdir -p $(CHECK_DIR)/orig/tree
	mkdir -p $(CHECK_DIR)/orig/array

check: check_dirs $(foreach instance,$(INSTANCES),tree/$(instance) array/$(instance))
	echo "all checked."

build_%: configure_%
	cmake --build build

configure_%:
	mkdir -p build
	(cd build && cmake -DWITH_CLI=On -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=$* -G Ninja ..)

configure_Profile:
	mkdir -p build
	(cd build && cmake -DWITH_CLI=On -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Profile -G Ninja ..)

.PHONY: check_dirs check build configure
