#!/usr/bin/env bash
./beval ./runscripts/runscript-gpusat_hermann.xml > eval_hermann.xml
./beval ./runscripts/runscript-gpusat_hermann.xml | ./bconv > result_hermann.ods
