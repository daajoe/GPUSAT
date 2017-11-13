#!/usr/bin/env bash
./beval ./runscripts/runscript-gpusat_cobra.xml > eval_cobra.xml
./beval ./runscripts/runscript-gpusat_cobra.xml | ./bconv > result_cobra.ods
