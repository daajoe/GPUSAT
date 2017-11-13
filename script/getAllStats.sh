#!/usr/bin/env bash
python3 ./getStats.py
python3 ./getModels.py
./Summary.R
Rscript -e "library(knitr); knit('Summary.Rmd')"
