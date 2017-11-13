#!/bin/bash

# Might as well ask for password up-front, right?
sudo -v

# Keep-alive: update existing sudo time stamp if set, otherwise do nothing.
while true; do sudo -v; sleep 300; kill -0 "$$" || exit; done 2>/dev/null &

# start benchmark
./output/gpusat_single_improved/hermann/start.py &> output.txt