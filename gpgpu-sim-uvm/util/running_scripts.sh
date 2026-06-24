#!/bin/bash

# For running this script, the configurations to be running have to be provided, and optionally a name to track the status of the jobs while running
# if no name is provided, a default name "test_$configuration" will be used

# For No security/ No modification         -->   ./running_script N chosen_name 
# For Baseline security   -->   ./running_script B chosen_name
# For Salus               -->   ./running_script S chosen name

configuration=$1

if test "$configuration" = "N" 
then
name=${2:-"unmodified"}
./run_simulations.py -B rodinia-3.1_v1 -C QV100-2B_INSN-rHBM -N $name