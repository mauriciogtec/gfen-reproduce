#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -p skx-normal
#SBATCH -J rideaustin
#SBATCH -t 12:0:0
#SBATCH -A Measuring-exposure-t
#
export WALLTIME=300.0
export JULIAPATH="julia"
export SCRIPT="/work/05863/mgarciat/stampede2/gfen-reproduce/4_modelfit_script.jl"
export NGENS=16
export GENSIZE=16
export CVSPLITS=5
export JULIA_NUM_THREADS=5
#
srun $JULIAPATH $SCRIPT $WALLTIME $NGENS $GENSIZE $CVSPLITS 24.csv 25.csv 26.csv 27.csv 28.csv 29.csv 30.csv 31.csv 32.csv 33.csv 34.csv 35.csv
