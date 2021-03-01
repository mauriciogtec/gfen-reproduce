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
srun $JULIAPATH $SCRIPT $WALLTIME $NGENS $GENSIZE $CVSPLITS 12.csv 13.csv 14.csv 15.csv 16.csv 17.csv 18.csv 19.csv 20.csv 21.csv 22.csv 23.csv
