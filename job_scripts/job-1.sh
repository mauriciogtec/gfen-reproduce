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
srun $JULIAPATH $SCRIPT $WALLTIME $NGENS $GENSIZE $CVSPLITS 00.csv 01.csv 02.csv 03.csv 04.csv 05.csv 06.csv 07.csv 08.csv 09.csv 10.csv 11.csv
