#!/bin/bash
#
#SBATCH -N 2
#SBATCH -n 32
#SBATCH -p normal
#SBATCH -J rideaustin
#SBATCH -t 2:00:00
#SBATCH -A Measuring-exposure-t

#
export JULIAPATH="julia"
export SCRIPT="/work/05863/mgarciat/stampede2/gfen-reproduce/7_bayes_modelfit_script.jl"
export JULIA_NUM_THREADS=8

#
srun $JULIAPATH $SCRIPT
