#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 7
#SBATCH -p skx-normal
#SBATCH -J rideaustin
#SBATCH -t 6:0:0
#
export WALLTIME=300.0
export JULIAPATH="julia"
export SCRIPT="C:\Users\mbg877\Google Drive\GFLRideAustin\spt_paper_reproducibility\4_modelfit_script.jl"
export NGENS=4
export GENSIZE=16
export NSPLITS=5
export JULIA_NUM_THREADS=1
#
$JULIAPATH $SCRIPT $WALLTIME $NGENS $GENSIZE $NSPLITS split_26-28.csv split_28-30.csv split_28-32.csv split_30-32.csv split_4-6.csv split_4-8.csv split_6-8.csv
