# This file is helpful for creating jobfiles for a cluster
#   supercomputing environment.

# --- TASK VARIABLES ---
# how many job files to create (targets will be split in files)
# each file will correspond to one cluster or the ones specified
nclusters = 1
procs = 12
# for cluster (supercomputing) environments you need to specify queue
# each environment has different queues
queue = "skx-normal"
# slurm command
cmd = "sbatch"
# julia path, usually in cluster must me specified, if julia is
# on a personal computer and its in the path just write julia
juliapath = "julia"
# juliapath = "/home1/05863/mgarciat/julia-1./julia"
# jobname used by slurm launcher
jobname = "rideaustin"
# max running time
maxjobtime = "12:0:0"

# -- MODEL FIT VARIABLES ---
# variables that will be passed to model_script.jl
workspace = pwd()  # for home computer
modelscript = joinpath(workspace, "4_modelfit_script.jl")
ngens = 16
gensize = 16
num_threads = 5
walltime = 300.0
cvsplits = 5
datadir = joinpath(workspace, "productivity_splits/")
# name for the resulting bash file
launcherfile = joinpath(workspace, "slurm_launcher.sh")
jobdir = joinpath(workspace, "job_scripts")
allocation = "Measuring-exposure-t"

# this is the header of the script
header = """#!/bin/bash
#
#SBATCH -N $nclusters
#SBATCH -n $procs
#SBATCH -p $queue
#SBATCH -J $jobname
#SBATCH -t $maxjobtime
#SBATCH -A $allocation
#
export WALLTIME=$(float(walltime))
export JULIAPATH="$juliapath"
export SCRIPT="$modelscript"
export NGENS=$ngens
export GENSIZE=$gensize
export CVSPLITS=$cvsplits
export JULIA_NUM_THREADS=$num_threads
#
srun \$JULIAPATH \$SCRIPT \$WALLTIME \$NGENS \$GENSIZE \$CVSPLITS"""

if isfile(launcherfile)
    rm(launcherfile)
end
if isdir(jobdir)
    rm(jobdir, recursive=true)
end
mkdir(jobdir)

datafiles = readdir(datadir)
N = length(datafiles)  # must be 2^tree_depth - 1 splits
open(launcherfile, "a") do launcherio
    num_files = if N % procs == 0
        N ÷ procs
    else
        N ÷ procs + 1
    end
    for j in 1:num_files
        # cd into folder and sbatch then go back
        jobfile = joinpath(jobdir, "job-$(j).sh")
        open(jobfile, "w") do jobio
            joblist = []
            for i in 1:procs
                idx = procs * (j - 1) + i
                if idx > N
                    break
                end
                file = datafiles[idx]
                push!(joblist, file)
            end 
            joblist = join(joblist, " ")
            write(jobio, "$header $joblist\n")
        end
        write(launcherio, "$cmd \"$jobfile\"\n")
    end
end


