#!/bin/bash

#SBATCH -J c3k_fluxfiles
#SBATCH -n 20 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 6:00:00 # Runtime 
#SBATCH -p conroy # Partition to submit to
#SBATCH --mem-per-cpu=2000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -o /n/scratchlfs/conroy_lab/bdjohnson/run_ckc/logs/c3k_flux_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/scratchlfs/conroy_lab/bdjohnson/run_ckc/logs/c3k_flux_%A_%a.err # Standard err goes to this file


source activate py2pro_env
cd /n/scratchlfs/conroy_lab/bdjohnson/run_ckc

python make_flux.py --np=${SLURM_JOB_CPUS_PER_NODE} --feh=-99 --ck_vers=c3k_v1.3