#!/bin/bash

#SBATCH -J c3k_nirspec
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 6:00:00 # Runtime 
#SBATCH -p conroy # Partition to submit to
#SBATCH --mem-per-cpu=2000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -o /n/scratchlfs/conroy_lab/bdjohnson/run_ckc/logs/c3k_jwst_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/scratchlfs/conroy_lab/bdjohnson/run_ckc/logs/c3k_jwst_%A_%a.err # Standard err goes to this file


source activate py2pro_env
cd /n/scratchlfs/conroy_lab/bdjohnson/run_ckc

seddir=c3k_jwst
mkdir $seddir
python make_ckc_nirspec.py --zindex=${SLURM_ARRAY_TASK_ID} --ck_vers=c3k_v1.3 \
                           --seddir=${seddir} --sedname=${seddir} --verbose=False \
                           --oversample=3