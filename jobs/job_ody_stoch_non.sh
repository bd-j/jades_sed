#!/bin/bash

#SBATCH -J stochparfit_jades
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 24:00:00 # Runtime 
#SBATCH -p conroy,shared # Partition to submit to
#SBATCH --mem-per-cpu=2000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -o /n/scratchlfs/conroy_lab/bdjohnson/jades_sed/logs/stochnonfit_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/scratchlfs/conroy_lab/bdjohnson/jades_sed/logs/stochnonfit_%A_%a.err # Standard err goes to this file

source activate jades

catalog=stochastic_mist_ckc14.h5
lsf_file=jwst_nirspec_prism_disp.fits
sgroup=DEEP_R100
nbins_sfh=6
# Do every 10th galaxy
declare -i objid
objid=${SLURM_ARRAY_TASK_ID}*10
echo $objid

outdir=./output
out=$outdir/stochastic_nonparametric_$objid


python ./fit_stochastic_with_nonparametric.py \
        --objid=$objid \
        --add_neb --nbins_sfh=$nbins_sfh \
        --smoothssp=True --sublibres=True \
        --lsf_file=$lsf_file \
        --datafile=$catalog --sgroup=$sgroup \
        --dynesty --nlive_init=400 \
        --outfile=$out