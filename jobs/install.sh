module purge

module load gcc/7.1.0-fasrc01 hdf5/1.10.1-fasrc02
module load python/3.6.3-fasrc02

conda create -n jades python=3.6 numpy scipy matplotlib ipython

# Activate
source activate jades

conda install astropy
conda install h5py
pip install dynesty

# FSPS
cd ~/codes/fsps
git checkout c3k_jwst
cd src
make clean
make all

# python-FSPS
cd ~/codes/python-fsps
python setup.py install

# sedpy
cd ~/codes/sedpy
python setup.py install

# prospector
cd ~/codes/prospector
python setup.py install

# jadespro
cd ~/codes/jades_sed
python setup.py install

# copy relevant scripts
cd $jadesdir
cp ~/codes/jades_sed/scripts/fitting/fit*py .
cp ~/codes/jades_sed/jobs/job_ody*sh .