import os
import shutil
import sys
import math
import astropy
import numpy as np
import argparse
from astropy.io import fits
from astropy.table import Table
import matplotlib
import matplotlib.pyplot as plt
import bagpipes as pipes
from collections import defaultdict

#import progressbar
#widgets=[
#	' [', progressbar.ETA(), '] ',
#	progressbar.Bar(),
#	' (', progressbar.Timer(), ') ',
#]

# Speed of light.
c = 2.998e+18

np.random.seed(seed=43985)

# Set the cosmology
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

# This is a hack-ey code for assigning Re_maj and sersic_n values 
def assign_re_maj_sersic():

  mu, sigma = -1.3, 0.28196488286292126 # mean and standard deviation
  log_re_maj = np.random.normal(mu, sigma, 1)[0]
  re_maj = 10**(log_re_maj)

  sersic_prob_array_sum = np.array([0.11400426, 0.26293572, 0.41736112, 0.54845875, 0.63635522,
    0.70527742, 0.7591326 , 0.80336181, 0.83674858, 0.86362328,
    0.88485529, 0.90347585, 0.92075103, 0.93542799, 0.94905376,
    0.96183329, 0.97275506, 0.98225542, 0.99146158, 1.        ])

  sersic_n_array = np.array([0.100021  , 0.47501855, 0.8500161 , 1.22501365, 1.6000112 ,
    1.97500875, 2.3500063 , 2.72500385, 3.1000014 , 3.47499895,
    3.8499965 , 4.22499405, 4.5999916 , 4.97498915, 5.3499867 ,
    5.72498425, 6.0999818 , 6.47497935, 6.8499769 , 7.22497445,
    7.599972  ])

  sersic_random_number = np.random.uniform()

  for n in range(0, len(sersic_prob_array_sum)):
    if (n == 0):
      if ((sersic_random_number > 0) & (sersic_random_number <= sersic_prob_array_sum[n])):
				sersic_n_output = np.random.uniform(low=sersic_n_array[n], high=sersic_n_array[n+1])
    else:
      if ((sersic_random_number > sersic_prob_array_sum[n-1]) & (sersic_random_number <= sersic_prob_array_sum[n])):
        sersic_n_output = np.random.uniform(low=sersic_n_array[n], high=sersic_n_array[n+1])
	
  return re_maj, sersic_n_output

re_maj = np.zeros(10000)
sersic_n = np.zeros(10000)
for n in range(0, 10000):
  re_maj[n], sersic_n[n] = assign_re_maj_sersic()

# Do you want to create only a constant SFH? Or only bursts? Or both?
add_constant = 1
add_bursts = 1

# The output file name 
bagpipes_burst_galaxy_file_name = 'burst_galaxies_constant_SF_z_6_8_TEST'

# Observed redshift range
zmin = 6
zmax = 8

# Burst redshift range
burst_z_min = 9
burst_z_max = 13

# Burst mass range
burst_mass_min = 7
burst_mass_max = 9

# Constant period mass range and redshift start
constant_mass_min = 7
constant_mass_max = 9
constant_z_start_max = 9

# The BAGPIPES/JADES filter file we're using. 
JADES_filt_list = np.loadtxt("/Users/efcl/Dropbox/JWST/data-to-science/stochastic_SFH_example/JADES_All_Filters_BAGPIPES_HSTNIRCam.dat", dtype="str")

number_objects = 10
galaxy_redshift = np.zeros(number_objects)
constant_sf_start_z = np.zeros(number_objects)
constant_mass_formed = np.zeros(number_objects)
burst1_redshift = np.zeros(number_objects)
burst1_mass = np.zeros(number_objects)
burst2_redshift = np.zeros(number_objects)
burst2_mass = np.zeros(number_objects)

HST_F435W = np.zeros(number_objects)
HST_F606W = np.zeros(number_objects)
HST_F775W = np.zeros(number_objects)
HST_F814W = np.zeros(number_objects)
HST_F850LP = np.zeros(number_objects)
NRC_F070W = np.zeros(number_objects)
NRC_F090W = np.zeros(number_objects)
NRC_F115W = np.zeros(number_objects)
NRC_F150W = np.zeros(number_objects)
NRC_F200W = np.zeros(number_objects)
NRC_F277W = np.zeros(number_objects)
NRC_F335M = np.zeros(number_objects)
NRC_F356W = np.zeros(number_objects)
NRC_F410M = np.zeros(number_objects)
NRC_F444W = np.zeros(number_objects)

re_maj = np.zeros(number_objects)
sersic_n = np.zeros(number_objects)

galaxy_ids = np.zeros(number_objects, dtype = int)

beagleParamDict = defaultdict(list)

# I use a progressbar, but I've commented it out. 
#for obj in progressbar.progressbar(range(0,number_objects), term_width=80, widgets=widgets):
for obj in range(0, number_objects):

  # Create the ID
  galaxy_ids[obj] = obj+1

  # Set the random values for the various input parameters for the galaxies.
  galaxy_redshift[obj] = np.random.uniform(zmin, zmax)#  7.0
  burst1_redshift[obj] = np.random.uniform(burst_z_min, burst_z_max)#  12.0
  burst1_mass[obj] = np.random.uniform(burst_mass_min, burst_mass_max)#  8.25527
  burst1_age = cosmo.age(galaxy_redshift[obj]).value - cosmo.age(burst1_redshift[obj]).value
  burst2_redshift[obj] = np.random.uniform(burst_z_min, burst_z_max)#  10.0
  burst2_mass[obj] = np.random.uniform(burst_mass_min, burst_mass_max)#  8.34242
  burst2_age = cosmo.age(galaxy_redshift[obj]).value - cosmo.age(burst2_redshift[obj]).value

  constant_sf_start_z[obj] = np.random.uniform(galaxy_redshift[obj], constant_z_start_max)#  8.0
  age_max_value = cosmo.age(galaxy_redshift[obj]).value - cosmo.age(constant_sf_start_z[obj]).value
  constant_sf_end_z = galaxy_redshift[obj]
  age_min_value = 0.0
  constant_mass_formed[obj] = np.random.uniform(constant_mass_min, constant_mass_max)#  8.0


  # Dust Here:
  dust = {}
  dust["type"] = "Calzetti"
  dust["Av"] = 0.2
  dust["eta"] = 1.  #I can't implement a value of 3. for a Calzetti dust curve in BEAGLE

  # Nebular Emission here:
  nebular = {}
  nebular["logU"] = -3.
	
	# Here's where I create the SFH
  if (add_constant == 1):
    constant = {}                        # tophat function
    constant["age_max"] = age_max_value      # Time since SF switched on: Gyr
    constant["age_min"] = age_min_value      # Time since SF switched off: Gyr
    constant["massformed"] = constant_mass_formed[obj]
    constant["metallicity"] = 0.02
	
  if (add_bursts == 1):
    burst1 = {}
    burst1["age"] = burst1_age
    burst1["massformed"] = burst1_mass[obj]
    burst1["metallicity"] = 0.02

    burst2 = {}
    burst2["age"] = burst2_age
    burst2["massformed"] = burst2_mass[obj]
    burst2["metallicity"] = 0.02


  
	
  # Add the various components to the model
  model_components = {}
  model_components["redshift"] = galaxy_redshift[obj]
  model_components["t_bc"] = 0.01
  model_components["veldisp"] = 50.
  if (add_constant == 1):
    model_components["constant"] = constant
  if (add_bursts == 1):
    model_components["burst1"] = burst1
    model_components["burst2"] = burst2
  model_components["dust"] = dust
  model_components["nebular"] = nebular

  #Make the mock input file for BEAGLE
  beagleParamDict['start_age-1'].append(np.log10(age_max_value*1E9))
  beagleParamDict['end_age-1'].append(np.log10(age_min_value*1E9))
  print age_min_value
  beagleParamDict['mass-1'].append(constant_mass_formed[obj])
  #Assuming you're meaning to make this solar metallicity - BEAGLE has a different
  #definition of solar metallicity at Z=0.0154, and requires log10(Z/Z_sun)
  beagleParamDict['metallicity-1'].append(np.log10(0.02))
  beagleParamDict['ssp_age-2'].append(np.log10(burst1_age*1E9))
  beagleParamDict['mass-2'].append(burst1_mass[obj])
  beagleParamDict['metallicity-2'].append(np.log10(0.02))
  beagleParamDict['ssp_age-3'].append(np.log10(burst2_age*1E9))
  beagleParamDict['mass-3'].append(burst2_mass[obj])
  beagleParamDict['metallicity-3'].append(np.log10(0.02))
  beagleParamDict['redshift'].append(galaxy_redshift[obj])
  beagleParamDict['tauV_eff'].append(dust['Av']/1.086)
  beagleParamDict['nebular_logU'].append(-3) #I'm not sure whether logU has the same parameterisation
                                            #in bagpipes as it does in BEAGLE
	
  # Create the model galaxy
  model = pipes.model_galaxy(model_components, filt_list=JADES_filt_list, spec_wavs=np.arange(6000., 50000., 2.))
  outputSpec = {}
  outputSpec['wl'] = model.spectrum[:,0]
  outputSpec['spec'] = model.spectrum[:,1]
  outputTable = Table(outputSpec)
  outputTable.write("bagpipes_spectra/"+str(galaxy_ids[obj])+"_spectrum.fits", overwrite=True)
  
  # Get the photometry ready for output, putting it into fnu units
  output_phot_wavelengths = model.filter_set.eff_wavs
  output_phot_flam = model.photometry
  output_phot_fnu = model.photometry * (output_phot_wavelengths**2)/c

  HST_F435W[obj] = output_phot_fnu[0]/1e-23/1e-9
  HST_F606W[obj] = output_phot_fnu[1]/1e-23/1e-9
  HST_F775W[obj] = output_phot_fnu[2]/1e-23/1e-9
  HST_F814W[obj] = output_phot_fnu[3]/1e-23/1e-9
  HST_F850LP[obj] = output_phot_fnu[4]/1e-23/1e-9
  NRC_F070W[obj] = output_phot_fnu[5]/1e-23/1e-9
  NRC_F090W[obj] = output_phot_fnu[6]/1e-23/1e-9
  NRC_F115W[obj] = output_phot_fnu[7]/1e-23/1e-9
  NRC_F150W[obj] = output_phot_fnu[8]/1e-23/1e-9
  NRC_F200W[obj] = output_phot_fnu[9]/1e-23/1e-9
  NRC_F277W[obj] = output_phot_fnu[10]/1e-23/1e-9
  NRC_F335M[obj] = output_phot_fnu[11]/1e-23/1e-9
  NRC_F356W[obj] = output_phot_fnu[12]/1e-23/1e-9
  NRC_F410M[obj] = output_phot_fnu[13]/1e-23/1e-9
  NRC_F444W[obj] = output_phot_fnu[14]/1e-23/1e-9
	
  # You can plot the individual object photometry on its spectra using SENSIBLE FNU UNITS
  #plt.plot((output_sed_wavelengths*(1+galaxy_redshift))/10000, output_sed_fnu/1e-23/1e-9 )
  #plt.scatter((output_phot_wavelengths)/10000, output_phot_fnu/1e-23/1e-9, color = 'red', zorder = 10)
  #plt.xlim([0.0,6.0])
  #plt.ylim([0.0,30])
  #plt.xlabel('Observed Wavelength (Microns)')
  #plt.ylabel('F$_{nu}$ (nJy)')
  #plt.show()

  # Re Major and Sersic Index
  re_maj[obj], sersic_n[obj] = assign_re_maj_sersic()

# Now, let's output the fits file. 
# First, let's make the  dtype and colnames arrays
colnames = np.zeros(25, dtype ='S25')
colnames[0] = 'ID'
colnames[1] = 'redshift'
colnames[2] = 'Re_maj'
colnames[3] = 'sersic_n'
colnames[4] = 'HST_F435W'
colnames[5] = 'HST_F606W'
colnames[6] = 'HST_F775W'
colnames[7] = 'HST_F814W'
colnames[8] = 'HST_F850LP'
colnames[9] = 'NRC_F070W'
colnames[10] = 'NRC_F090W'
colnames[11] = 'NRC_F115W'
colnames[12] = 'NRC_F150W'
colnames[13] = 'NRC_F200W'
colnames[14] = 'NRC_F277W'
colnames[15] = 'NRC_F335M'
colnames[16] = 'NRC_F356W'
colnames[17] = 'NRC_F410M'
colnames[18] = 'NRC_F444W'

colnames[19] = 'redshift_start_constant'
colnames[20] = 'logmass_constant'
colnames[21] = 'redshift_burst1'
colnames[22] = 'logmass_burst1'
colnames[23] = 'redshift_burst2'
colnames[24] = 'logmass_burst2'

# And finally, let's write out the output file.
outtab = Table([galaxy_ids, galaxy_redshift, re_maj, sersic_n,
	HST_F435W, HST_F606W, HST_F775W, HST_F814W, HST_F850LP, NRC_F070W, NRC_F090W, NRC_F115W, 
	NRC_F150W, NRC_F200W, NRC_F277W, NRC_F335M, NRC_F356W, NRC_F410M, NRC_F444W, 
	constant_sf_start_z, constant_mass_formed, burst1_redshift, burst1_mass,
	burst2_redshift, burst2_mass], names=colnames)

exists = os.path.isfile(bagpipes_burst_galaxy_file_name+'.fits')
if exists:
  os.system('rm '+bagpipes_burst_galaxy_file_name+'.fits')

outtab.write(bagpipes_burst_galaxy_file_name+'.fits')

#I'm being horrible and messy and allowing the columns to fall in whatever order they want
outputTable = Table(beagleParamDict)
outputTable.write("BEAGLEinput.fits",overwrite=True)

# This would allow you to plot using the built-in BAGPIPES codes 
#fig = model.plot()
#sfh = model.sfh.plot()
