from astropy.io import fits
import numpy as np
from pandeia.engine.calc_utils import build_default_calc
from pandeia.engine.perform_calculation import perform_calculation
import pandeia.engine
from astropy.table import Table
from collections import OrderedDict
import copy
import pylab
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
                    '-b', '--beagle-output-file',
                    help="the beagle output file containing simulated spectra",
                    action="store",
                    type=str,
                    dest="beagleOutputFile",
                    required=True
                    )

parser.add_argument(
                    '-s', '--sizes-file',
                    help="file containing size info.  Must have same number of rows as BEAGLE output file.  If not supplied, objects will be simulated as point source.",
                    action="store",
                    type=str,
                    dest="sizesFile",
                    required=False,
                    default=None
                    )

parser.add_argument(
                    '-o', '--output-folder',
                    help="output folder where fits files containing noisy spectra are written",
                    action="store",
                    type=str,
                    dest="outputFolder",
                    required=False,
                    default=None
                    )



parser.add_argument(
                    '-n',
                    help="number of objects in file to make spectra of",
                    action="store",
                    type=str,
                    dest="n",
                    required=False,
                    default=None
                    )

args = parser.parse_args()

#This script was adapted from a short script written by Michael Maseda demonstrating how to
#set up emission line S/N calculations in pandeia.


def bn(n):
    return 2.*n-1./3.+4./(405.*n)+46./(25515.*n**2.)+131./(1148175.*n**3.)-2194697./(30690717750.*n**4.)
def majorminor(n,re,ellip):
    #ellipticity is defined as (major-minor)/major
    scale_length=re/(bn(n)**n)
    #scale length is the circularized radius, i.e. r_scale = sqrt(a*b)
    major_axis=scale_length/np.sqrt(1.-ellip)
    minor_axis=scale_length*np.sqrt(1.-ellip)
    return (major_axis,minor_axis)

def sn_user_spec(inputs, disperser = 'prism', filt = 'clear', ngroup = 19, nint = 2, nexp = 36):
    wl = inputs['wl'] #in microns
    spec = inputs['spec'] #in mJy
    xoff = inputs['xoff']
    yoff = inputs['yoff']
    
    configuration=build_default_calc('jwst','nirspec','msa')
    configuration['configuration']['instrument']['disperser']=disperser
    configuration['configuration']['instrument']['filter']=filt
    #E - I *think* shutter location is just so that the detector gap is placed in the correct place
    configuration['configuration']['instrument']['shutter_location']='q3_345_20'#'q4_345_20'
    #exposure specifications from DEEP APT file
    configuration['configuration']['detector']['ngroup']=ngroup
    configuration['configuration']['detector']['nint']=nint
    #PRISM
    configuration['configuration']['detector']['nexp']=nexp
    configuration['configuration']['detector']['readmode']='nrsirs2'
    configuration['strategy']['dithers'][0]['on_source'] = inputs['onSource']
    configuration['configuration']['instrument']['slitlet_shape'] = inputs['slitletShape']
    
    #default configuration['configuration'] has 1x3 shutter config
    scene = {}
    if (inputs['re_circ'] > 0):
        sersic=inputs['sersic_n']
        rc = inputs['re_circ']
        ellip=1.-inputs['axis_ratio']
        pa=inputs['position_angle']
        #pandiea wants scale lengths not half-light radii
        major_axis,minor_axis=majorminor(sersic,rc,ellip)
        scene['position'] = {'x_offset':xoff, 'y_offset': yoff, 'orientation': pa, 'position_parameters':['x_offset','y_offset','orientation']}
        scene['shape']={'geometry':'sersic','major': major_axis,'minor':minor_axis,'sersic_index':sersic}
    else:
        pa=inputs['position_angle']
        #this is the dummy trigger to go for a point source
        scene['position'] = {'x_offset':xoff, 'y_offset': yoff, 'orientation': pa, 'position_parameters':['x_offset','y_offset','orientation']}
        scene['shape']={'geometry':'point'}
    
    scene['spectrum'] = {}
    scene['spectrum']['name'] = "continuum_spectrum"
    scene['spectrum']['redshift'] = 0 #because otherwise it shifts the wavelength array...
    #it doesn't seem to do anything with the normalization of the
    #source spectrum, however?!
    tempIdx = np.where((wl >= filterWlDict[filt]['low']) & (wl <= filterWlDict[filt]['high']))[0]
    scene['spectrum']['sed'] = {'sed_type': 'input', 'spectrum': [wl[tempIdx], spec[tempIdx]], 'unit':'mJy'}
    scene['spectrum']['normalization'] = {}
    scene['spectrum']['normalization'] = {'type': 'none'}
    
    configuration['scene'][0]=scene
    report=perform_calculation(configuration)
    return report


#exposureDict = {'DEEP':{'clear':{'ngroup':19,'nint':2,'nexp':36},'f070lp':{'ngroup':19,'nint':2,'nexp':9},'f100lp':{'ngroup':19,'nint':2,'nexp':9},'f170lp':{'ngroup':19,'nint':2,'nexp':9},'f290lp':{'ngroup':19,'nint':2,'nexp':9}},\
#                'MEDIUM':{'clear':{'ngroup':13,'nint':1,'nexp':9},'f070lp':{'ngroup':13,'nint':1,'nexp':9},'f100lp':{'ngroup':13,'nint':1,'nexp':9},'f170lp':{'ngroup':13,'nint':1,'nexp':9},'f290lp':{'ngroup':13,'nint':1,'nexp':9}},\
#                'MEDIUM_HST':{'clear':{'ngroup':16,'nint':1,'nexp':6},'f070lp':{'ngroup':13,'nint':1,'nexp':6},'f100lp':{'ngroup':13,'nint':1,'nexp':6},'f170lp':{'ngroup':13,'nint':1,'nexp':6},'f290lp':{'ngroup':16,'nint':1,'nexp':6}},\
#                'DEEP_WORST_CASE':{'clear':{'ngroup':19,'nint':2,'nexp':12}},\
#                'MEDIUM_WORST_CASE':{'clear':{'ngroup':13,'nint':1,'nexp':3}},\
#                'MEDIUM_HST_WORST_CASE':{'clear':{'ngroup':16,'nint':1,'nexp':6}}}
#exposureDict = {'DEEP':{'clear':{'ngroup':19,'nint':2,'nexp':36}},\
#                'MEDIUM':{'clear':{'ngroup':13,'nint':1,'nexp':9}},\
#                'MEDIUM_HST':{'clear':{'ngroup':16,'nint':1,'nexp':6}},\
#                'DEEP_WORST_CASE':{'clear':{'ngroup':19,'nint':2,'nexp':12}},\
#                'MEDIUM_WORST_CASE':{'clear':{'ngroup':13,'nint':1,'nexp':3}},\
#                'MEDIUM_HST_WORST_CASE':{'clear':{'ngroup':16,'nint':1,'nexp':2}}}
exposureDict = {'DEEP':{'clear':{'ngroup':19,'nint':2,'nexp':36}}}

cat = fits.open(args.beagleOutputFile)
sizesSupplied = False
if args.sizesFile is not None:
  input = fits.open(args.sizesFile)
  sizesSupplied = True

grating = ['g140m','g140m','g235m','g395m']
filter = ['f070lp','f100lp','f170lp','f290lp']
filterDict = {'clear':'prism',\
                'f070lp':'g140m',\
                'f100lp':'g140m',\
                'f170lp':'g235m',\
                'f290lp':'g395m'}

filterWlDict = {'clear':{'low':0.7,'high':5.1},\
                'f070lp':{'low':0.7,'high':1.2},\
                'f100lp':{'low':1.1,'high':1.8},\
                'f170lp':{'low':1.7,'high':3.1},\
                'f290lp':{'low':3.0,'high':5.1}}

wl = cat['FULL SED WL'].data[0][0]
specArr = cat['FULL SED'].data
c_light = 2.99792e+18 # Ang/s
nObj = len(cat['GALAXY PROPERTIES'].data['redshift'])
if args.n is not None:
    if args.n < nObj:
        nObj = args.n+1
for i in range(nObj):
    z = cat['GALAXY PROPERTIES'].data['redshift'][i]
    tempWl = wl*(1+z)
    tempSpec = specArr[i,:]/(1+z)
    tempSpec = tempWl**2/c_light*tempSpec*1E23*1E3 #in mJy
    tempWl = tempWl/1E4 #in microns
    deduped = {w: s for w, s in reversed(zip(list(tempWl), list(tempSpec)))}
    tempWl = np.asarray(sorted(deduped.keys()))
    tempSpec = np.asarray([deduped[k] for k in tempWl])
    inputs = {}
    inputs['wl'] = tempWl
    inputs['spec'] = tempSpec
    inputs['xoff'] = 0.
    inputs['yoff'] = 0.
    if sizesSupplied:
      inputs['axis_ratio'] = input[1].data['axis_ratio'][i]
      inputs['sersic_n'] = input[1].data['sersic_n'][i]
      inputs['position_angle'] = input[1].data['position_angle'][i]
      inputs['re_circ'] = input[1].data['Re_maj'][i]*np.sqrt(input[1].data['axis_ratio'][i])
    else:
      inputs['axis_ratio'] = 1.
      inputs['sersic_n'] = -99
      inputs['position_angle'] = 0.
      inputs['re_circ'] = -99
    inputs['onSource'] = [False,True,False]
    inputs['slitletShape'] = [[0,-1],[0,0],[0,1]]
                                                    
    #produce a mock spectrum for extended and point source, for each of the filter/grating configurations
    for key in exposureDict.keys():
        for filt in exposureDict[key].keys():
            print filt
            report = sn_user_spec(inputs, disperser = filterDict[filt], filt = filt, ngroup = exposureDict[key][filt]['ngroup'], nint = exposureDict[key][filt]['nint'], nexp = exposureDict[key][filt]['nexp'])
            outputDict = OrderedDict()
            outputDict['wl'] = report['1d']['extracted_flux'][0]
            noise = report['1d']['target'][1]/report['1d']['sn'][1]
            noisy_spectrum = report['1d']['target'][1]+np.random.normal(size=len(report['1d']['target'][1]))*noise
            outputDict['fnu'] = noisy_spectrum
            outputDict['fnu_err'] = noise
            outputDict['sn'] = report['1d']['sn'][1]
            outputDict['fnu_noiseless'] = report['1d']['target'][1]
            tempIdx = np.where((report['1d']['sn'][0] >= 0.6450*(1+z)) & (report['1d']['sn'][0] <= 0.6650*(1+z)))[0]
            print tempIdx
            if len(tempIdx) == 0: #use region around OII 3727
              print 0.6450*(1+z), 0.3727*(1+z)
              tempIdx = np.where((report['1d']['sn'][0] >= 0.3600*(1+z)) & (report['1d']['sn'][0] <= 0.3800*(1+z)))[0]
              print tempIdx
            
            sn = np.max(report['1d']['sn'][1][tempIdx])
            if sn > 3:
                if filt == 'clear':
                    folder = args.outputFolder+key+"_R100/"
                else:
                    folder = args.outputFolder+key+"_R1000/"
                idStr = str(i)
                if sizesSupplied:
                  if 'ID' in input[1].data.dtype.names:
                    idStr = str(input[1].data['ID'][i])
                outputFile = folder+'/'+idStr+'_'+filt+'_'+filterDict[filt]+'_extended.fits'
                outputTable = Table(outputDict)
                outputTable.write(outputFile,overwrite=True)
#            inputs2 = copy.deepcopy(inputs)
#            inputs2['re_maj'],inputs2['re_min'] = -1,-1
#            report = sn_user_spec(inputs2, disperser = filterDict[filt], filt = filt, ngroup = exposureDict[key][filt]['ngroup'], nint = exposureDict[key][filt]['nint'], nexp = exposureDict[key][filt]['nexp'])
#            outputDict = OrderedDict()
#            outputDict['wl'] = report['1d']['extracted_flux'][0]
#            outputDict['extracted_flux'] = report['1d']['extracted_flux'][1]
#            outputDict['extracted_noise'] = report['1d']['extracted_noise'][1]
#            outputDict['sn'] = report['1d']['sn'][1]
#            if filt == 'clear':
#                folder = "/Users/user/Documents/JWST/data-to-science/"+key+"_R100/"
#            else:
#                folder = "/Users/user/Documents/JWST/data-to-science/"+key+"_R1000/"
#            outputFile = folder+str(input[1].data['ID'][i])+'_'+filt+'_'+filterDict[filt]+'_ps.fits'
#            outputTable = Table(outputDict)
#            outputTable.write(outputFile,overwrite=True)
