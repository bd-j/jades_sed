import os
import numpy as np
from astropy.io import fits
import h5py

from astropy.table import Table
from collections import OrderedDict
import argparse

from pandeia_utils import bn, majorminor, sn_user_spec

c_light = 2.99792e18  # Ang/s

#This script was adapted from a short script written by Michael Maseda
#demonstrating how to set up emission line S/N calculations in pandeia.

# Input spectrum file has format
#[FULL SED WL] (nw,)     in Angstroms
#[FULL SED] (nobj, nw)   in erg/s/cm^2/A
#[GALAXY PROPERTIES] (nobj,), structured with
#   "redshift",

exposureDict = {'DEEP':{'clear':{'ngroup':19,'nint':2,'nexp':36}}}
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

Gratings = ['g140m', 'g140m', 'g235m', 'g395m']
Filters = ['f070lp', 'f100lp', 'f170lp', 'f290lp']
filterDict = {'clear': 'prism',
              'f070lp': 'g140m',
              'f100lp': 'g140m',
              'f170lp': 'g235m',
              'f290lp': 'g395m'}


def get_beagle_input(iobj, args):
    """Given a set of arguments and an object index, return the input spectrum

    Returns
    --------

    z : float
        The redshift

    wave : ndarray
           The observed frame wavelength, in micron

    spec : ndarray, same shape as wave
           The observed frame fluxes in mJy

    sizes : A structured array
    """
    cat = fits.open(args.spectrum_file)
    sizes, sizes_supplied = None, False
    if args.sizes_file is not None:
        sizes = fits.open(args.sizes_file)[1].data
        sizes_supplied = True

    wl = cat['FULL SED WL'].data[0][0]
    spec = cat['FULL SED'].data[iobj, :]
    z = cat['GALAXY PROPERTIES'].data['redshift'][iobj]

    tempWl = wl * (1 + z)
    tempSpec = spec / (1 + z)
    # convert from erg/s/cm^2/AA to mJy
    tempSpec *= tempWl**2 / c_light * 1e23 * 1e3
    tempWl /= 1e4  #  AA to micron
    deduped = {w: s for w, s in reversed(zip(list(tempWl), list(tempSpec)))}
    tempWl = np.asarray(sorted(deduped.keys()))
    tempSpec = np.asarray([deduped[k] for k in tempWl])

    return z, tempWl, tempSpec, sizes


def get_pro_input(iobj, args):
    """Given a set of arguments and an object index, return the input spectrum

    Returns
    --------

    z : float
        The redshift

    wave : ndarray
           The observed frame wavelength, in micron

    spec : ndarray, same shape as wave
           The observed frame fluxes in mJy

    sizes : A structured array
    """
    with h5py.File(args.spectrum_file, "r") as f:
        cat = f[str(iobj)]["prospector_intrinsic"]
        wl = cat["wavelength"][:]
        spec = cat["spectrum"][:]
        z = cat.attrs["object_redshift"]

    sizes = None
    if args.sizes_file != "":
        sizes = fits.open(args.sizes_file)[1].data

    # convert from maggies to mJy
    tempSpec = spec * 3631 * 1e3
    # AA to micron
    tempWl = wl / 1e4

    return z, tempWl, tempSpec, sizes


def build_input(iobj, args):

    # get redshifted intrinsic spectrum
    try:
        z, tempWl, tempSpec, sizes = get_beagle_input(iobj, args)
    except:
        z, tempWl, tempSpec, sizes = get_pro_input(iobj, args)

    inputs = {"wl": tempWl, "spec": tempSpec,
              "xoff": 0, "yoff": 0,
              "axis_ratio": 1, "sersic_n": -99,
              "position_angle": 0, "re_circ": -99,
              "onSource": [False, True, False],
              "slitletShape": [[0, -1], [0, 0], [0, 1]],
              "ID": iobj}
    if sizes is not None:
        inputs['axis_ratio'] = sizes['axis_ratio'][iobj]
        inputs['sersic_n'] = sizes['sersic_n'][iobj]
        inputs['position_angle'] = sizes['position_angle'][iobj]
        inputs['re_circ'] = (sizes['Re_maj'][iobj] *
                             np.sqrt(sizes['axis_ratio'][iobj])
                             )
        if 'ID' in sizes.dtype.names:
            inputs["ID"] = sizes["ID"][iobj]

    return inputs, z


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrum_file", type=str, default="")
    parser.add_argument("--sizes_file", type=str, default="")
    parser.add_argument("--hdf5_catalog", type=str, default="")
    parser.add_argument("--nobj", type=int, default=-99)
    parser.add_argument("--output_folder", default=".")
    args = parser.parse_args()

    assert os.path.exists(args.output_folder)
    hfile = args.hdf5_catalog

    for iobj in range(args.nobj):

        inputs, z = build_input(iobj, args)
        # Produce a mock spectrum for extended and point source,
        # for each of the filter/grating configurations
        for exp in exposureDict.keys():
            for filt in exposureDict[exp].keys():
                print(filt)
                report = sn_user_spec(inputs, disperser=filterDict[filt],
                                      filt=filt,
                                      ngroup=exposureDict[exp][filt]['ngroup'],
                                      nint=exposureDict[exp][filt]['nint'],
                                      nexp=exposureDict[exp][filt]['nexp'])
                snr = report['1d']['sn'][1]
                spec = report['1d']['target'][1]
                unc = spec / snr
                noise = np.random.normal(0, np.abs(unc))
                noisy_spectrum = spec + noise

                outputDict = OrderedDict()
                outputDict['wl'] = report['1d']['extracted_flux'][0]
                outputDict['fnu'] = noisy_spectrum
                outputDict['fnu_err'] = unc
                outputDict['sn'] = snr
                outputDict['fnu_noiseless'] = spec

                # Checking for S/N around Halpha or OIII
                snwl = report['1d']['sn'][0]
                tempIdx = np.where((snwl >= 0.6450 * (1 + z)) &
                                   (snwl <= 0.6650 * (1 + z)))[0]
                #print(tempIdx)
                if len(tempIdx) == 0:  # use region around OII 3727
                    print(0.6450 * (1 + z), 0.3727 * (1 + z))
                    tempIdx = np.where((snwl >= 0.3600 * (1 + z)) &
                                       (snwl <= 0.3800 * (1 + z)))[0]
                    print(tempIdx)

                # Write spectrum if S/N > 3
                sn = np.max(report['1d']['sn'][1][tempIdx])
                print(iobj, sn)

                if filt == "clear":
                    tag = "{}_{}".format(exp, "R100")
                else:
                    tag = "{}_{}".format(exp, "R1000")

                if args.sizes_file != "":
                    tag += "_withSizes"

                # Write to h5py
                if hfile != "":
                    with h5py.open(hfile, "rw") as hcat:
                        group = hcat[str(iobj)].create_group(tag)
                        group.attrs["snr_line"] = sn
                        for k, v in outputDict.items():
                            d = group.create_dataset(k, data=v)

                if sn < 3:
                    pass
                folder = os.path.join(args.output_folder, tag)
                os.makedirs(folder, exist_ok=True)
                #idStr = "{:04.0f}".format(int(inputs["ID"]))
                idStr = "{:.0f}".format(int(inputs["ID"]))
                outName = idStr+'_'+filt+'_'+filterDict[filt]+'_extended.fits'
                outputFile = os.path.join(folder, outName)
                print(outputFile)
                outputTable = Table(outputDict)
                outputTable.write(outputFile, overwrite=True)
