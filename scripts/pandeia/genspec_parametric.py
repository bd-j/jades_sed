#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Script to generate intrinsic resolution spectra of a large sample
of JAGUAR galaxies given thier properties.  The resulting spectra can be
used as the input to pandeia
"""

import time, sys, glob, os
from os.path import join as pjoin
from copy import deepcopy

import h5py
import numpy as np
from astropy.io import fits

from prospect import prospect_args
from parametric_fsps import uni, build_model, build_sps, build_obs


if __name__ == '__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--fullspec', action="store_true",
                        help="If set, generate the full wavelength array.")
    parser.add_argument('--smoothssp', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--datadir', type=str, default="/Users/bjohnson/Projects/jades_d2s5/data/",
                        help="location of the beagle parameters and S/N curves")
    parser.add_argument('--seed', type=int, default=101,
                        help=("RNG seed for the noise. Negative values result"
                              "in no seed being set."))
    parser.add_argument('--outroot', type=str, default="parametric_fsps")

    args = parser.parse_args()
    run_params = vars(args)

    sname = os.path.join(args.datadir, "noisy_spectra",
                         "evolvingMstarSfr_JAGUARstyle.fits")
    bname = os.path.join(args.datadir, "noisy_spectra", "parametric",
                         "evolvingMstarSfrBEAGLEinput.fits")
    scat = np.array(fits.getdata(sname))
    nobj = len(scat)
    bcat = np.array(fits.getdata(bname))
    assert len(bcat) == nobj

    sps = build_sps(**run_params)
    run_params["sps_libraries"] = [uni(l) for l in sps.ssp.libraries]

    tag = args.outroot + "_{}_{}".format(*run_params["sps_libraries"])
    hfile = tag + ".h5"

    #nobj = 10
    object_ids = np.arange(nobj)
    with h5py.File(hfile, "x") as out:
        for objid in object_ids:
            obj = out.create_group(str(objid))

            # store beagle parameters
            bpars = obj.create_dataset("beagle_parameters", data=bcat[objid])
            # store sizes (and other JAGUAR information)
            spars = obj.create_dataset("jaguar_parameters", data=scat[objid])
            # Try to store the BEAGLE + pandeia spectrum
            specf = os.path.join(args.datadir, "noisy_spectra", "parametric", "DEEP_R100",
                                 "{}_clear_prism_extended.fits".format(objid))
            if os.path.exists(specf):
                bspec = obj.create_group("BEAGLE_DEEP_R100_withSizes")
                bdat = np.array(fits.getdata(specf))
                for c in bdat.dtype.names:
                    d = bspec.create_dataset(c, data=bdat[c])

    run_params["datafile"] = hfile

    with h5py.File(hfile, "a") as out:
        for objid in object_ids:
            obj = out[str(objid)]

            # Generate spectrum
            try:
                obs = build_obs(objid=objid, sps=sps, **run_params)
            except(AssertionError):
                print("parameters for {} out of prior range".format(objid))

            # Store spectrum
            try:
                pspec = obj.create_group("prospector_intrinsic")
            except(NameError):
                del obj["propspector_intrinsic"]
                pspec = obj.create_group("prospector_intrinsic")
            pspec.create_dataset("wavelength", data=obs["wavelength"])
            pspec.create_dataset("spectrum", data=obs["spectrum"])
            pspec.attrs["libraries"] = run_params["sps_libraries"]
            pspec.attrs["object_redshift"] = obs["object_redshift"]
            pspec.attrs["seed"] = args.seed
            pspec.attrs["wave_units"] = "angstroms"
            pspec.attrs["flux_units"] = "maggies"
            for k, v in obs["model_params"].items():
                pspec.attrs[k] = v
