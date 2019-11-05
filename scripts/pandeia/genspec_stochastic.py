#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, sys, glob, os
from os.path import join as pjoin
from copy import deepcopy

import h5py
import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo

from prospect import prospect_args
from stochastic_fsps import uni, build_model, build_sps, build_obs


# Observed redshift range
zrange = (6, 8)
zstart_max = 9
# Burst redshift range
zrange_burst = (9, 13)
# Burst mass range
mrange_burst = (7, 9)
# Constant period mass range and redshift start
mrange_const = (7, 9)


def draw_stochastic_bursts(dseed=None, zrange=(5, 8), zrange_burst=(9, 13),
                           mrange_burst=(7, 9), mrange_const=(7, 9),
                           zstart_max=9, nburst_max=3, mrange_tot=(8, 10.),
                           gas_logu=-3, logzsol=-0.2, tauV_eff=0.3, **extras):

    # All the draws
    np.random.seed(dseed)
    nburst_draw = np.random.randint(nburst_max) + 1
    object_redshift = np.random.uniform(*zrange)
    zstart = np.random.uniform(object_redshift, zstart_max)
    zbursts = np.random.uniform(object_redshift, zrange_burst[1], size=(nburst_draw,))
    logm_const = np.random.uniform(*mrange_const)
    logm_burst = np.random.uniform(*mrange_burst, size=(nburst_draw,))
    if mrange_tot is not None:
        logm_tot = np.random.uniform(*mrange_tot)

    # Age of universe at the redshift of observed object
    tuniv = cosmo.age(object_redshift).to("Gyr").value

    # convert drawn values to FSPS/prospector parameters
    good_bursts = zbursts > object_redshift
    age_const = tuniv - cosmo.age(zstart).to("Gyr").value
    age_bursts = tuniv - cosmo.age(zbursts[good_bursts]).to("Gyr").value
    mconst = 10**logm_const
    mburst = 10**logm_burst[good_bursts]

    # Put components in lists
    mass = np.array([mconst] + mburst.tolist())
    tage = np.array([age_const] + age_bursts.tolist())
    ncomp = len(mass)
    nburst = ncomp - 1
    #print(mass, sfr, tage)
    #print(ncomp, nburst, nburst_max, nburst_draw)
    # renormalize masses to follow a mass - sfr relation?
    if mrange_tot is not None:
        mass = mass * 10**logm_tot / mass.sum()
        #sfr = mass[0] / (tage[0] * 1e9)
        #mtot = 10**(np.log10(sfr) + 8)
        #mass[1:] = (mtot - mass[0]) / mass[1:].sum() * mass[1:]
    sfr = mass[0] / (tage[0] * 1e9)

    # Put into a structred array
    dtype = get_dtype(nburst_max + 1)
    row = np.zeros(1, dtype=dtype)
    row["redshift"] = object_redshift
    row["nburst"] = nburst
    row["seed"] = dseed
    row["sfr"] = sfr
    row["mass"][0, :ncomp]  = mass
    row["sfh"][0, :ncomp]   = np.array([1] + nburst * [0])
    row["tage"][0, :ncomp]  = tage
    row["const"][0, :ncomp] = np.array([1] + nburst * [0])
    row["metallicity"] = logzsol
    row["gas_logu"] = gas_logu
    row["tauV_eff"] = tauV_eff
    row["zcomp"][0, 0] = zstart
    row["zcomp"][0, 1:ncomp] = zbursts[good_bursts]

    return row


def get_dtype(max_nburst):

    nmax = max_nburst
    cname = ["redshift", "metallicity", "gas_logu", "tauV_eff", "sfr"]
    mcname = ["mass", "tage", "sfh", "const", "zcomp"]
    exname = ["nburst", "seed"]

    cols = ([(n, np.float) for n in cname] +
            [(n, np.float, (nmax,)) for n in mcname] +
            [(n, np.int) for n in exname]
            )
    dtype = np.dtype(cols)
    return dtype


if __name__ == "__main__":

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
    parser.add_argument('--nobj', type=int, default=250,
                        help=("Number of mock spectra to generate"))
    parser.add_argument('--outroot', type=str, default="stochastic")

    args = parser.parse_args()
    run_params = vars(args)
    nobj = args.nobj
    #params = []
    #for objid in range(nobj):
    #    params.append(draw_stochastic_bursts(dseed=objid, **run_params))

    sps = build_sps(**run_params)
    run_params["sps_libraries"] = [uni(l) for l in sps.ssp.libraries]

    tag = args.outroot + "_{}_{}".format(*run_params["sps_libraries"])
    hfile = tag + ".h5"

    object_ids = np.arange(nobj)
    with h5py.File(hfile, "x") as out:
        for objid in object_ids:
            obj = out.create_group(str(objid))
            # store stochastic parameters
            sdat = draw_stochastic_bursts(dseed=objid, **run_params)
            spars = obj.create_dataset("stochastic_parameters", data=sdat)

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
