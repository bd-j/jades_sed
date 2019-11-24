#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as pl

from numpy.lib.recfunctions import append_fields
import h5py
from astropy.io import fits

from sfhplot import delay_tau_ssfr, delay_tau_mwa
from sfhplot import ratios_to_sfrs, nonpar_mwa


def construct_parameters(parsets):
    rectified_samples = []
    # Add SFR to samples
    for s in parsets:
        ssfr = delay_tau_ssfr([s["tau"], s["tage"]])
        sfr = ssfr * s["mass"]
        mwa = delay_tau_mwa([s["tau"], s["tage"]])
        cols = ["ssfr", "sfr", "agem"]
        vals = [ssfr, sfr, mwa]
        if (type(s) is dict):
            print("updating dict")
            s.update({c: v for c, v in zip(cols, vals)})
            rectified_samples.append(deepcopy(s))
        elif (type(s) is np.ndarray):
            rectified_samples.append(append_fields(s, cols, vals))

    return rectified_samples


def construct_stoch_parameters(parsets):

    rectified_samples = []
    # Add SFR to samples
    for s in parsets:
        zred = s["redshift"]
        sfr = s["mass"][:, 0] / s["tage"][:, 0] * 1e9
        mtot = s["mass"].sum(axis=-1)
        ssfr = sfr / mtot
        mwa = s["mass"][:, 0] * s["tage"][:, 0] / 2 + (s["mass"][:, 1:] * s["tage"][:, 1:]).sum(axis=-1)
        mwa /= mtot
        cols = ["ssfr", "sfr1", "totmass", "agem", "zred"]
        vals = [ssfr, sfr, mtot, mwa, zred]
        if (type(s) is dict):
            print("updating dict")
            s.update({c: v for c, v in zip(cols, vals)})
            rectified_samples.append(deepcopy(s))
        elif (type(s) is np.ndarray):
            rectified_samples.append(append_fields(s, cols, vals))

    return rectified_samples


def construct_nonpar_parameters(parsets, agebins=[[]]):
    rectified_samples = []
    # Add SFR to samples
    for s in parsets:
        #zred = s["redshift"]
        logmass = s["logmass"]
        logsfr_ratios = s["logsfr_ratios"]
        sfhs = np.array([ratios_to_sfrs(logm, sr, agebins)
                        for logm, sr in zip(logmass, logsfr_ratios)])
        mwa = nonpar_mwa(logmass, logsfr_ratios, agebins)
        mtot = 10**logmass
        sfr = sfhs[:, 0]
        ssfr = sfr / mtot

        cols = ["ssfr", "sfr1", "totmass", "agem"]
        vals = [ssfr, sfr, mtot, mwa]
        if (type(s) is dict):
            print("updating dict")
            s.update({c: v for c, v in zip(cols, vals)})
            rectified_samples.append(deepcopy(s))
        elif (type(s) is np.ndarray):
            rectified_samples.append(append_fields(s, cols, vals))

    return rectified_samples



def get_stoch_truths(results, catname=""):
    jcat = []
    with h5py.File(catname, "r") as catalog:
        for res in results:
            objid = res["run_params"]["objid"]
            jcat.append(catalog[str(objid)]["stochastic_parameters"][()])

    fcat = np.hstack(jcat)
    #for k, v in fcat.items():
        #try:
        #    fcat[k] = v[:, None]
        #except(TypeError):
        #    continue
    #jcat = convert(jcat)
    return fcat


def beagle_to_fsps(beagle):
    fpars = {}
    # Basic
    fpars["mass"] = 10**beagle["mass"]
    fpars["zred"] = beagle["redshift"]
    fpars["sfr"] = 10**beagle["sfr"]
    # SFH
    fpars["tage"] = 10**(beagle["max_stellar_age"] - 9)
    fpars["tau"] = 10**(beagle["tau"] - 9)
    fpars["logzsol"] = beagle["metallicity"]
    # Dust
    mu, tveff = 0.4, beagle["tauV_eff"]
    fpars["dust2"] = mu * tveff
    fpars["dust_ratio"] = 1.5
    # Neb
    fpars["gas_logu"] = beagle["nebular_logU"]
    fpars["gas_logz"] = beagle["metallicity"]

    return fpars


def get_truths(results, catname=""):
    jcat = []
    with h5py.File(catname, "r") as catalog:
        for res in results:
            objid = res["run_params"]["objid"]
            jcat.append(catalog[str(objid)]["beagle_parameters"][()])

    jcat = np.hstack(jcat)
    fcat = beagle_to_fsps(jcat)
    #for k, v in fcat.items():
        #try:
        #    fcat[k] = v[:, None]
        #except(TypeError):
        #    continue
    #jcat = convert(jcat)
    return fcat
