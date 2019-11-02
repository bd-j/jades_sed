#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, glob, time
import numpy as np
import matplotlib.pyplot as pl

import h5py
from astropy.io import fits

pl.rcParams["font.family"] = "serif"
pl.rcParams["font.serif"] = ["STIXGeneral"]
pl.rcParams["mathtext.fontset"] = "custom"
pl.rcParams["mathtext.rm"] = "serif"
pl.rcParams["mathtext.sf"] = "serif"
pl.rcParams['mathtext.it'] = 'serif:italic'

catname = ("/Users/bjohnson/Projects/jades_d2s5/data/"
           "noisy_spectra/pstochastic_mist_ckc14.h5")


def get_snr():
    pass

if __name__ == "__main__":
    
    # extract snr and parameters
    snr, jcat = [], []
    
    sgroup = "DEEP_R100" #"_withSizes"
    with h5py.File(catname, "r") as catalog:
        objlist = list(catalog.keys())
        for strobj in objlist:
            try:
                etc = catalog[strobj][sgroup]
            except:
                continue
            snr.append(etc["sn"][:])
            #jcat.append(catalog[strobj]["jaguar_parameters"][()])
            jcat.append(catalog[strobj]["stochastic_parameters"][()])
    jcat = np.hstack(jcat)
    snr = np.array(snr)
    
    median_snr = np.nanmedian(snr, axis=-1)
    #median_snr[median_snr< 0] = 1e-2
    peak_snr = np.nanmax(snr, axis=-1)
    sel = (jcat["redshift"] > 4) & (jcat["redshift"] < 5)

    from scipy.stats import binned_statistic_2d
    psn_map, xp, yp, bp = binned_statistic_2d(jcat["mStar"][sel], jcat["SFR_10"][sel], peak_snr[sel], 
                                           statistic="median", bins=30)
    msn_map, xm, ym, bm = binned_statistic_2d(jcat["mStar"][sel], jcat["SFR_10"][sel], median_snr[sel], 
                                           statistic="median", bins=30)

    fig, axes = pl.subplots(1, 2, sharey=True)
    ax = axes[0]
    c = ax.pcolor(xm, ym, np.log10(msn_map.T), cmap="viridis", vmin=0, vmax=3) 
    cbar = fig.colorbar(c, ax=ax, label="(median of) log Median S/N")
    ax.set_xlabel(r"$\log M_*$")
    ax.set_ylabel(r"$\log SFR$")
    ax.set_title("Mock=Parametric; S/N=DEEP with sizes")
    
    ax = axes[1]
    c = ax.pcolor(xp, yp, np.log10(psn_map.T), cmap="viridis", vmin=0, vmax=3) 
    cbar = fig.colorbar(c, ax=ax, label="(median of) log Peak S/N")
    ax.set_xlabel(r"$\log M_*$")
    ax.set_title("Mock=Parametric; S/N=DEEP with sizes")
    
    pl.show()
    
    sys.exit()
    