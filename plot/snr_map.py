#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, glob, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
from scipy.stats import binned_statistic_2d

import h5py
from astropy.io import fits

pl.rcParams["font.family"] = "serif"
pl.rcParams["font.serif"] = ["STIXGeneral"]
pl.rcParams["mathtext.fontset"] = "custom"
pl.rcParams["mathtext.rm"] = "serif"
pl.rcParams["mathtext.sf"] = "serif"
pl.rcParams['mathtext.it'] = 'serif:italic'

mock, sgroup = "parametric", "DEEP_R100_withSizes"

catname = ("/Users/bjohnson/Projects/jades_d2s5/data/"
           "noisy_spectra/{}_mist_ckc14_rhalf.h5")
catname = catname.format(mock)


def get_snr():
    pass


if __name__ == "__main__":

    zlims = [2, 3, 4, 5, 6, 7]

    # extract snr and parameters
    snr, jcat = [], []

    with h5py.File(catname, "r") as catalog:
        objlist = list(catalog.keys())
        for strobj in objlist:
            try:
                etc = catalog[strobj][sgroup]
            except:
                continue
            snr.append(etc["sn"][:])
            try:
                jcat.append(catalog[strobj]["jaguar_parameters"][()])
            except:
                jcat.append(catalog[strobj]["stochastic_parameters"][()])
    jcat = np.hstack(jcat)
    snr = np.array(snr)

    median_snr = np.nanmedian(snr, axis=-1)
    #median_snr[median_snr< 0] = 1e-2
    peak_snr = np.nanmax(snr, axis=-1)

    cmap = matplotlib.cm.get_cmap('magma')
    nbins = len(zlims) - 1
    fig, axes = pl.subplots(nbins, 2, sharex="col", figsize=(11.5, 11.5))
    for iz in range(nbins):
        zlo, zhi = zlims[iz], zlims[iz + 1]
        sel = (jcat["redshift"] > zlo) & (jcat["redshift"] < zhi)
        if len(sel) < 5:
            continue
        zlabel = "{} $< z <$ {}".format(zlo, zhi)
        psn_map, xp, yp, bp = binned_statistic_2d(jcat["mStar"][sel], jcat["SFR_10"][sel], peak_snr[sel], 
                                           statistic="median", bins=30)
        msn_map, xm, ym, bm = binned_statistic_2d(jcat["mStar"][sel], jcat["SFR_10"][sel], median_snr[sel], 
                                           statistic="median", bins=30)

        ax = axes[iz, 0]
        c = ax.pcolor(xm, ym, np.log10(msn_map.T), cmap=cmap, vmin=0, vmax=3) 
        cbar = fig.colorbar(c, ax=ax, label="(median of) log Median S/N")
        ax.set_xlabel(r"$\log M_*$")
        ax.set_ylabel(r"$\log SFR$")
        ax.text(0.1, 0.8, zlabel, transform=ax.transAxes)

        ax = axes[iz, 1]
        c = ax.pcolor(xp, yp, np.log10(psn_map.T), cmap=cmap, vmin=0, vmax=3) 
        cbar = fig.colorbar(c, ax=ax, label="(median of) log Peak S/N")
        ax.set_xlabel(r"$\log M_*$")
        ax.text(0.1, 0.8, zlabel, transform=ax.transAxes)

    axes[0, 0].set_title("Mock={}; {}".format(mock, sgroup))
    axes[0, 1].set_title("Mock={}; S/N={}".format(mock, sgroup))

    fig.savefig("figures/snr_map.pdf", dpi=600)
    pl.show()
    sys.exit()