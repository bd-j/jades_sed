#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, glob, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
from numpy.lib.recfunctions import append_fields

import h5py
from astropy.io import fits
#import seaborn as sns

from prospect.io.read_results import results_from
from plotutils import sample_posterior, chain_to_struct
from plotutils import twodhist, get_cmap
from sfhplot import delay_tau_ssfr


pl.rcParams["font.family"] = "serif"
pl.rcParams["font.serif"] = ["STIXGeneral"]
pl.rcParams["mathtext.fontset"] = "custom"
pl.rcParams["mathtext.rm"] = "serif"
pl.rcParams["mathtext.sf"] = "serif"
pl.rcParams['mathtext.it'] = 'serif:italic'


catname = ("/Users/bjohnson/Projects/jades_d2s5/data/"
           "noisy_spectra/parametric_mist_ckc14.h5")


def get_truths(results, catname=catname):
    jcat = []
    with h5py.File(catname, "r") as catalog:
        for res in results:
            objid = res["run_params"]["objid"]
            jcat.append(catalog[str(objid)]["beagle_parameters"][()])

    jcat = np.hstack(jcat)
    #jcat = convert(jcat)
    return jcat


def setup(files):
    results, observations, models = [], [], []
    for fn in files:
        try:
            res, obs, model = results_from(fn)
        except(OSError, KeyError):
            print("Bad file: {}".format(fn))
            continue
        results.append(res)
        observations.append(obs)
        models.append(model)
    return results, observations, models


if __name__ == "__main__":

    ftype = "parametric_parametric"
    search = "/Users/bjohnson/Projects/jades_d2s5/jobs/output/v2/{}*h5".format(ftype)
    files = glob.glob(search)
    results, obs, models = setup(files)

    # --- construct samples ----
    nsample = 1000
    samples = [sample_posterior(res["chain"], res["weights"], nsample=nsample)
               for res in results]
    samples = [chain_to_struct(s, m) for s, m in zip(samples, models)]
    rectified_samples = []
    # Add SFR to samples
    for s in samples:
        ssfr = delay_tau_ssfr([s["tau"][:, 0], s["tage"][:, 0]])
        sfr = ssfr * s["mass"][:, 0]
        rectified_samples.append(append_fields(s, ["ssfr", "sfr"], [ssfr, sfr]))

    samples = rectified_samples
    truths = get_truths(results)
    redshifts = np.array([t["redshift"] for t in truths])

    #sys.exit()

    par1, par2 = "mass", "sfr"
    zlims = [2, 3, 4, 5, 6, 7, 8]

    color = "royalblue"
    levels = np.array([1.0 - np.exp(-0.5 * 1**2)] )  # 1-sigma
    #levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
    contour_cmap = get_cmap(color, levels)
    nbins = len(zlims) - 1
    fig, axes = pl.subplots(3, 2, sharex="col", sharey="row", 
                            figsize=(10.25, 11.5), squeeze=False)
    for iz in range(nbins):
        ax = axes.flat[iz]
        zlo, zhi = zlims[iz], zlims[iz + 1]
        choose = np.where((truths["redshift"] > zlo) & (truths["redshift"] < zhi))[0]
        for idx in choose:
            p1 = np.squeeze(samples[idx][par1]) 
            #p1 -= 10**truths[idx][par1])/10**truths[idx][par1] 
            p2 = np.squeeze(samples[idx][par2]) 
            #p2 -= 10**truths[idx][par2])/10**truths[idx][par2]
            znorm = (truths[idx]["redshift"] - zlo) / (zhi - zlo)
            X, Y, H, V, clevels, _ = twodhist(p1, p2, levels=levels, smooth=0.05)
            #ax = sns.kdeplot(p1, p2, cmap="Reds", n_levels=3, shade=True,
            #                 shade_lowest=False, ax=ax)
            ax.contourf(X, Y, H, clevels, antialiased=True, colors=contour_cmap)
            ax.contour(X, Y, H, V, colors=color)

        ax.text(0.2, 0.8, r"${:.1f}\,<\,z\,<\,{:.1f}$".format(zlo, zhi), transform=ax.transAxes)

    [ax.set_xscale("log") for ax in axes.flat]
    [ax.set_yscale("log") for ax in axes.flat]

    [ax.set_xlim(2e6, 1e10) for ax in axes.flat]
    [ax.set_ylim(3e-2, 100) for ax in axes.flat]
    [ax.set_ylabel("SFR") for ax in axes[:, 0]]
    [ax.set_xlabel(r"$M_{\rm formed}$") for ax in axes[-1, :]]

    fig.suptitle("Mock={}; Model={}; S/N=DEEP with sizes".format(*ftype.split("_")))
    fig.savefig("figures/mass_sfr_{}.png".format(ftype))

    pl.show()
