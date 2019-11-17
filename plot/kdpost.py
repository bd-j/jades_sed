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

from dpost_par_par import construct_parameters, get_truths

pl.rcParams["font.family"] = "serif"
pl.rcParams["font.serif"] = ["STIXGeneral"]
pl.rcParams["mathtext.fontset"] = "custom"
pl.rcParams["mathtext.rm"] = "serif"
pl.rcParams["mathtext.sf"] = "serif"
pl.rcParams['mathtext.it'] = 'serif:italic'


catname = ("/Users/bjohnson/Projects/jades_d2s5/data/"
           "noisy_spectra/parametric_mist_ckc14.h5")


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

    nsample = 1000
    ftype = "parametric_parametric"
    search = "/Users/bjohnson/Projects/jades_d2s5/jobs/output/v2/{}*h5".format(ftype)
    files = glob.glob(search)
    results, obs, models = setup(files)

    names = results[0]["theta_labels"]
    # --- construct samples ----
    samples = [sample_posterior(res["chain"], res["weights"], nsample=nsample)
               for res in results]
    samples = [chain_to_struct(s, m, names=names) for s, m in zip(samples, models)]
    samples = construct_parameters(samples)
    truths = get_truths(results)
    truths = construct_parameters([truths])[0]
    redshifts = truths["zred"]

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
    tfig, taxes = pl.subplots(3, 2, sharex="col", sharey="row", 
                             figsize=(10.25, 11.5), squeeze=False)
    for iz in range(nbins):
        ax = axes.flat[iz]
        tax = taxes.flat[iz]
        zlo, zhi = zlims[iz], zlims[iz + 1]
        choose = np.where((redshifts > zlo) & (redshifts < zhi))[0]
        tax.plot(truths["mass"][choose], truths["sfr"][choose], 
                 marker="o", linestyle="", markersize=2)
        for idx in choose:
            p1 = np.squeeze(samples[idx][par1])
            #p1 -= 10**truths[idx][par1])/10**truths[idx][par1] 
            p2 = np.squeeze(samples[idx][par2])
            #p2 -= 10**truths[idx][par2])/10**truths[idx][par2]
            znorm = (redshifts[idx] - zlo) / (zhi - zlo)
            X, Y, H, V, clevels, _ = twodhist(p1, p2, levels=levels, smooth=0.05)
            #ax = sns.kdeplot(p1, p2, cmap="Reds", n_levels=3, shade=True,
            #                 shade_lowest=False, ax=ax)
            ax.contourf(X, Y, H, clevels, antialiased=True, colors=contour_cmap)
            ax.contour(X, Y, H, V, colors=color)

        ax.text(0.2, 0.8, r"${:.1f}\,<\,z\,<\,{:.1f}$".format(zlo, zhi), transform=ax.transAxes)
        tax.text(0.2, 0.8, r"${:.1f}\,<\,z\,<\,{:.1f}$".format(zlo, zhi), transform=tax.transAxes)

    allaxes = axes.flatten().tolist() + taxes.flatten().tolist()

    [ax.set_xscale("log") for ax in allaxes]
    [ax.set_yscale("log") for ax in allaxes]

    [ax.set_xlim(2e6, 1e10) for ax in allaxes]
    [ax.set_ylim(3e-2, 100) for ax in allaxes]
    [ax.set_ylabel("SFR") for ax in axes[:, 0]]
    [ax.set_xlabel(r"$M_{\rm formed}$") for ax in axes[-1, :]]
    [ax.set_ylabel("SFR") for ax in taxes[:, 0]]
    [ax.set_xlabel(r"$M_{\rm formed}$") for ax in taxes[-1, :]]

    fig.suptitle("Mock={}; Model={}; S/N=DEEP with sizes".format(*ftype.split("_")))
    tfig.suptitle("Mock={}; Model={}; S/N=DEEP with sizes".format(*ftype.split("_")))
    fig.savefig("figures/mass_sfr_{}.png".format(ftype), dpi=400)
    tfig.savefig("figures/mass_sfr_truth_{}.png".format(ftype), dpi=400)

    pl.show()
