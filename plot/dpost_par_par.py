#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot P(delta parameter) vs parameter with vilin plots.
"""

import sys, os, glob, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
from numpy.lib.recfunctions import append_fields

import h5py
from astropy.io import fits

from prospect.io.read_results import results_from
from plotutils import sample_posterior, chain_to_struct
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
        res, obs, model = results_from(fn)
        results.append(res)
        observations.append(obs)
        models.append(model)
    return results, observations, models


if __name__ == "__main__":
    
    parameter = "mass"
    xparam = "mass"
    nsample = 500
    ftype = "parametric_parametric"
    search = "/Users/bjohnson/Projects/jades_d2s5/jobs/output/v2/{}*h5"
    files = glob.glob(search.format(ftype))
    results, obs, models = setup(files)

    # --- construct samples ----
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

    zlims = [2, 3, 4, 5, 6, 7]

    cmap = matplotlib.cm.get_cmap('viridis')
    nbins = len(zlims) - 1
    fig, axes = pl.subplots(nbins, 1, sharex="col", figsize=(10.5, 9.5))
    for iz in range(nbins):
        ax = axes[iz]
        zlo, zhi = zlims[iz], zlims[iz+1]
        choose = ((truths["redshift"] > zlo) & (truths["redshift"] < zhi))
        dataset = [(np.squeeze(s[parameter]) - 10**t[parameter])/10**t[parameter] 
                   for s, t, c in zip(samples, truths, choose)
                   if c]
        thist = truths[choose]
        positions = thist[xparam]

        vparts = ax.violinplot(dataset, positions, widths=0.15,
                               showmedians=False, showmeans=False, 
                               showextrema=False)

        #pmin, pmax = thist["redshift"].min(), thist["redshift"].max()
        pmin, pmax = zlo, zhi
        norm = matplotlib.colors.Normalize(vmin=pmin, vmax=pmax)
        zreds = (thist["redshift"] - pmin) / (pmax - pmin)

        for z, pc in zip(zreds, vparts['bodies']):
            pc.set_facecolor(cmap(z))
            pc.set_edgecolor(cmap(z))
            pc.set_alpha(0.5)

        # colorbar and prettify
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label("redshift")
        ax.axhline(0, linestyle=":", color="k")
        ax.set_ylabel(r"${}\, /\, {}_{{input}}-1$".format(parameter, parameter))

    ax.set_xlabel(r"$\log \, ({}_{{\rm input}})$".format(xparam), fontsize=14)
    [ax.set_ylim(-0.5, 0.5) for ax in axes]
    axes[0].set_title("Mock={}; Model={}; S/N=DEEP with sizes".format(*ftype.split("_")))
    #fig.savefig("figures/delta_{}.png".format(parameter), dpi=600)
    pl.show()
