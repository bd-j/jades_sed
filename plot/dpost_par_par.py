#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot P(delta parameter) vs parameter with vilin plots.
"""

import sys, os, glob, time
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
from numpy.lib.recfunctions import append_fields

import h5py
from astropy.io import fits

from prospect.io.read_results import results_from
from plotutils import sample_posterior, chain_to_struct
from sfhplot import delay_tau_ssfr, delay_tau_mwa


pl.rcParams["font.family"] = "serif"
pl.rcParams["font.serif"] = ["STIXGeneral"]
pl.rcParams["mathtext.fontset"] = "custom"
pl.rcParams["mathtext.rm"] = "serif"
pl.rcParams["mathtext.sf"] = "serif"
pl.rcParams['mathtext.it'] = 'serif:italic'


catname = ("/Users/bjohnson/Projects/jades_d2s5/data/"
           "noisy_spectra/parametric_mist_ckc14.h5")


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


def get_truths(results, catname=catname):
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

    parameter = "agem"
    xparam = "mass"
    nsample = 500
    ftype = "parametric_parametric"
    search = "/Users/bjohnson/Projects/jades_d2s5/jobs/output/v2/{}*h5"
    files = glob.glob(search.format(ftype))
    results, obs, models = setup(files)

    names = results[0]["theta_labels"]
    # --- construct samples ----
    samples = [sample_posterior(res["chain"], res["weights"], nsample=nsample)
               for res in results]
    samples = [chain_to_struct(s, m, names=names) for s, m in zip(samples, models)]
    samples = construct_parameters(samples)
    truths = get_truths(results)
    truths = construct_parameters([truths])[0]

    #sys.exit()

    redshifts = truths["zred"]

    #sys.exit()

    zlims = [2, 3, 4, 5, 6, 7]

    cmap = matplotlib.cm.get_cmap('viridis')
    nbins = len(zlims) - 1
    fig, axes = pl.subplots(nbins, 1, sharex="col", figsize=(10.5, 9.5))
    for iz in range(nbins):
        ax = axes[iz]
        zlo, zhi = zlims[iz], zlims[iz + 1]
        choose = ((redshifts > zlo) & (redshifts < zhi))
        dataset = [(np.squeeze(s[parameter]) / truths[parameter][i] - 1.0)
                   for i, s in enumerate(samples) if choose[i]]
        positions = np.log10(truths[xparam][choose])

        vparts = ax.violinplot(dataset, positions, widths=0.15,
                               showmedians=False, showmeans=False,
                               showextrema=False)

        #pmin, pmax = thist["redshift"].min(), thist["redshift"].max()
        pmin, pmax = zlo, zhi
        norm = matplotlib.colors.Normalize(vmin=pmin, vmax=pmax)
        zreds = (redshifts[choose] - pmin) / (pmax - pmin)

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
    [ax.set_ylim(-0.75, 0.75) for ax in axes]
    axes[0].set_title("Mock={}; Model={}; S/N=DEEP with sizes".format(*ftype.split("_")))
    #fig.savefig("figures/delta_{}.png".format(parameter), dpi=600)
    pl.show()
