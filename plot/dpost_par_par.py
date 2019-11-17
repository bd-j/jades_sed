#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot P(delta parameter) vs parameter with vilin plots.
"""

import sys, os, glob, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib

from transforms import construct_parameters, get_truths
from plotutils import sample_posterior, chain_to_struct, setup


pl.rcParams["font.family"] = "serif"
pl.rcParams["font.serif"] = ["STIXGeneral"]
pl.rcParams["mathtext.fontset"] = "custom"
pl.rcParams["mathtext.rm"] = "serif"
pl.rcParams["mathtext.sf"] = "serif"
pl.rcParams['mathtext.it'] = 'serif:italic'


catname = ("/Users/bjohnson/Projects/jades_d2s5/data/"
           "noisy_spectra/parametric_mist_ckc14.h5")


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
