#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot P(delta parameter) vs parameter with vilin plots.
"""

import sys, os, glob, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from prospect.io.read_results import results_from
from plotutils import sample_posterior, chain_to_struct, setup
from transforms import construct_parameters, construct_stoch_parameters
from transforms import get_stoch_truths


pl.rcParams["font.family"] = "serif"
pl.rcParams["font.serif"] = ["STIXGeneral"]
pl.rcParams["mathtext.fontset"] = "custom"
pl.rcParams["mathtext.rm"] = "serif"
pl.rcParams["mathtext.sf"] = "serif"
pl.rcParams['mathtext.it'] = 'serif:italic'


catname = ("/Users/bjohnson/Projects/jades_d2s5/data/"
           "noisy_spectra/stochastic_mist_ckc14.h5")


def delta_plot(parameter="mass", tparameter="totmass",
               xparam="totmass", nsample=500):

    ftype = "stochastic_parametric"
    search = "/Users/bjohnson/Projects/jades_d2s5/jobs/output/v2/{}*h5"
    files = glob.glob(search.format(ftype))
    results, obs, models = setup(files)
    names = results[0]["theta_labels"]

    # --- construct samples ----
    samples = [sample_posterior(res["chain"], res["weights"], nsample=nsample)
               for res in results]
    samples = [chain_to_struct(s, m, names=names) for s, m in zip(samples, models)]
    samples = construct_parameters(samples)
    truths = get_stoch_truths(results, catname=catname)
    truths = construct_stoch_parameters([truths])[0]

    #sys.exit()

    redshifts = truths["zred"]

    #sys.exit()

    zlims = [5, 6, 7, 8]

    cmap = matplotlib.cm.get_cmap('viridis')
    nbins = len(zlims) - 1
    fig, axes = pl.subplots(nbins, 1, sharex="col", figsize=(10.5, 9.5))
    for iz in range(nbins):
        ax = axes[iz]
        zlo, zhi = zlims[iz], zlims[iz + 1]
        choose = ((redshifts > zlo) & (redshifts < zhi))
        if choose.sum() == 1:
            continue
        dataset = [(np.squeeze(s[parameter]) / truths[tparameter][i])
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
        ax.axhline(1.0, linestyle=":", color="k")
        ax.set_ylabel(r"${}\, /\, {}_{{input}}$".format(parameter, parameter))

    ax.set_xlabel(r"$\log \, ({}_{{\rm input}})$".format(xparam), fontsize=14)
    [ax.set_ylim(0.1, 2.5) for ax in axes]
    axes[0].set_title("Mock={}; Model={}; S/N=DEEP without sizes".format(*ftype.split("_")))
    #fig.savefig("figures/delta_{}.png".format(parameter), dpi=600)
    #pl.show()
    return fig, axes


if __name__ == "__main__":

    with PdfPages("delta_stochastic_parametric.pdf") as pdf:
        fig, axes = delta_plot(parameter="mass", tparameter="totmass")
        pdf.savefig(fig)
        pl.close(fig)
        fig, axes = delta_plot(parameter="sfr", tparameter="sfr")
        pdf.savefig(fig)
        pl.close(fig)
        fig, axes = delta_plot(parameter="agem", tparameter="agem")
        pdf.savefig(fig)
