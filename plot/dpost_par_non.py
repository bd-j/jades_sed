#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot P(delta parameter) vs parameter with vilin plots.
"""

import sys, os, glob, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

import jadespro.parfiles.nonparametric_fsps as parfile

from plotutils import sample_posterior, chain_to_struct, setup
from transforms import construct_parameters, construct_nonpar_parameters
from transforms import get_truths


pl.rcParams["font.family"] = "serif"
pl.rcParams["font.serif"] = ["STIXGeneral"]
pl.rcParams["mathtext.fontset"] = "custom"
pl.rcParams["mathtext.rm"] = "serif"
pl.rcParams["mathtext.sf"] = "serif"
pl.rcParams['mathtext.it'] = 'serif:italic'


catname = ("/Users/bjohnson/Projects/jades_d2s5/data/"
           "noisy_spectra/parametric_mist_ckc14_rscale.h5")

pretty = {"mass": "M_*", "sfr": "SFR", "agem": "<t>_m"}
truthpar = {"mass": "mass", "sfr": "sfr", "agem": "agem"}
fitpar = {"mass": "totmass", "sfr": "sfr1", "agem": "agem"}


def single_delta_plot(files, ax=None, parameter="mass", xparam="mass",
                      zlims=(2, 7), nsample=500):

    results, obs, models = setup(files)
    names = results[0]["theta_labels"]
    if models[0] is None:
        models = [parfile.build_model(**r["run_params"]) for r in results]
    agebins = [m.params["agebins"].copy() for m in models]

    # --- construct samples ----
    samples = [sample_posterior(res["chain"], res["weights"], nsample=nsample)
               for res in results]
    samples = [chain_to_struct(s, m, names=names) 
               for s, m in zip(samples, models)]
    samples = [construct_nonpar_parameters([s], agebins=a)[0]
               for s, a in zip(samples, agebins)]
    truths = get_truths(results, catname=catname)
    truths = construct_parameters([truths])[0]
    tparameter = truthpar[parameter]

    redshifts = truths["zred"]
    zlo, zhi = zlims

    # --- Plot setup ---
    if ax is None:
        fig, ax = pl.subplots()
    else:
        fig = None
    choose = ((redshifts > zlo) & (redshifts < zhi))
    if choose.sum() < 1:
        return fig, ax
    # colorbar and prettify
    pmin, pmax = zlo, zhi
    cmap = matplotlib.cm.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(vmin=pmin, vmax=pmax)
    if fig is not None:
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label("redshift")

    # --- Plot data ---
    dataset = [(np.squeeze(s[fitpar[parameter]]) / truths[tparameter][i])
               for i, s in enumerate(samples) if choose[i]]
    positions = np.log10(truths[truthpar[xparam]][choose])

    vparts = ax.violinplot(dataset, positions, widths=0.15,
                           showmedians=False, showmeans=False,
                           showextrema=False)

    zreds = (redshifts[choose] - pmin) / (pmax - pmin)
    for z, pc in zip(zreds, vparts['bodies']):
        pc.set_facecolor(cmap(z))
        pc.set_edgecolor(cmap(z))
        pc.set_alpha(0.5)

    # --- Prettify ----
    ax.axhline(1.00, linestyle=":", color="k")
    ax.set_ylabel(r"${}$ (output / input)".format(pretty[parameter]))

    ax.set_xlabel(r"$\log \, ({}_{{\rm input}})$".format(xparam), fontsize=14)
    ax.set_ylim(0.1, 2.5)
    return fig, ax


if __name__ == "__main__":

    ftype = "parametric_nonparametric"
    search = "/Users/bjohnson/Projects/jades_d2s5/jobs/output/v2/{}*h5"
    files = glob.glob(search.format(ftype))
    tag = "Mock={}\nModel={}\nS/N=DEEP with sizes".format(*ftype.split("_"))

    zlims = [3, 4, 5]
    nbins = len(zlims) - 1

    with PdfPages("figures/deltaX_parametric_nonparametric.pdf") as pdf:
        for par in ["mass", "sfr", "agem"]:
            fig, axes = pl.subplots(nbins, 1, sharex="col", figsize=(10.5, 9.5))
            for iz in range(nbins):
                zl = (zlims[iz], zlims[iz+1])
                _, ax = single_delta_plot(files, ax=axes[iz], parameter=par,
                                          zlims=zl)
                ax.text(0.8, 0.75, "${:3.1f} < z <  {:3.1f}$".format(*zl), transform=ax.transAxes)
            [ax.set_xlim(7, 10) for ax in axes]
            axes[0].set_title(tag.replace("\n", ", "))
            pdf.savefig(fig)
            pl.close(fig)
            

        #fig, axes = delta_plot(parameter="mass")
        #pdf.savefig(fig)
        #pl.close(fig)
        #fig, axes = delta_plot(parameter="sfr")
        #[ax.set_xlim(7, 10) for ax in axes]
        #pdf.savefig(fig)
        #pl.close(fig)
        #fig, axes = delta_plot(parameter="agem")
        #[ax.set_xlim(6.8, 9.5) for ax in axes]
        #pdf.savefig(fig)
