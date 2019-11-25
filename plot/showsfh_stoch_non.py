#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Make summary plots for a delay-tau fit to a NIRSPEC spectrum.
"""

import glob, sys
import numpy as np
import matplotlib.pyplot as pl

from prospect.io.read_results import results_from, get_sps
import jadespro.parfiles.nonparametric_fsps as parfile

from plotutils import chain_to_struct, marginal, sample_posterior
from showspec import show_best_spec
import sfhplot

from transforms import construct_nonpar_parameters, construct_stoch_parameters
from transforms import get_stoch_truths

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["STIXGeneral"]
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "serif"
rcParams["mathtext.sf"] = "serif"
rcParams['mathtext.it'] = 'serif:italic'


catname = ("/Users/bjohnson/Projects/jades_d2s5/data/"
           "noisy_spectra/stochastic_mist_ckc14.h5")


def get_axes(figsize=None):
    gs = GridSpec(3, 3, height_ratios=[10, 10, 10], 
                  width_ratios=[1, 1, 1],
                  left=0.1, right=0.87, wspace=0.28, hspace=0.35)

    fig = pl.figure(figsize=figsize)
    saxes = fig.add_subplot(gs[0, :])
    haxes = fig.add_subplot(gs[1, :])
    paxes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    return fig, saxes, haxes, paxes

def prettify():
    pass

pretty = {"mass": "M*", "sfr": "SFR", "agem": "<t>_m (Gyr)"}
truthpar = {"mass": "totmass", "sfr": "sfr", "agem": "agem"}
fitpar = {"mass": "totmass", "sfr": "sfr1", "agem": "agem"}


def show(fn, nsample=1000, figsize=None):

    truth_kwargs = {"color": "k", "linestyle": "-"}
    par_kwargs = {"color": "slateblue", "alpha": 0.5}

    res, obs, model = results_from(fn)
    if model is None:
        model = parfile.build_model(**res["run_params"])
    agebins = model.params["agebins"].copy()

    names = res["theta_labels"]
    objid = res["run_params"]["objid"]
    snr = np.median((obs["spectrum"] / obs["unc"])[obs["mask"]])

    # --- construct samples ----
    samples = sample_posterior(res["chain"], res["weights"], nsample=nsample)
    samples = chain_to_struct(samples, model)
    samples = construct_nonpar_parameters([samples], agebins=agebins)[0]
    truths = get_stoch_truths([res], catname=catname)
    truths = construct_stoch_parameters([truths])[0]

    # get sfhs
    st, ss = sfhplot.stoch_params_to_sfh(truths[0], sig=0.01)
    npt, nps, _ = sfhplot.params_to_sfh(samples, agebins=agebins)

    pars = ["mass", "sfr", "agem"]
    x = np.array([np.squeeze(samples[fitpar[p]]) for p in pars])

    fig, sax, hax, daxes = get_axes(figsize=figsize)
    sax = show_best_spec(res, ax=sax)
    hax = sfhplot.show_sfh(npt, sfrs=nps, ax=hax, post_kwargs=par_kwargs)
    hax.plot(st, ss, **truth_kwargs)

    # Prettification
    hax.set_xlabel("lookback time (Gyr)")
    hax.set_ylabel("SFR")
    sax.set_xlabel(r"$\lambda \, (\mu m)$")
    sax.set_ylabel(r"$f_{\nu}$ (maggies)")
    sax.set_title("object {}; $z$={:3.2f}, median snr={:3.1f}".format(objid, obs["object_redshift"], snr))
    sax.legend()
    sax.set_xlim(0.5, 5.2)

    mc, mb = np.log10(truths[0]["mass"][0]), np.log10(truths[0]["mass"][1:].sum())
    hax.text(0.1, 0.8, "logm_const={:2.1f}\nlogm_bursts={:2.1f}".format(mc, mb),
             transform=hax.transAxes)

    for i, p in enumerate(pars):
        ax = daxes[i]
        marginal(np.log10(x[i]), ax=ax, color="slateblue", alpha=0.5)
        t = np.log10(truths[truthpar[p]][0])
        ax.axvline(t, linestyle=":", color="k")
        ax.set_xlabel("log " + pretty[p])

        xlim = list(ax.get_xlim())
        r = np.diff(xlim)
        xlim[0] = min(t - 0.1*r, xlim[0] - 0.1*r)
        xlim[1] = max(t + 0.1*r, xlim[1] + 0.1*r)
        ax.set_xlim(*xlim)
        ax.set_yticklabels("")

    return fig, sax, hax, daxes


if __name__ == "__main__":

    ftype = "stochastic_nonparametric"
    search = "/Users/bjohnson/Projects/jades_d2s5/jobs/output/v2/{}_{}*h5"
    files = glob.glob(search.format(ftype, ""))
    figsize = (7, 7)

    with PdfPages("figures/{}_sfhs.pdf".format(ftype)) as pdf:
        for fn in files:
            #try:
            print(fn)
            fig, s, h, d = show(fn, figsize=figsize)
            #except(ValueError):
            #    continue
            pdf.savefig(fig)
            pl.close(fig)