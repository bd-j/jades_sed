#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Make summary plots for a delay-tau fit to a NIRSPEC spectrum.
"""

import glob, sys
import numpy as np
import matplotlib.pyplot as pl

from prospect.io.read_results import results_from, get_sps
from jadepro.parfiles.parametric_fsps import build_model

from plotutils import chain_to_struct, marginal, sample_posterior
import sfhplot

from transforms import construct_parameters, construct_stoch_parameters
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
    gs = GridSpec(2, 3, height_ratios=[10, 10], width_ratios=[1, 1, 1],
                  left=0.1, right=0.87, wspace=0.28, hspace=0.28)

    fig = pl.figure(figsize=figsize)
    haxes = fig.add_subplot(gs[0, :])
    saxes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    return fig, haxes, saxes


pretty = {"mass": "M*", "sfr": "SFR", "agem": "<t>_m (Gyr)"}
truthpar = {"mass": "totmass", "sfr": "sfr", "agem": "agem"}


def show(fn, nsample=1000):

    truth_kwargs = {"color": "k", "linestyle": "-"}
    par_kwargs = {"color": "slateblue", "alpha": 0.5}

    res, obs, model = results_from(fn)
    if model is None:
        model = build_model(**res["run_params"])

    names = res["theta_labels"]
    objid = res["run_params"]["objid"]
    snr = np.median((obs["spectrum"] / obs["unc"])[obs["mask"]])

    # --- construct samples ----
    samples = sample_posterior(res["chain"], res["weights"], nsample=nsample)
    samples = chain_to_struct(samples, model, names=names)
    samples = construct_parameters([samples])[0]
    truths = get_stoch_truths([res])
    truths = construct_stoch_parameters([truths])[0]

    # get sfhs
    st, ss = sfhplot.stoch_params_to_sfh(truths[0], sig=0.01)
    pt, ps, _ = sfhplot.params_to_sfh(samples, time=st)

    pars = ["mass", "sfr", "agem"]
    x = np.array([np.squeeze(samples[p]) for p in pars])

    fig, hax, daxes = get_axes()
    hax = sfhplot.show_sfh(pt, sfrs=ps, ax=hax, post_kwargs=par_kwargs)
    hax.plot(st, ss, **truth_kwargs)
    hax.set_xlabel("lookback time (Gyr)")
    hax.set_ylabel("SFR")
    hax.set_title("object {}; $z$={:3.2f}, median snr={:3.1f}".format(objid, obs["object_redshift"], snr))
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

    return fig, hax, daxes


if __name__ == "__main__":

    ftype = "stochastic_parametric"
    search = "/Users/bjohnson/Projects/jades_d2s5/jobs/output/v2/{}_{}*h5"
    files = glob.glob(search.format(ftype, ""))

    with PdfPages("{}_sfhs.pdf".format(ftype)) as pdf:
        for fn in files:
            fig, h, d = show(fn)
            pdf.savefig(fig)
            pl.close(fig)