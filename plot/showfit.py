#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Make summary plots for a delay-tau fit to a NIRSPEC spectrum.
"""

import glob, sys
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import gammainc

from prospect.io.read_results import results_from, get_sps
from plotutils import chain_to_struct
from cornerplot import marginal

from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["STIXGeneral"]
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "serif"
rcParams["mathtext.sf"] = "serif"
rcParams['mathtext.it'] = 'serif:italic'


def delay_tau_sfr(tage=None, tau=None, mass=None, power=1, **extras):
    ''' just for the last age
    '''
    tt = tage / tau
    sfr = tt**power * np.exp(-tt) / tau
    mtot = gammainc(power+1, tt)
    return sfr/mtot * 1e-9 * mass

def struct_dict(struct):
    d = {n: struct[n] for n in struct.dtype.names}
    return d


def get_axes(figsize):
    gs = GridSpec(3, 2, width_ratios=[10, 10], height_ratios=[2, 1, 1],
                  left=0.1, right=0.87, wspace=0.28, hspace=0.28)

    fig = pl.figure(figsize=figsize)
    haxes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    sax = fig.add_subplot(gs[1,:])
    rax = fig.add_subplot(gs[2,:], sharex=sax)
    return fig, haxes, sax, rax


def dash():
    pass

if __name__ == "__main__":

    objid=2
    fn = glob.glob("obj{}*h5".format(objid))[0]

    res, obs, model = results_from(fn)
    sps = get_sps(res)
    imax = np.argmax(res["lnprobability"])
    pmax = res["chain"][imax, :]
    spec, phot, x = model.mean_model(pmax, obs=obs, sps=sps)

    samples = chain_to_struct(res["chain"], model)
    sampled = struct_dict(samples)
    weights = res["weights"]

    mass = np.squeeze(samples["mass"])
    sfr = np.squeeze(delay_tau_sfr(**sampled))
    x = np.array([mass, sfr])


    fig, haxes, sax, rax = get_axes((10, 8))
    ax = haxes[0]
    marginal(np.log10(mass), ax=ax, weights=weights, color="maroon", alpha=0.5)
    ax.axvline(np.log10(obs["input_params"]["mass"][0]), linestyle=":", color="k")
    ax.set_xlabel("log Formed Mass")
    ax = haxes[1]
    marginal(sfr, ax=ax, weights=weights, color="maroon", alpha=0.5)
    sfrin = delay_tau_sfr(**obs["input_params"])[0]
    ax.axvline(sfrin, linestyle=":", color="k")
    ax.set_xlabel("SFR (M$_\odot$ / yr)")

    #pl.show()
    #sys.exit()

    wave = obs["wavelength"] / 1e4

    ax = sax
    g = obs["mask"]
    ax.plot(wave[g], obs["spectrum"][g], color="slateblue", label="Mock")
    ax.plot(wave[g], spec[g], color="maroon", label="Best fit")
    ax.legend()

    ax = rax
    chi = (spec - obs["spectrum"]) / obs["unc"]
    ax.plot(wave[g], chi[g], color="orange", label="$\chi$")

    rax.set_xlabel(r"$\lambda \, (\mu m)$")
    rax.set_ylabel(r"$\chi$")

    fig.savefig("obj{}_dashboard.png".format(objid))
