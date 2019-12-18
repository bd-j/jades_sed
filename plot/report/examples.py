#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
import h5py

import sfhplot
from transforms import construct_stoch_parameters

from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["STIXGeneral"]
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "serif"
rcParams["mathtext.sf"] = "serif"
rcParams['mathtext.it'] = 'serif:italic'

stoch_catname = ("/Users/bjohnson/Projects/jades_d2s5/data/"
                 "noisy_spectra/stochastic_mist_ckc14.h5")
par_catname = ("/Users/bjohnson/Projects/jades_d2s5/data/"
               "noisy_spectra/parametric_mist_ckc14_rhalf.h5")


if __name__ == "__main__":

    if True:
        # --- stoch SFH plot ---
        objid = 10

        jcat = []
        with h5py.File(stoch_catname, "r") as catalog:
            jcat.append(catalog[str(objid)]["stochastic_parameters"][()])
        truths = np.hstack(jcat)
        truths = construct_stoch_parameters([truths])[0]

        st, ss = sfhplot.stoch_params_to_sfh(truths[0], sig=0.01)

        truth_kwargs = {"color": "k", "linestyle": "-"}
        hfig, hax = pl.subplots(figsize=(8, 3.6),)
        hax.plot(st, ss, **truth_kwargs)
        hax.set_xlabel("lookback time (Gyr)")
        hax.set_ylabel("SFR (M$_\odot$/yr)")
        mc, mb = np.log10(truths[0]["mass"][0]), np.log10(truths[0]["mass"][1:].sum())
        z = truths[0]["redshift"]
        tag = "z={:3.2f}\nlogm_const={:2.1f}\nlogm_bursts={:2.1f}".format(z, mc, mb)
        hax.text(0.1, 0.8, tag,
                transform=hax.transAxes)
        hfig.savefig("report/example_sfh_stochastic.pdf")
        pl.show()

    if False:
        # --- Spectrum plot ----
        objid = 300
        fig, axes = pl.subplots(2, 1, sharex=True, figsize=(8, 3),
                                gridspec_kw={"height_ratios": [3,1]})
        with h5py.File(par_catname, "r") as catalog:

            mock = catalog[str(objid)]["prospector_intrinsic"]
            sim = catalog[str(objid)]["DEEP_R100_withSizes"]
            pars = catalog[str(objid)]["beagle_parameters"]
            
            tag = "z={:3.2f}, S/N = DEEP with Sizes"
            tag = tag.format(pars["redshift"])
            #, 10**pars["sfr"], 10**pars["stellar_mass"])

            ax = axes[0]
            ax.plot(mock["wavelength"][:] / 1e4, mock["spectrum"][:] * 3631e3,
                    label="Prospector Model", color="k")
            ax.plot(sim["wl"][:], sim["fnu_noiseless"][:],
                    label="Pandeia prediction", color="tomato", linewidth=2)
            ax.plot(sim["wl"][:], sim["fnu"][:],
                    label="Noised prediction", color="royalblue")
            ax.set_ylabel(r"$f_\nu$ (mJy)")
            ax.legend()
            ax.set_title(tag)
            
            ax = axes[1]
            ax.plot(sim["wl"][:], sim["sn"][:])
            ax.set_xlabel(r"$\lambda_{obs}$ ($\mu$m)")
            ax.set_ylabel("S/N/pixel")

            [ax.set_xlim(0.68, 5.1) for ax in axes]
            fig.savefig("report/example_spectrum.pdf")
            pl.show()