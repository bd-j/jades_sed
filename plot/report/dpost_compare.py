#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot P(delta parameter) vs parameter with vilin plots.
"""

import sys, os, glob, time
import numpy as np
import matplotlib.pyplot as pl
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from dpost_par_non import single_delta_plot as sdp_pn
from dpost_par_par import single_delta_plot as sdp_pp
from dpost_stoch_par import single_delta_plot as sdp_sp
from dpost_stoch_non import single_delta_plot as sdp_sn

pl.rcParams["font.family"] = "serif"
pl.rcParams["font.serif"] = ["STIXGeneral"]
pl.rcParams["mathtext.fontset"] = "custom"
pl.rcParams["mathtext.rm"] = "serif"
pl.rcParams["mathtext.sf"] = "serif"
pl.rcParams['mathtext.it'] = 'serif:italic'


if __name__ == "__main__":

    search = "/Users/bjohnson/Projects/jades_d2s5/jobs/output/v2/{}*h5"

    if False:
        # --- Parametric Mocks ---
        pnname = "parametric_nonparametric"
        pnfiles = glob.glob(search.format(pnname))
        pntag = "Mock={}\nModel={}".format(*pnname.split("_"))
        pnid = [int(os.path.basename(f).split("_")[2]) for f in pnfiles]

        ppname = "parametric_parametric"
        ppfiles = glob.glob(search.format(ppname))
        pptag = "Mock={}\nModel={}".format(*ppname.split("_"))
        ppid = [int(os.path.basename(f).split("_")[2]) for f in ppfiles]

        pnfiles = [name for n, name in zip(pnid, pnfiles) if n in ppid]
        ppfiles = [name for n, name in zip(ppid, ppfiles) if n in pnid]

        fig, axes = pl.subplots(3, 2, sharex="col", figsize=(10, 8))
        for i, par in enumerate(["mass", "sfr", "agem"]):
            ax = axes[i, 0]
            _, ax = sdp_pp(ppfiles, ax=ax, zlims=(3, 5), parameter=par)
            ax.set_xlim(7, 10)
            ax = axes[i, 1]
            _, ax = sdp_pn(pnfiles, ax=ax, zlims=(3, 5), parameter=par)
            ax.set_xlim(7, 10)

        [ax.set_xlabel("") for ax in axes[:2, :].flat]
        #axes[0, 0].set_title(pptag.replace("\n", ", "))
        #axes[0, 1].set_title(pntag.replace("\n", ", "))
        ax = axes[0, 0]
        ax.text(0.6, 0.7, pptag, transform=ax.transAxes)
        ax = axes[0, 1]
        ax.text(0.6, 0.7, pntag, transform=ax.transAxes)
        fig.suptitle("S/N=DEEP with sizes\n$3 < z < 5$")
        pl.show()
        fig.savefig("report/deltaX_parametric_compare.pdf")

    if True:
        # --- Stochastic Mocks ---
        snname = "stochastic_nonparametric"
        snfiles = glob.glob(search.format(snname))
        sntag = "Mock={}\nModel={}".format(*snname.split("_"))
        snid = [int(os.path.basename(f).split("_")[2]) for f in snfiles]

        spname = "stochastic_parametric"
        spfiles = glob.glob(search.format(spname))
        sptag = "Mock={}\nModel={}".format(*spname.split("_"))
        spid = [int(os.path.basename(f).split("_")[2]) for f in spfiles]

        #pnfiles = [name for n, name in zip(pnid, pnfiles) if n in ppid]
        #ppfiles = [name for n, name in zip(ppid, ppfiles) if n in pnid]

        fig, axes = pl.subplots(3, 2, sharex="col", figsize=(10, 8))
        for i, par in enumerate(["mass", "sfr", "agem"]):
            ax = axes[i, 0]
            _, ax = sdp_sp(spfiles, ax=ax, zlims=(5, 9), parameter=par)
            ax.set_xlim(7.5, 10.5)
            ax = axes[i, 1]
            _, ax = sdp_sn(snfiles, ax=ax, zlims=(5, 9), parameter=par)
            ax.set_xlim(7.5, 10.5)

        [ax.set_xlabel("") for ax in axes[:2, :].flat]
        #axes[0, 0].set_title(pptag.replace("\n", ", "))
        #axes[0, 1].set_title(pntag.replace("\n", ", "))
        ax = axes[0, 0]
        ax.text(0.6, 0.7, sptag, transform=ax.transAxes)
        ax = axes[0, 1]
        ax.text(0.6, 0.7, sntag, transform=ax.transAxes)
        fig.suptitle("S/N=DEEP with sizes\n$3 < z < 5$")
        pl.show()
        fig.savefig("report/deltaX_stochastic_compare.pdf")
