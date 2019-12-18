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

pl.rcParams["font.family"] = "serif"
pl.rcParams["font.serif"] = ["STIXGeneral"]
pl.rcParams["mathtext.fontset"] = "custom"
pl.rcParams["mathtext.rm"] = "serif"
pl.rcParams["mathtext.sf"] = "serif"
pl.rcParams['mathtext.it'] = 'serif:italic'


catname = ("/Users/bjohnson/Projects/jades_d2s5/data/"
           "noisy_spectra/parametric_mist_ckc14.h5")


if __name__ == "__main__":

    search = "/Users/bjohnson/Projects/jades_d2s5/jobs/output/v2/{}*h5"
    pnname = "parametric_nonparametric"
    pnfiles = glob.glob(search.format(pnname))
    pntag = "Mock={}\nModel={}\nS/N=DEEP with sizes".format(*pnname.split("_"))
    pnid = [int(os.path.basename(f).split("_")[2]) for f in pnfiles]

    ppname = "parametric_parametric"
    ppfiles = glob.glob(search.format(ppname))
    pptag = "Mock={}\nModel={}\nS/N=DEEP with sizes".format(*ppname.split("_"))
    ppid = [int(os.path.basename(f).split("_")[2]) for f in ppfiles]

    pnfiles = [name for n, name in zip(pnid, pnfiles) if n in ppid]
    ppfiles = [name for n, name in zip(ppid, ppfiles) if n in pnid]

    fig, axes = pl.subplots(3, 2)
    for i, par in enumerate(["mass", "sfr", "agem"]):
        ax = axes[i, 0]
        fig, ax = sdp_pp(ppfiles, zlims=(2, 7), parameter=par)
        ax.set_xlim(7, 10)
        ax = axes[i, 1]
        fig, ax = sdp_pn(pnfiles, zlims=(2, 7), parameter=par)
        ax.set_xlim(7, 10)

    pl.show()