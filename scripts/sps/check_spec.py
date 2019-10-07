#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as pl
pl.ion()

import h5py
from ckc.utils import construct_outwave

uv = construct_outwave(100, 1000, resolution=250, oversample=3)
opt = construct_outwave(1000, 2.4e4, resolution=500, oversample=3)
mir = construct_outwave(2.4e4, 5e4, resolution=500, oversample=3, logarithmic=True)
fir = construct_outwave(5e4, 1e8, resolution=50, oversample=3, logarithmic=True)
nw = len(uv) + len(opt) + len(mir) + len(fir)

import h5py

with h5py.File("c3k_v1.3_feh+0.00_afe+0.0.sed_jwst.h5", "r") as f:
    f.visit(print)
    print(f["spectra"].shape)
    wnat = f["wavelengths"][:]
    pnat = f["parameters"][:]
    gnat = f["spectra"][:].max(axis=-1) > 1e-32
    sn = f["spectra"][:]

with h5py.File("c3k_v1.3_feh+0.00_afe+0.0.sed_jwst.fsps.h5", "r") as ff:
    ff.visit(print)
    print(ff["spectra"].shape)
    wf = ff["wavelengths"][:]
    pf = ff["parameters"][:]
    gf = np.nanmax(ff["spectra"][:], axis=-1) > 1e-32
    sf = ff["spectra"][:]

# Bool flag for interpolated SEDs with nans
nan = np.sum(~np.isfinite(sf), axis=-1) > 0
bnan = (gf & nan)
binds = np.where(bnan)[0]
# biggest pixel with a nan (-1 for no nans)
mnan = np.array([np.argmax(np.insert(wf[~np.isfinite(s)], 0, 0)) for s in sf]) - 1

# bool flag for native grid SEDs with Nans
nannat = np.sum(~np.isfinite(sn), axis=-1) > 0
# bool flag for negative or zero numbers
negnat = np.sum(sn <= 0, axis=-1) > 0

mneg = np.array([np.argmax(np.insert(wf[s <= 0], 0, 0)) for s in sn]) - 1

fig, ax = pl.subplots()
ax.plot(wf, sf[gf][100, :])
ax.set_xlim(90, 1e4)



hf, hax = pl.subplots()
hax.plot(pf["logt"], pf["logg"], 'o', linestyle="", color="grey", alpha=0.5)
hax.plot(pnat["logt"], pnat["logg"], 'o', linestyle="", color="slateblue", markersize=3, alpha=0.7)
hax.plot(pf["logt"][gf], pf["logg"][gf], 's', linestyle="", color="green", markersize=8, alpha=0.5, markerfacecolor="none")
hax.plot(pf["logt"][bnan], pf["logg"][bnan], 's', linestyle="", color="maroon", markersize=8, alpha=0.7)
hax.set_ylim(6, -1.5)
hax.set_xlim(4.8, 3.2)


sys.exit()

from ckc.sed_to_fsps import *
basel_pars = get_basel_params()
cwave, cspec, valid = get_binary_spec(len(basel_pars), zstr="0.0200",
                                        speclib='BaSeL3.1/basel')

fluxfn = "/Users/bjohnson/Projects/ckc/ckc/spectra/fullres/c3k/c3k_v1.3_feh+0.00_afe+0.0.flux.h5"
#fluxfile = h5py.File(fluxfn, "r")

sedfile = "c3k_v1.3_feh+0.00_afe+0.0.sed_jwst.h5"
from prospect.sources import StarBasis
interpolator = StarBasis(sedfile, use_params=['logg', 'logt'], logify_Z=False,n_neighbors=1,rescale_libparams=True) 


