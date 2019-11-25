#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
import h5py

from prospect.utils.plotting import get_best
from astropy.cosmology import Planck15 as cosmo


def show_mock(catalog, idx, ax):

    etc = catalog[str(idx)]["DEEP_R100"]
    cat = catalog[str(idx)]["stochastic_parameters"]
    ax.plot(etc["wl"], etc["fnu_noiseless"])
    ax.plot(etc["wl"], etc["fnu"], alpha=0.5)
    info = "z={:4.2f}\nS/N={:2.1f}".format(cat["redshift"][0], np.nanmedian(etc["sn"]))
    ax.text(0.1, 0.8, info, transform=ax.transAxes)

    return ax


def show_best_spec(res, ax=None, as_residual=False, module=None,
                   data_kwargs={"color": "k", "label": "Data"},
                   truth_kwargs={"color": "maroon", "label": "Noiseless"},
                   model_kwargs={"color": "slateblue", "label": "Best Model"}):
    obs = res["obs"]
    wave, spec = obs["wavelength"].copy(), obs["spectrum"]
    unc, mask = obs["unc"], obs["mask"]
    wave /= 1e4
    try:
        true = spec - obs["added_noise"]
        snr = true / unc
    except(KeyError):
        true = None
        snr = spec / unc

    try:
        best = res["bestfit"]
        wb, sb = obs["wavelength"].copy(), best["spectrum"]
    except(KeyError):
        #print("no stored spectrum")
        wb, sb = get_best_spec(res, module=module)

    wb /= 1e4

    if as_residual:
        spec = (spec - sb) / unc
        unc = 1.0
        wb, sb = 0, 0

    m = mask & (snr > 0.2) & (unc > 0)

    ax.plot(wave[m], spec[m], **data_kwargs)
    ax.fill_between(wave[m], (spec-unc)[m], (spec+unc)[m],
                    alpha=0.1, color="k")
    if true is not None:
        ax.plot(wave[m], true[m], **truth_kwargs)
    ax.plot(wb, sb, **model_kwargs)

    lims = spec[m].min(), spec[m].max()
    r = np.diff(lims)
    ax.set_ylim(lims[0] - 0.1 * r, lims[1] + 0.1 * r)

    return ax


def get_best_spec(res, module=None):
    if module is not None:
        sps = module.build_sps(**res["run_params"])
        model = module.build_model(**res["run_params"])
        _, best = get_best(res)
        s, p, x = model.mean_model(best, obs=res["obs"], sps=sps)
        return res["obs"]["wavelength"], s
    else:
        return 0, 0


if __name__ == "__main__":
    catalog = h5py.File("stochastic_mist_ckc14.h5", "r")
    objlist = list(catalog.keys())
    objlist = [str(f) for f in range(len(objlist))]
    sgroup = "DEEP_R100"

    snr, jcat = [], []
    for strobj in objlist:
        try:
            etc = catalog[strobj][sgroup]
        except:
            continue
        snr.append(etc["sn"][:])
        jcat.append(catalog[strobj]["stochastic_parameters"][()])
    jcat = np.hstack(jcat)
    snr = np.array(snr)
    median_snr = np.nanmedian(snr, axis=-1)

    oo = np.argsort(median_snr)
    idx = oo[-5]
