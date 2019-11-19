#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for comparing the Prospector model spectra to BEAGLE model spectra.
Requires the 'parametric_fsps` module.
"""

import numpy as np
import matplotlib.pyplot as pl

from prospect import prospect_args
from prospect.sources.constants import jansky_cgs, to_cgs_at_10pc
from parametric_fsps import uni, build_model, build_sps, build_obs, get_beagle

from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["STIXGeneral"]
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "serif"
rcParams["mathtext.sf"] = "serif"
rcParams['mathtext.it'] = 'serif:italic'


if __name__ == "__main__":

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--smoothssp', action="store_true",
                        help="If set, smooth the SSPs before constructing composite spectra.")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--datadir', type=str, default="/Users/bjohnson/Projects/jades_d2s5/data/",
                        help="location of the beagle parameters and S/N curves")
    parser.add_argument('--seed', type=int, default=101,
                        help=("RNG seed for the noise. Negative values result"
                              "in random noise."))

    args = parser.parse_args()
    run_params = vars(args)
    sps = build_sps(**run_params)
    run_params["sps_libraries"] = [uni(l) for l in sps.ssp.libraries]

    tag = "parametric_{}_{}".format(*run_params["sps_libraries"])

    pro, beagle = [], []
    objids = np.arange(10)
    for objid in objids:
        bw, bsn, bc, bsp = get_beagle(objid, datadir=run_params["datadir"])
        if bsp is None:
            continue
        obs = build_obs(objid=objid, sps=sps, **run_params)
        pro.append(obs)
        beagle.append(bsp)

    fig, ax = pl.subplots(len(beagle), 2, sharex=True,
                          figsize=(12.5, 2 + 3*(len(beagle))), squeeze=False)
    for i, (p, b) in enumerate(zip(pro, beagle)):
        a = ax[i, 0]
        assert np.allclose(b["wl"], p["wavelength"] / 1e4)
        wave = b["wl"]
        spec = (p["spectrum"] - p["added_noise"]) * 3631e3
        g = b["sn"] > 1
        a.plot(wave[g], b["fnu_noiseless"][g], alpha=0.6,
               color="maroon", label="BEAGLE+Pandeia")
        a.plot(wave[g], spec[g], alpha=0.6,
               color="slateblue", label="Prospect")
        inp = p["input_params"]
        txt = "z={:3.2f}\nZ={:3.2f}".format(inp["zred"][0], inp["logzsol"][0])
        a.text(0.1, 0.7, txt, transform=a.transAxes)
        a = ax[i, 1]
        ratio = spec / b["fnu_noiseless"]
        a.plot(wave[g], ratio[g], color="orange")
        a.axhline(1.0, linestyle="--", color="black")
        a.axhline(1.3, linestyle=":", color="black", alpha=0.5)
        a.axhline(0.7, linestyle=":", color="black", alpha=0.5)
        a.set_ylabel("Prospect / Beagle")

    [a.set_xlabel(r"$\lambda (\mu m, observed)$") for a in ax[-1, :]]
    ax[0, 0].legend()

    fig.savefig("figures/{}_beagle_vs_pro.pdf".format(tag))
    pl.close(fig)
