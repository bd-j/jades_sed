#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Script to fit a mock NIRSPEC R=100 (PRISM) spectrum -- generated
with a parametric SFH -- of a high-z galaxy with a delay-tau SFH and 
BEAGLE-like parameters
"""

import time, sys
import numpy as np
import h5py

from prospect.fitting import fit_model
from prospect.io import write_results as writer

from parametric_fsps import build_model, build_sps, build_obs, uni

# -----------
# Everything
# ------------

def build_all(**kwargs):
    with h5py.File(kwargs["datafile"], "r") as cat:
        idx = str(kwargs["objid"])
        red = cat[idx]["beagle_parameters"]["redshift"]
    sps = build_sps(object_redshift=red, **kwargs)
    obs = build_obs(sps=sps, **kwargs)
    model = build_model(object_redshift=red, **kwargs)

    return obs, model, sps, (None, None)


if __name__ == '__main__':

    # - Parser with default arguments -
    from prospect import prospect_args
    parser = prospect_args.get_parser()
    # - Add custom arguments -

    # --- model ---
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    # --- ssp ---
    parser.add_argument('--fullspec', action="store_true",
                        help="If set, generate the full wavelength array.")
    parser.add_argument('--smoothssp', default=True,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--sublibres', default=True,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--lsf_file', type=str, default=(""),
                        help="File with the LSF data to use when smoothing SSPs")
    # --- data ---
    parser.add_argument('--objid', type=int, default=0,
                        help="zero-index row number in the table to fit.")
    parser.add_argument('--datafile', type=str,
                        default=("/Users/bjohnson/Projects/jades_d2s5/data/"
                                 "noisy_spectra/parametric_mist_ckc14.h5"),
                        help="File with beagle parameters and S/N curves")
    parser.add_argument("--sgroup", type=str, default="DEEP_R100_withSizes",
                        help=("The type of pandeia mock to use for the wavelength"
                              " vector and S/N curve"))
    parser.add_argument('--seed', type=int, default=101,
                        help=("RNG seed for the noise. Negative values result"
                              "in random noise."))

    args = parser.parse_args()
    run_params = vars(args)
    obs, model, sps, noise = build_all(**run_params)

    run_params["object_redshift"] = obs["object_redshift"]
    run_params["sps_libraries"] = [uni(l) for l in sps.ssp.libraries]
    run_params["param_file"] = __file__

    print(model)

    if args.debug:
        sys.exit()

    #hfile = setup_h5(model=model, obs=obs, **run_params)
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())
    hfile = "{0}_{1}_mcmc.h5".format(args.outfile, ts)
    output = fit_model(obs, model, sps, noise, **run_params)

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass
