#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to make spectra using SFHs described by constant SFR with a
couple stochastic bursts
"""

import time, sys
from copy import deepcopy
import numpy as np

from parametric_fsps import build_sps, uni


def build_model(ncomp=1, add_duste=False, add_neb=True, **extras):
    """Build a model appropriate for a stochastic SFH.  This model is
    *not* meant to be fit as such
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import SedModel, transforms, priors
    # --- Get a basic delay-tau SFH parameter set. ---
    model_params = TemplateLibrary["parametric_sfh"]
    # add components
    for par in ["mass", "tage", "sfh", "const"]:
        init = np.zeros(ncomp)
        init[0] = 1
        model_params[par] = {"isfree": False, "N": ncomp, "init": init.copy()}

    # --- Adjust model initial values to be BEAGLEish ---
    # IMF
    model_params["imf_type"]["init"] = 1
    model_params["imf_upper_limit"] = {"N": 1, "isfree": False, "init": 100}
    # power-law dust with fixed dust1 / dust2 = 1.5 (== (1-0.4)/0.4)
    #model_params["dust2"]["init"] = dust2
    model_params["dust_type"]["init"] = 0
    model_params["dust1"] = {"N": 1, "isfree": False, "init": 0.,
                             'depends_on': transforms.dustratio_to_dust1}
    model_params["dust1_index"] = {"N": 1, "isfree": False, "init": -1.3}
    model_params["dust_ratio"] = {"N": 1, "isfree": False, "init": 1.5,
                                  "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    # metallicity
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2.1, maxi=0.25)
    #model_params["logzsol"]["init"] = logzsol

    # Dust emission
    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    # Nebular emission (not added to spectrum within FSPS)
    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])
        model_params["nebemlineinspec"]["init"] = False
        model_params["add_neb_continuum"]["init"] = True
        #model_params["gas_logu"]["init"] = -3

    return SedModel(model_params)


def build_obs(objid=0, datafile="", seed=0, sps=None,
              fullspec=True, sgroup="DEEP_R100", **kwargs):
    
    from prospect.utils.obsutils import fix_obs

    # Get stochastic parameters and S/N for this object
    wave, snr, scat = get_stochastic(objid, datafile=datafile, sgroup=sgroup)
    fsps_pars, ncomp = stochastic_to_fsps(scat)
    
    # now get a model, set it to the input values, and compute
    model = build_model(ncomp=ncomp, **kwargs)
    model.params.update(fsps_pars)
    assert np.isfinite(model.prior_product(model.theta))

    # Get SPS
    if sps is None:
        sps = build_sps(object_redshift=scat["redshift"], **kwargs)

    # Barebones obs dictionary,
    # use the full output wavelength array if `fullspec`
    if fullspec:
        wave = sps.ssp.wavelengths
        snr = 1000
    else:
        assert wave is not None
    obs = {"wavelength": wave, "spectrum": None, "filters": None}
    
    # Generate model
    spec, phot, mfrac = model.mean_model(model.theta, sps=sps, obs=obs)
    
    # make some noise in here
    unc = spec / np.clip(snr, 1e-2, np.inf)
    if int(seed) > 0:
        np.random.seed(int(seed))
    noise = np.random.normal(0, 1.0, size=len(unc)) * unc

    noisy_spec = spec + noise
    mask = (snr > 1e-2)
    mock = {"spectrum": noisy_spec,
            "unc": unc,
            "mask": mask,
            "filters": None,
            "maggies": None,
            "added_noise": noise,
            "object_redshift": scat["redshift"][0],
            "seed": seed}

    mock["model_params"] = deepcopy(model.params)

    obs.update(mock)
    # This ensures all required keys are present
    obs = fix_obs(obs)

    return obs


def stochastic_to_fsps(cat):
    
    ncomp = int(cat["nburst"] + 1)
    
    fpars = {}
    fpars["mass"]  = cat["mass"][0, :ncomp]
    fpars["sfh"]   = cat["sfh"][0, :ncomp]
    fpars["tage"]  = cat["tage"][0, :ncomp]
    fpars["const"] = cat["const"][0, :ncomp]
    fpars["zred"]  = cat["redshift"]
    fpars["logzsol"] = cat["metallicity"]
    
    mu, tveff = 0.4, cat["tauV_eff"]
    fpars["dust2"] = mu * tveff
    fpars["dust_ratio"] = 1.5
    # Neb
    fpars["gas_logu"] = cat["gas_logu"]
    fpars["gas_logz"] = cat["metallicity"]

    return fpars, ncomp


def get_stochastic(idx, datafile="", sgroup="DEEP_R100"):
    import h5py
    with h5py.File(datafile, "r") as data:
        cat = data[str(idx)]["stochastic_parameters"][()]
        try:
            sdat = data[str(idx)][sgroup]
            #spec = sdat["fnu_noiseless"][:]
            wave = sdat["wl"][:] * 1e4
            snr = sdat["sn"][:]
        except:
            #print("Could not find unique spec for {}".format(idx))
            wave = None
            #spec = None
            snr = 100

    return wave, snr, cat


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
                                 "noisy_spectra/stochastic_mist_ckc14.h5"),
                        help="File with beagle parameters and S/N curves")
    parser.add_argument("--sgroup", type=str, default="DEEP_R100",
                        help=("The type of pandeia mock to use for the wavelength"
                              " vector and S/N curve"))
    parser.add_argument('--seed', type=int, default=101,
                        help=("RNG seed for the noise. Negative values result"
                              "in random noise."))


    args = parser.parse_args()
    run_params = vars(args)
    run_params["object_redshift"] = 4.0
    
    model = build_model(**run_params)
    print(model)
    sys.exit()
    
    #sps = build_sps(**run_params)
    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__