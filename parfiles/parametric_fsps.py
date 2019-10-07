#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Script to fit a NIRSPEC R=100 (PRISM) spectrum of a high-z galaxy
"""

import time, sys, glob
from os.path import join as pjoin
from copy import deepcopy

import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.utils.smoothing import sigma_to_fwhm, ckms

from sedpy.observate import getSED
from prospect.sources import CSPSpecBasis, to_cgs
from prospect.sources.constants import cosmo, jansky_cgs, lightspeed

try:
    from astropy.io import fits
    nirspec_lsf_file = "/Users/bjohnson/Projects/jades_d2s5/data/jwst_nirspec_prism_disp.fits"
    prism_lsf = np.array(fits.getdata(nirspec_lsf_file))
except:
    pass

# --------------
# Model Definition
# --------------


def uni(x):
    if sys.version_info >= (3,):
        return x.decode()
    else:
        return x


def build_model(fixed_metallicity=None, add_duste=False, add_neb=True,
                object_redshift=None, prism_lsf=prism_lsf, smoothssp=False,
                library_resolution=500, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param add_duste: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel, transforms

    # --- Get a basic delay-tau SFH parameter set. ---
    model_params = TemplateLibrary["parametric_sfh"]

    # --- Adjust model initial values to be BEAGLEish ---
    # IMF
    model_params["imf_type"]["init"] = 1
    model_params["imf_upper_limit"] = {"N": 1, "isfree": False, "init": 100}
    # power-law dust with fixed dust1 / dust2 = 1.5 (== (1-0.4)/0.4)
    model_params["dust_type"]["init"] = 0
    model_params["dust1"] = {"N": 1, "isfree": False, "init": 0.,
                             'depends_on': transforms.dustratio_to_dust1}
    model_params["dust1_index"] = {"N": 1, "isfree": False, "init": -1.3}
    model_params["dust_ratio"] = {"N": 1, "isfree": False, "init": 1.5,
                                  "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    # adjust priors
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e10)

    # --- NIRSPEC PRISM smoothing ---
    model_params["library_resolution"] = {"N": 1, "isfree": False, 
                                          "init": library_resolution}
    if (prism_lsf is not None) & (~smoothssp):
        model_params.update(TemplateLibrary["spectral_smoothing"])
        model_params["fftsmooth"]["init"] = False
        model_params["smoothtype"]["init"] = "lsf"
        model_params["sigma_smooth"]["init"] = None
        model_params["sigma_smooth"]["isfree"] = False
        model_params["lsf_function"] = {"isfree": False, "init": nirspec_lsf}        
        #model_params["nirspec_prism_lsf"] = {"isfree": False, "init": prism_lsf}

    # --- Optional bells ---
    # Metallicity
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

    # Redshift
    if object_redshift != 0.0:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift
        # Set age prior to max age of universe
        maxage = cosmo.age(object_redshift).to_value("Gyr")
        tprior = priors.TopHat(mini=0.001, maxi=maxage)
        model_params["tage"]["prior"] = tprior

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
        model_params["gas_logu"]["isfree"] = True

    # Now instantiate the model using this new dictionary of parameter
    # specifications
    model = sedmodel.SedModel(model_params)

    return model


def nirspec_lsf(observed_wave, library_resolution=500, 
                nirspec_prism_lsf=prism_lsf, **extras):
    """Get sigma(lambda_observed) for NIRSPEC PRISM. (units of \AA)
    """
    lwave = nirspec_prism_lsf["WAVELENGTH"] * 1e4
    sig = lwave / nirspec_prism_lsf["R"] / sigma_to_fwhm
    libsig = lwave / library_resolution / sigma_to_fwhm
    sig = np.clip(np.sqrt(sig**2 - libsig**2), 0, np.inf)
    sigma = np.interp(observed_wave, lwave, sig)
    return sigma

# --------------
# SPS Object
# --------------

def lineprofile(wave, linewave, linelum, sigma):
    """Lay down multiple gaussians (in velocity) on the given wave-axis.

    output units are lum / Hz
    """
    cang = ckms * 1e13
    mu, A, sigma = np.atleast_2d(linewave), np.atleast_2d(linelum), np.atleast_2d(sigma)
    dv = ckms * (wave[:, None] / mu - 1)
    dv_dnu = ckms * wave[:, None]**2 / (cang * mu)

    # this gives lum / km/s
    val = (A / (sigma * np.sqrt(np.pi * 2)) *
           np.exp(-dv**2 / (2 * sigma**2)))
    # convert to lum / Hz
    val *= dv_dnu
    return val.sum(axis=-1)


class JadesSpecBasis(CSPSpecBasis):

    def get_spectrum(self, outwave=None, filters=None, **params):
        # Spectrum in Lsun/Hz per solar mass formed, restframe
        wave, spectrum, mfrac = self.get_galaxy_spectrum(**params)

        # Add nebular lines (note they are assumed normalized for one solar
        # mass here, for tabular SFHs need to divide out total mass)
        lineres = params.get("library_resolution", [500])
        lines_added = params.get("nebemlineinspec", True)
        if (lineres[0] > 0) & (~lines_added[0]):
            linelum = self.ssp.emline_luminosity
            if linelum.ndim > 1:
                linelum = linelum[0]
            linewave = self.ssp.emline_wavelengths
            sigma_v = ckms / lineres / sigma_to_fwhm
            linespec = lineprofile(wave, linewave, linelum, sigma_v)
            spectrum += linespec

        # Redshifting + Wavelength solution
        # We do it ourselves.
        a = 1 + params.get('zred', 0)
        wa, sa = wave * a, spectrum * a  # Observed Frame
        if outwave is None:
            outwave = wa

        # Observed frame photometry, as absolute maggies
        if filters is not None:
            mags = getSED(wa, lightspeed/wa**2 * sa * to_cgs, filters)
            phot = np.atleast_1d(10**(-0.4 * mags))
        else:
            phot = 0.0

        # Spectral smoothing.
        do_smooth = (('sigma_smooth' in params) and
                     ('sigma_smooth' in self.reserved_params))
        if do_smooth:
            # We do it ourselves.
            smspec = self.smoothspec(wa, sa, None, lsf=params["lsf_function"][0],
                                     outwave=outwave, **params)
        elif outwave is not wa:
            # Just interpolate
            smspec = np.interp(outwave, wa, sa, left=0, right=0)
        else:
            # no interpolation necessary
            smspec = sa

        # Distance dimming and unit conversion
        zred = self.params.get('zred', 0.0)
        if (zred == 0) or ('lumdist' in self.params):
            # Use 10pc for the luminosity distance (or a number
            # provided in the dist key in units of Mpc)
            dfactor = (self.params.get('lumdist', 1e-5) * 1e5)**2
        else:
            lumdist = cosmo.luminosity_distance(zred).value
            dfactor = (lumdist * 1e5)**2
        # Spectrum will be in maggies
        smspec *= to_cgs / dfactor / (3631*jansky_cgs)

        # Convert from absolute maggies to apparent maggies
        phot /= dfactor

        # Mass normalization
        mass = np.sum(self.params.get('mass', 1.0))
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            # Convert input normalization units from current stellar mass to mass formed
            mass /= mfrac

        return smspec * mass, phot * mass, mfrac


def build_sps(zcontinuous=1, compute_vega_mags=False, smoothssp=False,
              object_redshift=None, library_resolution=500,
              lsf_file=nirspec_lsf_file, **extras):

    sps = JadesSpecBasis(zcontinuous=zcontinuous,
                         compute_vega_mags=compute_vega_mags)
    assert sps.ssp.libraries[1] == b"ckc14"

    # Do LSF smoothing at the ssp level?
    # Faster but need to know object redshift
    if smoothssp:
        from astropy.io import fits
        prism_lsf = np.array(fits.getdata(lsf_file))
        w = sps.ssp.wavelengths
        wa = w * (1 + object_redshift)
        sigma = nirspec_lsf(wa, library_resolution=library_resolution,
                            nirspec_prism_lsf=prism_lsf)
        sps.ssp.params['smooth_lsf'] = True
        sps.ssp.params["smooth_velocity"] = False
        sps.ssp.set_lsf(wave, sigma)

    return sps


# --------------
# Observational Data
# --------------

def build_obs(objid=0, datadir="", seed=0, sps=None,
              prism_lsf=prism_lsf, **kwargs):
    """Load spectrum from a FITS file

    :param specfile:
        Name (and path) of the ascii file containing the photometry.

    :param idx: int
        The catalog object index

    :returns obs:
        Dictionary of observational data.
    """
    from prospect.utils.obsutils import fix_obs

    # Get BEAGLE parameters and S/N for this object
    bwave, snr, bcat, _ = get_beagle(objid, datadir=datadir)
    fsps_pars = beagle_to_fsps(bcat)
    if sps is None:
        sps = build_sps(object_redshift=bcat["redshift"], **kwargs)

    # Barebones obs dictionary,
    # set to nirspec wavelength points if lsf is supplied.
    fullspec = prism_lsf is None
    if fullspec:
        bwave = sps.ssp.wavelengths
        snr = 1000
    else:
        assert bwave is not None
    obs = {"wavelength": bwave, "spectrum": None, "filters": None}

    # now get a model, set it to the beagle values, and compute
    model = build_model(object_redshift=bcat["redshift"], prism_lsf=prism_lsf, **kwargs)
    model.params.update(fsps_pars)
    assert np.isfinite(model.prior_product(model.theta))

    spec, phot, mfrac = model.mean_model(model.theta, obs=obs, sps=sps)

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
            "object_redshift": bcat["redshift"][0],
            "object_id": objid}

    mock["input_params"] = deepcopy(fsps_pars)

    obs.update(mock)
    # This ensures all required keys are present
    obs = fix_obs(obs)

    return obs


def beagle_to_fsps(beagle):
    fpars = {}
    # Basic
    fpars["mass"] = 10**beagle["mass"]
    fpars["zred"] = beagle["redshift"]
    # SFH
    fpars["tage"] = 10**(beagle["max_stellar_age"] - 9)
    fpars["tau"] = 10**(beagle["tau"] - 9)
    fpars["logzsol"] = beagle["metallicity"]
    # Dust
    mu, tveff = 0.3, beagle["tauV_eff"]
    fpars["dust2"] = mu * tveff
    fpars["dust_ratio"] = 2.33
    # Neb
    fpars["gas_logu"] = beagle["nebular_logU"]
    fpars["gas_logz"] = beagle["metallicity"]

    return fpars


def get_beagle(idx, mock_sfh_type="parametric", datadir="", catf=None):
    from astropy.io import fits

    if catf is None:
        stype = mock_sfh_type
        catf = glob.glob(pjoin(datadir, "noisy_spectra", stype, "*input.fits"))[0]
    allcat = np.array(fits.getdata(catf))
    cat = allcat[allcat["ID"] == idx]
    specf = glob.glob(pjoin(datadir, "noisy_spectra", stype, "DEEP_R100",
                            "{}_*fits".format(idx)))
    if len(specf) == 1:
        bsp = np.array(fits.getdata(specf[0]))
        spec = bsp["fnu_noiseless"]
        wave = bsp["wl"] * 1e4
        snr = bsp["sn"]
    else:
        print("Could not find unique spec for {}".format(idx))
        bsp = None
        wave = None
        snr = 100

    return wave, snr, cat, bsp

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------

def build_all(**kwargs):

    obs = build_obs(**kwargs)
    model = build_model(object_redshift=obs["object_redshift"], **kwargs)

    return obs, model, build_sps(**kwargs), build_noise(**kwargs)


if __name__ == '__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--smoothssp', action="store_true",
                        help="If set, smooth the SSPs before constructing composite spectra.")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--objid', type=int, default=0,
                        help="zero-index row number in the table to fit.")
    parser.add_argument('--datadir', type=str, default="/Users/bjohnson/Projects/jades_d2s5/data/" ,
                        help="location of the beagle parameters and S/N curves")
    parser.add_argument('--seed', type=int, default=101,
                        help=("RNG seed for the noise. Negative values result"
                              "in random noise."))

    args = parser.parse_args()
    run_params = vars(args)
    obs, model, sps, noise = build_all(**run_params)

    run_params["object_redshift"] = obs["object_redshift"]
    run_params["sps_libraries"] = sps.ssp.libraries
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
