#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Script to fit a NIRSPEC R=100 (PRISM) spectrum of a high-z
galaxy with a delay-tau SFH and BEAGLE-like parameters
"""

import time, sys, glob
from os.path import join as pjoin
from copy import deepcopy

import numpy as np
from astropy.io import fits

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.utils.smoothing import sigma_to_fwhm, ckms

from sedpy.observate import getSED, load_filters
from prospect.sources import CSPSpecBasis, to_cgs
from prospect.sources.constants import cosmo, jansky_cgs, lightspeed


try:
    nirspec_lsf_file = "/Users/bjohnson/Projects/jades_d2s5/data/jwst_nirspec_prism_disp.fits"
    nirspec_lsf_table = np.array(fits.getdata(nirspec_lsf_file))
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
                object_redshift=None, library_resolution=500, sublibres=False,
                smoothstars=False, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param add_duste: (optional, default: False) Switch to add (fixed)
        parameters relevant for dust emission.

    :param add_neb: (optional, default: False) Switch to add (fixed) parameters
        relevant for nebular emission, and turn nebular emission on.

    :param smoothstars:
        If True, smooth the spectrum by a velocity dispersion given by the sigma_smooth parameter

    :param object_redshift: (optional, float)
        If supplied, fix the redshift to this number

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
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2.1, maxi=0.25)
    model_params["tau"]["prior"] = priors.LogUniform(mini=1e-2, maxi=10)
    model_params["mass"]["prior"] = priors.LogUniform(mini=5e5, maxi=1e11)

    # --- Smoothing ---
    if smoothstars:
        model_params.update(TemplateLibrary["spectral_smoothing"])

    # Fix Redshift?
    if object_redshift != 0.0:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift
        # Set age prior to max age of universe
        maxage = cosmo.age(object_redshift).to_value("Gyr")
        tprior = priors.TopHat(mini=0.001, maxi=maxage)
        model_params["tage"]["prior"] = tprior

    # --- Optional bells ---
    # Metallicity
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

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

    def line_spread(self, observed_wave):
        """
        :param observed_wave: array-like, shape(nw,)
            Observed wavelengths in Angstroms

        :returns sigma: array-like, shape(nw,)
            The dispersion of the gaussian linespread function, in AA at each of
            the supplied wavelengths.
        """
        lwave = self.lsf_table["WAVELENGTH"] * 1e4
        sig = lwave / self.lsf_table["R"] / sigma_to_fwhm
        if self.sublibres & (self.library_resolution > 0):
            libsig = lwave / self.library_resolution / sigma_to_fwhm
        else:
            libsig = 0
        sig = np.clip(np.sqrt(sig**2 - libsig**2), 0, np.inf)
        return np.interp(observed_wave, lwave, sig)


    def get_spectrum(self, outwave=None, filters=None, **params):
        # Spectrum in Lsun/Hz per solar mass formed, restframe
        wave, spectrum, mfrac = self.get_galaxy_spectrum(**params)
        zred = self.params.get('zred', 0.0)
        a = 1 + zred
        mass = np.sum(self.params.get('mass', 1.0))

        # Add nebular lines (note they are assumed normalized for one solar
        # mass here, for tabular SFHs need to divide out total mass)
        lines_added = (np.atleast_1d(self.params.get("nebemlineinspec", [True]))[0] &
                       np.atleast_1d(self.params.get("add_neb_emission", [True]))[0])
        if (~lines_added):
            linelum = self.ssp.emline_luminosity
            if linelum.ndim > 1:
                # tabular sfh
                linelum = linelum[0] / mass
            linewave = self.ssp.emline_wavelengths
            # This is the line width at the library resolution
            sigma_v = ckms / self.library_resolution / sigma_to_fwhm
            # Use the linespread function if it was done for the stars
            if self.ssp.params["smooth_lsf"]:
                sigma_lsf = self.line_spread(linewave * a)
                sv_lsf = ckms * sigma_lsf / (linewave * a)
                sigma_v = np.hypot(sigma_v, sv_lsf)

            # could restrict to relevant lines here....
            self._linespec = lineprofile(wave, linewave, linelum, sigma_v)
            spectrum += self._linespec

        # Redshifting + Wavelength solution
        # We do it ourselves.
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
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            # Convert input normalization units from current stellar mass to mass formed
            mass /= mfrac

        return smspec * mass, phot * mass, mfrac


def build_sps(zcontinuous=1, compute_vega_mags=False,
              object_redshift=None, smoothssp=False, library_resolution=500,
              sublibres=False, lsf_file="", **extras):

    sps = JadesSpecBasis(zcontinuous=zcontinuous,
                         compute_vega_mags=compute_vega_mags)

    # Add the data necessary for the lsf smoothing
    assert sps.ssp.libraries[1] == b"ckc14"
    sps.library_resolution = library_resolution
    sps.sublibres = sublibres
    if lsf_file == "":
        sps.lsf_table = nirspec_lsf_table
    else:
        sps.lsf_table = np.array(fits.getdata(lsf_file))

    # Do LSF smoothing at the ssp level?
    # Faster but need to know object redshift
    print(object_redshift)
    if smoothssp:
        w = sps.ssp.wavelengths
        wa = w * (1 + object_redshift)
        sigma = sps.line_spread(wa)
        sigma_v = ckms * sigma / wa
        good = (wa > 0.5e4) & (wa < 10e4)
        sps.ssp.params['smooth_lsf'] = True
        sps.ssp.params["smooth_velocity"] = True
        sps.ssp.set_lsf(w[good], sigma_v[good])

    return sps


# --------------
# Observational Data
# --------------

def build_obs(objid=0, datafile="", seed=0, sps=None,
              fullspec=False, sgroup="DEEP_R100_withSizes", **kwargs):
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
    wave, snr, bcat = get_beagle(objid, datafile=datafile, sgroup=sgroup)
    fsps_pars = beagle_to_fsps(bcat)
    
    # now get a model, set it to the beagle values, and compute
    model = build_model(object_redshift=bcat["redshift"], **kwargs)
    model.params.update(fsps_pars)
    assert np.isfinite(model.prior_product(model.theta))

    # Get SPS
    if sps is None:
        sps = build_sps(object_redshift=bcat["redshift"], **kwargs)

    # Barebones obs dictionary,
    # use the full output wavelength array if `fullspec`
    if fullspec:
        wave = sps.ssp.wavelengths
        snr = 1000
    else:
        assert wave is not None
    obs = {"wavelength": wave, "spectrum": None, "filters": None}

    # Build the spectrum
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
            "object_redshift": bcat["redshift"],
            "object_id": objid}

    mock["input_params"] = deepcopy(fsps_pars)
    mock["model_params"] = deepcopy(model.params)

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
    mu, tveff = 0.4, beagle["tauV_eff"]
    fpars["dust2"] = mu * tveff
    fpars["dust_ratio"] = 1.5
    # Neb
    fpars["gas_logu"] = beagle["nebular_logU"]
    fpars["gas_logz"] = beagle["metallicity"]

    return fpars


def get_beagle(idx, datafile="", sgroup="DEEP_R100_withSizes"):
    import h5py
    with h5py.File(datafile, "r") as data:
        cat = data[str(idx)]["beagle_parameters"][()]
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
    sps = build_sps(object_redshift=obs["object_redshift"], **kwargs)

    return obs, model, sps, build_noise(**kwargs)


if __name__ == '__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--fullspec', action="store_true",
                        help="If set, generate the full wavelength array.")
    parser.add_argument('--smoothssp', default=True,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--sublibres', default=True,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--lsf_file', type=str, default=(""),
                        help="File with the LSF data to use when smoothing SSPs")
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
