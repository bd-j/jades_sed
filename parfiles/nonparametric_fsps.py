#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Script to fit a NIRSPEC R=100 (PRISM) spectrum of a high-z
galaxy with a non-parametric SFH.
"""

import time, sys
from copy import deepcopy

import numpy as np
from astropy.io import fits

from sedpy.observate import getSED, load_filters
from prospect.sources import FastStepBasis, to_cgs
from prospect.utils.smoothing import sigma_to_fwhm, ckms
from prospect.sources.constants import cosmo, jansky_cgs, lightspeed


try:
    nirspec_lsf_file = "/Users/bjohnson/Projects/jades_d2s5/data/jwst_nirspec_prism_disp.fits"
    nirspec_lsf_table = np.array(fits.getdata(nirspec_lsf_file))
except:
    pass

# --------------
# Model Definition
# --------------


def build_model(fixed_metallicity=None, add_duste=False, add_neb=True,
                object_redshift=0.0, library_resolution=500, sublibres=False,
                smoothstars=False, nbins_sfh=7, **extras):
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
    from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins
    from prospect.models import priors, sedmodel, transforms

    maxage = cosmo.age(object_redshift).to_value("Gyr")
    
    # --- Get a basic parameter set and augment with continuity sfh ---
    model_params = TemplateLibrary["ssp"]
    _ = model_params.pop("tage")
    model_params.update(TemplateLibrary["continuity_sfh"])
    model_params = adjust_continuity_agebins(model_params, tuniv=maxage, nbins=nbins_sfh)

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
    model_params["logmass"]["prior"] = priors.TopHat(mini=5.3, maxi=11)

    # --- Smoothing ---
    if smoothstars:
        model_params.update(TemplateLibrary["spectral_smoothing"])

    # Fix Redshift?
    if object_redshift != 0.0:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

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


class JadesStepBasis(FastStepBasis):

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
            linewave, linelum = self.get_galaxy_elines()
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

    sps = JadesStepBasis(zcontinuous=zcontinuous,
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
    parser.add_argument('--nbins_sfh', type=int, default=8, 
                        help="Number of bins in the nonparametric sfh")
    # --- ssp ---
    parser.add_argument('--fullspec', action="store_true",
                        help="If set, generate the full wavelength array.")
    parser.add_argument('--smoothssp', default=True,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--sublibres', default=True,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--lsf_file', type=str, default=(""),
                        help="File with the LSF data to use when smoothing SSPs")

    args = parser.parse_args()
    run_params = vars(args)
    run_params["object_redshift"] = 4.0
    
    model = build_model(**run_params)
    print(model)
    sys.exit()
    
    #sps = build_sps(**run_params)
    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__