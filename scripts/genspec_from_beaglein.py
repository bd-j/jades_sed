#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to generate spectra with FSPS coresponding to particular BEAGLE
input parameters.
"""

import os, sys, glob
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import gamma, gammainc

from astropy.io import fits
from os.path import join as pjoin
from astropy.cosmology import Planck15 as cosmo

from prospect.utils.smoothing import smoothspec
from prospect.sources.constants import jansky_cgs, to_cgs_at_10pc
import fsps

to_cgs = to_cgs_at_10pc
datadir = "/Users/bjohnson/Projects/jades_d2s5/data/noisy_spectra/"
lsffile = "/Users/bjohnson/Projects/jades_d2s5/data/jwst_nirspec_prism_disp.fits"
lsf = np.array(fits.getdata(lsffile))

ckms = 2.998e5
sigma_to_fwhm = 2 * np.sqrt(2 * np.log(2))

def uni(x):
    if sys.version_info >= (3,):
        return x.decode()
    else:
        return x


def beagle_to_fsps(beagle, fneb=True):
    fpars = {}
    # Basic
    fpars["imf_upper_limit"] = 100
    fpars["imf_type"] = 1  # Chabrier
    # SFH
    fpars["sfh"] = 4 # delayed-tau
    fpars["zred"] = beagle["redshift"]
    fpars["tage"] = 10**(beagle["max_stellar_age"] - 9)
    fpars["tau"] = 10**(beagle["tau"] - 9)
    fpars["logzsol"] = beagle["metallicity"]
    # Dust
    mu, tveff = 0.3, beagle["tauV_eff"]
    fpars["dust2"] = mu * tveff
    fpars["dust1"] = (1-mu) * tveff
    # Neb
    fpars["add_neb_emission"] = True
    fpars["nebemlineinspec"] = fneb
    fpars["gas_logu"] = beagle["nebular_logU"]
    fpars["gas_logz"] = beagle["metallicity"]

    return fpars


def beagle_to_tabular(beagle, tyoung=0.01,
                      ntime=100, epsilon=1e-6):

    z = beagle["redshift"]
    tuniv = cosmo.age(z).to("Gyr").value          # Gyr
    mass = 10**beagle["mass"]                     # Msun
    maxage = 10**(beagle["max_stellar_age"] - 9)  # Gyr
    tau = 10**(beagle["tau"] - 9)                 # Gyr^{-1}
    sfr = 10**(beagle["sfr"] + 9)                 # Msun / Gyr
    tstart = tuniv - maxage                       # Gyr

    tdelay = np.linspace(tstart-epsilon, tuniv-tyoung, ntime)
    tnorm = (tdelay - tstart) / tau
    sdelay = tnorm * np.exp(-tnorm)
    sdelay[tnorm < 0] = 0.0
    mdelay = tau * gammainc(2, tnorm[-1])

    # add most recent bin
    my = sfr * tyoung
    if tyoung > 0:
        ty = np.array([tuniv - tyoung + epsilon, tuniv-epsilon])
        sy = np.zeros_like(ty) + sfr        
    else:
        ty = []
        sy = []

    sdelay *= (mass - my) / mdelay
    mdelay = mass - my

    t = np.hstack([tdelay, ty])
    s = np.hstack([sdelay, sy])
    return t, s * 1e-9


def genspec_fsps(beagle, sp, tabular=False,
                 tyoung=0.01, lineres=0):

    pars = beagle_to_fsps(beagle, fneb=(lineres <= 0))
    for k, v in pars.items():
        try:
            sp.params[k] = v
        except(KeyError):
            pass

    z = sp.params["zred"]
    sp.params["zred"] = 0
    if tabular:
        time, sfr = beagle_to_tabular(beagle, tyoung=tyoung)
        sp.params["sfh"] = 3
        sp.set_tabular_sfh(time, sfr)
        mass = 1.0
        tage=-99
    else:
        mass = 10**beagle["mass"]
        tage = pars["tage"]
    wave, spec = sp.get_spectrum(tage=tage)
    spec *= mass
    if lineres > 0:
        linelum = sp.emline_luminosity
        if linelum.ndim > 1:
            linelum = linelum[0]
        linewave = sp.emline_wavelengths
        sigma_v = ckms / lineres / sigma_to_fwhm
        linespec = lineprofile(wave, linewave, linelum, sigma_v)
        spec += linespec

    return wave, spec, sp.stellar_mass


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


def redshift_spectrum(wave, spec, zred, outwave=None,
                      smoothtype="lsf", sigma=150, Rlibrary=500,
                      **skwargs):
    """
    Parameters
    -----
    wave : ndarray of shape (nwave,) Units are angstroms.
           The spectral wavelength (Angstroms)

    spec : ndarray of shape (nwave,), units are Fnu
           The spectrum

    zred : float
           The redshift

    outwave : ndarray of shape (nout,) Units are angstroms
              The output wavelength grid (angstroms)

    lineres : float, optional,
              Resolution (lambda / FWHM) of the lines to be added to the
              library spectrum before smoothing.  If <= 0, then let FSPS
              add the lines.

    Rlibrary : float, optional
               The resolution (lambda/fwhm) of the input stellar/galaxy library

    smoothtype : string, optional
                 The type of smoothing to do.  One of "lsf" | "lambda" | "vel".
                 See prospect.utils.smoothspec for details.

    sigma : float, optional
            The resolution if the type of smoothing is not "lsf"

    """
    a = 1 + zred
    lumdist = cosmo.luminosity_distance(zred).to("Mpc").value
    dfactor = (lumdist * 1e5)**2

    wa, sa = wave * a, spec * a  # Observed Frame
    if outwave is not None:
        if smoothtype == "lsf":
            sig = lsf["WAVELENGTH"] / lsf["R"] / 2.355 * 1e4
            libsig = lsf["WAVELENGTH"] / Rlibrary / 2.355 * 1e4
            sig = np.sqrt(sig**2 - libsig**2)
            if np.any(sig <= 0):
                sig = np.clip(sig, 0, np.inf)
                lo = (sig == 0)
                print("{} points where library is too broad".format(lo.sum()))
                print("at lambda={}".format(lsf["WAVELENGTH"][lo].min()))
            sigma = np.interp(wa, lsf["WAVELENGTH"]*1e4, sig)
        sa = smoothspec(wa, sa, sigma, outwave=outwave,
                        smoothtype=smoothtype, **skwargs)

    # convert to mJy
    sa *= to_cgs / dfactor / (jansky_cgs) * 1e3
    return wa / 1e4, sa


def get_beagle(stype):

    inf = glob.glob(pjoin(datadir, stype, "*input*fits"))
    inp = np.array(fits.getdata(inf[0]))

    rows, ids, spec = [], [], []

    specf = glob.glob(pjoin(datadir, stype, "DEEP_R100*/*fits"))
    for i, f in enumerate(specf):
        sp = np.array(fits.getdata(f))
        spec.append(sp["fnu_noiseless"])
        idx = int(os.path.basename(f)[0])
        rows.append(inp[idx])
        ids.append(idx)

    wave = sp["wl"]
    return wave, np.array(spec), np.array(rows)


if __name__ == "__main__":

    # whether to add a constant sfr young component.
    tyoung = 0.00

    stype = "parametric"  # "parametric" | "stochastic"
    bwave, bspec, beaglein = get_beagle(stype)

    sp = fsps.StellarPopulation(zcontinuous=1)
    sp.params["add_neb_emission"] = True

    spectra, lspectra = [], []
    for row in beaglein:
        blob = genspec_fsps(row, sp, tabular=True, tyoung=tyoung)
        wave, spec, mstar = blob
        spectra.append(spec)
        mstar = np.atleast_1d(mstar)
        print(mstar[0])

        blob = genspec_fsps(row, sp, tabular=True, tyoung=tyoung, lineres=500)
        wave, spec, mstar = blob
        lspectra.append(spec)

    fspec = np.array(spectra)
    lspec = np.array(lspectra)
    # smooth with wavelength dependent LSF
    frspec = [redshift_spectrum(wave, s, z, outwave=bwave*1e4, fftsmooth=False)[1]
              for s, z in zip(fspec, beaglein["redshift"])]
    # smooth with constant dlambda
    #flspec = [redshift_spectrum(wave, s, z, outwave=bwave*1e4, smoothtype="lambda")[1]
    #          for s, z in zip(fspec, beaglein["redshift"])]
    # Add lines by hand before smoothing
    flspec = [redshift_spectrum(wave, s, z, outwave=bwave*1e4, fftsmooth=False)[1]
              for s, z in zip(lspec, beaglein["redshift"])]


    # don't smooth or resample
    fospec = [redshift_spectrum(wave, s, z, outwave=None)
              for s, z in zip(fspec, beaglein["redshift"])]

    # --- Plot spectra ---

    fig, ax = pl.subplots(len(beaglein), 2, sharex=True, figsize=(12.5, 13))
    for i in range(len(bspec)):
        a = ax[i, 0]
        a.plot(bwave, bspec[i], alpha=0.6, 
               color="maroon", label="BEAGLE/Pandeia")
        a.plot(bwave, frspec[i], alpha=0.6,
               color="slateblue", label="Prospect/LSF")
        a.plot(bwave, flspec[i], alpha=0.6, color="orange",
               label=r"Prospect/$\sigma_\lambda=150\AA$")
        a = ax[i, 1]
        a.plot(bwave, frspec[i] / bspec[i], alpha=0.6, color="slateblue",)
        a.plot(bwave, flspec[i] / bspec[i], alpha=0.6, color="orange")
        a.axhline(1.0, linestyle="--", color="black")
        a.axhline(1.3, linestyle=":", color="black", alpha=0.5)
        a.axhline(0.7, linestyle=":", color="black", alpha=0.5)
        #a.plot(fospec[i][0], fospec[i][1])
        #ax.plot(time, sfr)

    [a.set_xlabel(r"$\lambda (\mu m, observed)$") for a in ax[-1, :]]
    ax[0, 0].legend()

    fig.savefig("figures/beagle_vs_pro.pdf")
    pl.close(fig)

    # --- Plot SFHs ---
    fig, ax = pl.subplots()
    for i, row in enumerate(beaglein):
        time, sfr = beagle_to_tabular(row, tyoung=tyoung)
        ax.plot(time, sfr, label="Galaxy #{}".format(row["ID"]))

    ax.set_xlabel("time since Big Bang (Gyr)")
    ax.set_ylabel(r"$SFR (M_{\odot}/yr)$")
    ax.legend()
    fig.savefig("figures/beagle_sfhs.pdf")
    pl.close(fig)
