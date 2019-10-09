#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Use the full resolution ckc library to produce jwst nirspec appropriate R=500 spectra
"""

import os
from itertools import product

import numpy as np
import h5py

from prospect.sources import StarBasis
from ckc.ckc_to_fsps import sed, to_basel


# lambda_lo, lambda_hi, R_{out, fwhm}, use_fft
segments = [(100., 1000., 250, False),
            (1000., 2.4e4, 500, True),
            (2.4e4, 5e4, 500, True),
            (5e4, 1e8, 50, True)
            ]
oversample = 3
logarithmic = True


if __name__ == "__main__":

    # These are the set of feh and afe from which we will choose based on zindex
    fehlist = [-2.0, -1.75, -1.5, -1.25, -1.0,
               -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
    afelist = [0.0]

    from ckc.utils import get_ckc_parser
    # key arguments are:
    #  * --zindex
    #  * --oversample
    #  * --ck_vers
    #  * --basedir
    #  * --seddir
    #  * --sedname
    parser = get_ckc_parser()
    args = parser.parse_args()

    # -- Mess with some args ---
    args = parser.parse_args()
    args.fulldir = args.fulldir.format(args.ck_vers)

    # --- CHOOSE THE METALLICITY ---
    if args.zindex < 0:
        # for testing off odyssey
        feh, afe = 0.0, 0.0
    else:
        metlist = list(product(fehlist, afelist))
        feh, afe = metlist[args.zindex]
    print(feh, afe)

    # --- make the sed file ---
    sedfile = sed(feh, afe, segments, args)

    # --- Make the SED interpolated to basel logt, logg grid ---
    if "sedfile" not in locals():
        template = "{}/{}_feh{:+3.2f}_afe{:+2.1f}.{}.h5"
        sedfile = template.format(args.seddir, args.ck_vers, feh, afe, args.sedname)
    args.nowrite = False
    out = to_basel(feh, afe, sedfile, args)
    if args.nowrite:
        basel_pars, bwave, bspec, inds = out
