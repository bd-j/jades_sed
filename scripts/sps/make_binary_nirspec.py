#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert jwst resolution spectral library from h5 to FSPS
appropriate binary files
"""

import glob
import numpy as np
import matplotlib.pyplot as pl
from argparse import Namespace

from ckc.make_binary import prep_for_fsps


if __name__ == "__main__":

    fehlist = [-2.0, -1.75, -1.5, -1.25, -1.0,
               -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]

    args = Namespace()
    args.ck_vers = "c3k_v1.3"
    args.prefix = "ckc14_jwst"
    args.outdir = "/Users/bjohnson/Projects/jades_d2s5/data/c3k/"
    args.seddir = args.outdir
    args.sedname = "c3k_jwst"

    prep_for_fsps(fehlist=fehlist, args=args)
