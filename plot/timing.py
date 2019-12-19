#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, glob, os
import numpy as np
import matplotlib.pyplot as pl

from plotutils import setup

if __name__ == "__main__":

    search = "/Users/bjohnson/Projects/jades_d2s5/jobs/output/v2/{}*h5"

    bins = np.arange(1, 19)

    fig, ax = pl.subplots()

    for mock in ["parametric", "stochastic"]:
        for mod in ["parametric", "nonparametric"]:
            ftype = "{}_{}".format(mock, mod)
            if ftype == "stochastic_nonparametric":
                continue
            files = glob.glob(search.format(ftype))
            results, obs, models = setup(files)
            times = np.array([r["sampling_duration"] for r in results])
            nc = np.array([np.sum(r["ncall"]) for r in results])
            ax.hist(times / 3600., bins=bins, density=True, 
                    histtype="step", linewidth=3, label=ftype)
            print(ftype, np.median(times / nc))

    ax.legend()
    ax.set_xlabel("Sampling Duration (hours)")
    ax.set_ylabel("Normalized frequency")
    pl.show()
    
    fig.savefig("figures/pr_timing.pdf")