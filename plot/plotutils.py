#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pickle
from prospect.models import priors


__all__ = ["sample_prior", "sample_posterior",
           "_quantile", "quantile",
           "setup", "chain_to_struct", "get_truths",
           "step", "fill_between"]


def sample_prior(model, nsample=1e6):
    """Generate samples from the prior.

    :param model:
        A ProspectorParams instance.

    :param nsample: (int, optional, default: 1000000)
        Number of samples to take
    """
    labels = model.free_params
    chain = np.zeros([nsample, model.ndim])
    #chain = []
    for l in labels:
        prior = model.config_dict[l]["prior"]
        if isinstance(prior, priors.TopHat):
            val = np.linspace(prior.params["mini"], prior.params["maxi"], nsample)
            val = np.atleast_2d(val).T
        else:
            val = np.array([prior.sample() for i in range(int(nsample))])
        chain[:, model.theta_index[l]] = np.array(val)
        # chain.append()
    #chain = np.concatenate([c.T for c in chain]).T
    return chain, labels


def sample_posterior(chain, weights=None, nsample=int(1e4),
                     start=0, thin=1, extra=None):
    """
    :param chain:
        ndarray of shape (niter, ndim) or (niter, nwalker, ndim)

    :param weights:
        weights for each sample, of shape (niter,)

    :param nsample: (optional, default: 10000)
        Number of samples to take

    :param start: (optional, default: 0.)
        Fraction of the beginning of the chain to throw away, expressed as a float in the range [0,1]

    :param thin: (optional, default: 1.)
        Thinning to apply to the chain before sampling (why would you do that?)

    :param extra: (optional, default: None)
        Array of extra values to sample along with the parameters of the chain.
        ndarray of shape (niter, ...)
    """
    start_index = np.floor(start * (chain.shape[-2] - 1)).astype(int)
    if chain.ndim > 2:
        flatchain = chain[:, start_index::thin, :]
        nwalker, niter, ndim = flatchain.shape
        flatchain = flatchain.reshape(niter * nwalker, ndim)
    elif chain.ndim == 2:
        flatchain = chain[start_index::thin, :]
        niter, ndim = flatchain.shape

    if weights is not None:
        p = weights[start_index::thin]
        p /= p.sum()
    else:
        p = None

    inds = np.random.choice(niter, size=nsample, p=p)
    if extra is None:
        return flatchain[inds, :]
    else:
        return flatchain[inds, :], extra[inds, ...]


def quantile(xarr, q, weights=None):
   qq = [_quantile(x, q, weights=weights) for x in xarr]
   return np.array(qq)


def _quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.

    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.

    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.

    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.

    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.

    """

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles


def setup(files, sps=0):
    from prospect.io import read_results as reader
    blob = [reader.results_from(f) for f in files]
    obs = [b[1] for b in blob]
    results = [b[0] for b in blob]
    models = [b[2] for b in blob]

    if sps == 0:
        sps = reader.get_sps(results[0])
    elif sps > 0:
        sps = [reader.get_sps(r) for r in results]

    return results, obs, models, sps


def chain_to_struct(chain, model=None, names=None):
    """Given a (flat)chain and a model, convert the chain to a structured array
    """
    indict = type(chain) == dict
    if indict:
        n = 1
    else:
        n = len(chain)

    if model is not None:
        if indict:
            model.params.update(chain)
        else:
            model.set_parameters(chain[0])
        names = model.free_params
        dt = [(p, model.params[p].dtype, model.params[p].shape)
            for p in names]
    else:
        dt = [(str(p), "<f8", (1,)) for p in names]

    #print(dt)
    struct = np.zeros(n, dtype=np.dtype(dt))
    for i, p in enumerate(names):
        if model is not None:
            inds = model.theta_index[p]
        else:
            inds = slice(i, i+1, None)
        if indict:
            struct[p] = chain[p]
        else:
            struct[p] = chain[:, inds]

    return struct


def get_truths(res):
    mp = res['obs']['mock_params']
    try:
        mp = pickle.loads(res['obs']['mock_params'])
    except:
        pass

    truth_dict, truth_vector = {}, []
    for k in res["theta_labels"]:
        try:
            v = mp[k]
        except(KeyError):
            kk = '_'.join(k.split('_')[:-1])
            num = int(k.split('_')[-1]) - 1
            v = np.array([mp[kk][num]])
            print(k, v)
        #if k in pmap:
        #    v = pmap[k](v)
        truth_dict[k] = v
        truth_vector.append(v)

    return np.concatenate(truth_vector), truth_dict


def step(xlo, xhi, y=None, ylo=None, yhi=None, ax=None,
         label=None, linewidth=2, **kwargs):
    """A custom method for plotting step functions as a set of horizontal lines
    """
    clabel = label
    for i, (l, h) in enumerate(zip(xlo, xhi)):
        if y is not None:
            ax.plot([l,h],[y[i],y[i]], label=clabel, linewidth=linewidth, **kwargs)
        if ylo is not None:
            ax.fill_between([l,h], [ylo[i], ylo[i]], [yhi[i], yhi[i]], linewidth=0, **kwargs)
        clabel = None


def fill_between(x, y1, y2=0, ax=None, linewidth=0, **kwargs):
    """Plot filled region between `y1` and `y2`.

    This function works exactly the same as matplotlib's fill_between, except
    that it also plots a proxy artist (specifically, a rectangle of 0 size)
    so that it can be added it appears on a legend.
    """
    import matplotlib.pyplot as pl
    ax = ax if ax is not None else pl.gca()
    ax.fill_between(x, y1, y2, linewidth=0, **kwargs)
    p = pl.Rectangle((0, 0), 0, 0, **kwargs)
    ax.add_patch(p)
    return p


def violinplot(data, pos, widths, ax=None, 
               violin_kwargs={"showextrema": False},
               color="slateblue", alpha=0.5, span=None, **extras):
    ndim = len(data)
    clipped_data = []

    if type(color) is str:
        color = ndim * [color]

    if span is None:
        span = [0.999999426697 for i in range(ndim)]

    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except(TypeError):
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            xmin, xmax = _quantile(data[i], q)
        good = (data[i] > xmin) & (data[i] < xmax)
        clipped_data.append(data[i][good])
    
    parts = ax.violinplot(data, positions=pos, widths=widths, **violin_kwargs)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(color[i])
        pc.set_alpha(alpha)