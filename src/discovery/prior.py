import re

import numpy as np
import pandas as pd

from . import matrix
jnp = matrix.jnp

def uniform(par, a, b):
    def logpriorfunc(params):
        return matrix.jnp.where(matrix.jnp.logical_and(params[par] >= a, params[par] <= b), 0, -matrix.jnp.inf)

    return logpriorfunc


priordict_standard = {
    "(.*_)?efac": [0.9, 1.1],
    "(.*_)?t2equad": [-8.5, -5],
    "(.*_)?tnequad": [-8.5, -5],
    "(.*_)?log10_ecorr": [-8.5, -5],
    "(.*_)?rednoise_log10_A.*": [-20, -11],
    "(.*_)?rednoise_gamma.*": [0, 7],
    "(.*_)?rednoise_log10_fb": [-9, -6],
    "(.*_)?red_noise_log10_A": [-18, -11],  # deprecated
    "(.*_)?red_noise_gamma": [0, 7],  # deprecated
    "(.*_)?red_noise_log10_fb": [-9, -6],
    "crn_log10_A.*": [-18, -11],
    "crn_gamma.*": [0, 7],
    "crn_log10_fb": [-9, -6],
    "gw_(.*_)?log10_A": [-18, -11],
    "gw_(.*_)?gamma": [0, 7],
    "gw_log10_fb": [-9, -6],
    "(.*_)?dmgp_log10_A": [-20, -11],
    "(.*_)?dmgp_gamma": [0, 7],
    "(.*_)?dmgp_alpha": [1, 3],
    "(.*_)?chromgp_log10_A": [-20, -11],
    "(.*_)?chromgp_gamma": [0, 7],
    "(.*_)?chromgp_alpha": [1, 7],
    "(.*_)?dm_gp_log10_A": [-20, -11],
    "(.*_)?dm_gp_gamma": [0, 7],
    "(.*_)?dm_gp_alpha": [1, 3],
    "(.*_)?chrom_gp_log10_A": [-20, -11],
    "(.*_)?chrom_gp_gamma": [0, 7],
    "(.*_)?chrom_gp_alpha": [2.5, 14], # scattering noise. Should be steeper than DM
    "crn_log10_rho": [-9, -4],
    "gw_(.*_)?log10_rho": [-9, -4],
    r"(.*_)?red_noise_log10_rho\(([0-9]*)\)": [-9, -4],
    r"(.*_)?red_noise_crn_log10_rho\(([0-9]*)\)": [-9, -4],
    "cw_ra": [0, 2*np.pi],
    "cw_dec": [-0.5*np.pi, 0.5*np.pi],
    "cw_inc": [0, np.pi],
    "cw_sindec": [-1.0, 1.0],
    "cw_cosinc": [-1.0, 1.0],
    "cw_psi": [0, np.pi],
    "cw_log10_f0": [-9.0, -7.0],
    "cw_log10_h0": [-18.0, -11.0],
    "cw_phi_earth": [0., 2*np.pi],
    "(.*_)?cw_phi_psr": [0., 2*np.pi],
}

def getprior_uniform(par, priordict={}):
    priordict = {**priordict_standard, **priordict}

    for parname, range in priordict.items():
        if re.match(parname, par):
            return range

    raise KeyError(f'getprior_uniform: no prior for parameter {par}.')

def makelogprior_uniform(params, priordict={}):
    priordict = {**priordict_standard, **priordict}

    priors = []
    for par in params:
        for parname, range in priordict.items():
            if re.match(parname, par):
                priors.append(uniform(par, *range))
                break

    def logprior(params):
        return sum(prior(params) for prior in priors)

    return logprior


def makelogtransform_uniform(func, priordict={}):
    priordict = {**priordict_standard, **priordict}

    # figure out slices when there are vector arguments
    slices, offset = [], 0
    for par in func.params:
        if '(' in par:
            l = int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1
            slices.append(slice(offset, offset+l))
            offset = offset + l
        else:
            slices.append(offset)
            offset = offset + 1

    # build vectors of DF column names and of lower and upper uniform limits
    a, b = [], []
    columns = []
    for par, slice_ in zip(func.params, slices):
        for pname, prange in priordict.items():
            if re.match(pname, par):
                therange = prange
                break
        else:
            raise KeyError(f"No known prior for {par}.")

        if '(' in par:
            root = par[:par.index('(')]
            l = int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1

            for i in range(l):
                columns.append(f'{root}[{i}]')
                a.append(therange[0])
                b.append(therange[1])
        else:
            columns.append(par)
            a.append(therange[0])
            b.append(therange[1])

    a, b = matrix.jnparray(a), matrix.jnparray(b)

    def to_dict(ys):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))

        if len(a) != len(func.params):
            return {par: xs[slice_] for par, slice_ in zip(func.params, slices)}
        else:
            return dict(zip(func.params, xs))

    def to_vec(params):
        xs = jnp.zeros_like(a)
        for par, slice_ in zip(func.params, slices):
            xs = xs.at[slice_].set(params[par])

        # only Python 3.11: xs = jnp.r_[*[params[pname] for pname in func.params]]
        # only scalar parameters: xs = matrix.jnparray([params[pname] for pname in func.params])

        return jnp.arctanh((a + b - 2*xs)/(a - b))

    def to_df(ys, psrs=None):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))

        if psrs is None:
            return pd.DataFrame(np.array(xs), columns=columns)
        else:
            # rename columns from psr number to psr name
            psrdict = {f'{i}]': psr.name for i, psr in enumerate(psrs)}
            psrcols = [psrdict[par.split('[')[1]] + '_' + par.split('[')[0] if '[' in par else par for par in columns]
            return pd.DataFrame(np.array(xs), columns=psrcols).sort_index(axis=1)

    def logprior(ys):
        return jnp.sum(jnp.log(2.0) - 2.0 * jnp.logaddexp(ys, -ys))

    def logL(ys):
        return func(to_dict(ys))

    def transformed(ys):
        return logL(ys) + logprior(ys)

    transformed.params = func.params

    transformed.logprior = logprior
    transformed.logL = logL

    transformed.to_dict = to_dict
    transformed.to_vec = to_vec
    transformed.to_df = to_df

    return transformed


def makelogtransform_classic(func, priordict={}):
    priordict = {**priordict_standard, **priordict}

    a, b = [], []
    for par in func.params:
        for pname, prange in priordict.items():
            if re.match(pname, par):
                a.append(prange[0])
                b.append(prange[1])
                break
        else:
            raise KeyError(f"No known prior for {par}.")

    a, b = matrix.jnparray(a), matrix.jnparray(b)

    def to_dict(ys):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        return dict(zip(func.params, xs))

    def to_vec(params):
        xs = matrix.jnparray([params[pname] for pname in func.params])
        return jnp.arctanh((a + b - 2*xs)/(a - b))

    def to_df(ys):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        return pd.DataFrame(np.array(xs), columns=func.params)

    def logprior(ys):
        return jnp.sum(jnp.log(2.0) - 2.0 * jnp.logaddexp(ys, -ys))

        # return jnp.sum(jnp.log(0.5) - 2.0 * jnp.log(jnp.cosh(ys)))
        # but   log(0.5) - 2 * log(cosh(y))
        #     = log(0.5) - 2 * log((exp(x) + exp(-x))/2)
        #     = log(0.5) - 2 * (log(exp(x) - exp(-x)) - log(2.0))
        #     = log(2.0) - 2 * logaddexp(x, -x)

    def logL(ys):
        return func(to_dict(ys))

    def transformed(ys):
        return logL(ys) + logprior(ys)

    transformed.params = func.params

    transformed.logprior = logprior
    transformed.logL = logL

    transformed.to_dict = to_dict
    transformed.to_vec = to_vec
    transformed.to_df = to_df

    return transformed


def sample_uniform(params, priordict={}, n=1, fail=True):
    priordict = {**priordict_standard, **priordict}

    sample = {}
    for par in params:
        for parname, range in priordict.items():
            if parname == par or re.match(parname, par):
                if par.endswith(")"):
                    sample[par] = (
                        np.random.uniform(*range, size=int(par[par.index("(") + 1 : -1]))
                        if n == 1
                        else np.random.uniform(*range, size=(n, int(par[par.index("(") + 1 : -1])))
                    )
                else:
                    sample[par] = np.random.uniform(*range) if n == 1 else np.random.uniform(*range, size=n)
                break
        else:
            if fail:
                raise KeyError(f"No known prior for {par}.")

    return sample


# ----------------------------------------------------------------------
# Gaussian-prior extensions
#
# A truncated-Gaussian prior is layered on top of the existing
# uniform tanh transform. For any parameter listed in `gaussdict`,
# the prior on xs (the in-range value) becomes:
#     pi(xs) propto Normal(xs; mu, sigma)  for xs in (a, b)
# The tanh transform xs = 0.5*(b+a + (b-a)*tanh(ys)) already
# constrains xs to (a, b), so the Gaussian factor is automatically
# truncated to that interval (the truncation constant is dropped
# because it is independent of ys and irrelevant for MCMC).
#
# `gaussdict` maps a parameter name (exact) OR a regex pattern to
# a (mu, sigma) tuple. Exact-name keys win over regex patterns.
# Parameters not in `gaussdict` keep their uniform prior.
# ----------------------------------------------------------------------

priordict_gauss_standard = {}

# Opt-in defaults for per-pulsar intrinsic GP power-law params.
# Only includes patterns whose uniform range is already in priordict_standard,
# so the truncated-Gaussian transform lookup will not raise. Pass explicitly:
#     makelogtransform_uniform_gauss(func, gaussdict=prior.priordict_gauss_intrinsic), just test updated later
priordict_gauss_intrinsic = {
    "(.*_)?red_noise_log10_A": (-14.0, 3.0),
    "(.*_)?red_noise_gamma":   (5.0,   0.05),
    "(.*_)?dm_gp_log10_A":     (-14.5, 1.5),
    "(.*_)?dm_gp_gamma":       (3.5,   1.5),
    "(.*_)?chrom_gp_log10_A":  (-14.5, 1.5),
    "(.*_)?chrom_gp_gamma":    (3.5,   1.5),
}


def getprior_gauss(par, gaussdict):
    if not gaussdict:
        return None
    if par in gaussdict:
        return gaussdict[par]
    for pname, mu_sigma in gaussdict.items():
        if re.match(pname, par):
            return mu_sigma
    return None


def makelogtransform_uniform_gauss(func, priordict={}, gaussdict={}):
    priordict = {**priordict_standard, **priordict}
    gaussdict = {**priordict_gauss_standard, **gaussdict}

    slices, offset = [], 0
    for par in func.params:
        if '(' in par:
            l = int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1
            slices.append(slice(offset, offset+l))
            offset = offset + l
        else:
            slices.append(offset)
            offset = offset + 1

    a, b = [], []
    mu, sigma, gauss_mask = [], [], []
    columns = []
    matched_gauss = []
    for par, slice_ in zip(func.params, slices):
        for pname, prange in priordict.items():
            if re.match(pname, par):
                therange = prange
                break
        else:
            raise KeyError(f"No known prior for {par}.")

        gp_entry = getprior_gauss(par, gaussdict)
        if gp_entry is not None:
            matched_gauss.append(par)

        if '(' in par:
            root = par[:par.index('(')]
            l = int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1

            for i in range(l):
                columns.append(f'{root}[{i}]')
                a.append(therange[0])
                b.append(therange[1])
                if gp_entry is not None:
                    mu.append(gp_entry[0]); sigma.append(gp_entry[1]); gauss_mask.append(1.0)
                else:
                    mu.append(0.0); sigma.append(1.0); gauss_mask.append(0.0)
        else:
            columns.append(par)
            a.append(therange[0])
            b.append(therange[1])
            if gp_entry is not None:
                mu.append(gp_entry[0]); sigma.append(gp_entry[1]); gauss_mask.append(1.0)
            else:
                mu.append(0.0); sigma.append(1.0); gauss_mask.append(0.0)

    a, b = matrix.jnparray(a), matrix.jnparray(b)
    mu, sigma = matrix.jnparray(mu), matrix.jnparray(sigma)
    gauss_mask = matrix.jnparray(gauss_mask)

    def to_dict(ys):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        if len(a) != len(func.params):
            return {par: xs[slice_] for par, slice_ in zip(func.params, slices)}
        else:
            return dict(zip(func.params, xs))

    def to_vec(params):
        xs = jnp.zeros_like(a)
        for par, slice_ in zip(func.params, slices):
            xs = xs.at[slice_].set(params[par])
        return jnp.arctanh((a + b - 2*xs)/(a - b))

    def to_df(ys, psrs=None):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        if psrs is None:
            return pd.DataFrame(np.array(xs), columns=columns)
        else:
            psrdict = {f'{i}]': psr.name for i, psr in enumerate(psrs)}
            psrcols = [psrdict[par.split('[')[1]] + '_' + par.split('[')[0] if '[' in par else par for par in columns]
            return pd.DataFrame(np.array(xs), columns=psrcols).sort_index(axis=1)

    def logprior(ys):
        log_jac = jnp.sum(jnp.log(2.0) - 2.0 * jnp.logaddexp(ys, -ys))
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        gauss_logpdf = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) - 0.5 * ((xs - mu) / sigma) ** 2
        log_gauss = jnp.sum(gauss_mask * gauss_logpdf)
        return log_jac + log_gauss

    def logL(ys):
        return func(to_dict(ys))

    def transformed(ys):
        return logL(ys) + logprior(ys)

    transformed.params = func.params
    transformed.logprior = logprior
    transformed.logL = logL
    transformed.to_dict = to_dict
    transformed.to_vec = to_vec
    transformed.to_df = to_df
    transformed.gauss_params = matched_gauss

    return transformed


def sample_gauss(params, gaussdict, priordict={}, n=1, fail=True):
    priordict = {**priordict_standard, **priordict}
    gaussdict = {**priordict_gauss_standard, **gaussdict}

    sample = {}
    for par in params:
        gp = getprior_gauss(par, gaussdict)
        if gp is not None:
            mu_, sigma_ = gp
            for pname, prange in priordict.items():
                if pname == par or re.match(pname, par):
                    a_, b_ = prange
                    break
            else:
                raise KeyError(f"No known uniform range for {par} (needed for Gaussian truncation).")

            size = n if not par.endswith(")") else (n, int(par[par.index("(") + 1 : -1]))
            raw = np.random.normal(loc=mu_, scale=sigma_, size=size)
            sample[par] = np.clip(raw, a_, b_)
        else:
            for pname, prange in priordict.items():
                if pname == par or re.match(pname, par):
                    if par.endswith(")"):
                        sample[par] = (
                            np.random.uniform(*prange, size=int(par[par.index("(") + 1 : -1]))
                            if n == 1
                            else np.random.uniform(*prange, size=(n, int(par[par.index("(") + 1 : -1])))
                        )
                    else:
                        sample[par] = np.random.uniform(*prange) if n == 1 else np.random.uniform(*prange, size=n)
                    break
            else:
                if fail:
                    raise KeyError(f"No known prior for {par}.")

    return sample