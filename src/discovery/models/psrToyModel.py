import numpy as np
import jax.numpy as jnp
import discovery.const as const
import re

from .. import matrix
from .. import signals
from .. import prior
from .. import solar
from .. import likelihood
from .. import deterministic

def write_ml_json(df, savename):
    import json
    ml_idx = df['logl'].idxmax()
    ml_params = df.loc[ml_idx].to_dict()
    with open(savename, 'w') as f:
        json.dump(ml_params, f, indent=2)
    return

def update_priordict_standard_mpta():
    # Update the standard prior dictionary with PTA-specific parameters
    prior.priordict_standard.update({
        # White noise parameters
        '(.*_)?efac':               [0.5, 2],
        '(.*_)?log10_tnequad':      [-10, -5],
        '(.*_)?log10_ecorr':        [-10, -5],
        # Per-pulsar GW background parameters
        '(.*_)?bkgrnd_log10_A':     [-18, -11],
        # GP parameters
        '(.*_)?red_noise_log10_A.*':  [-18, -11],
        '(.*_)?red_noise_gamma.*':    [0, 7],
        '(.*_)?red_noise2_log10_A.*':  [-18, -11],
        '(.*_)?red_noise2_gamma.*':    [0, 7],
        '(.*_)?dm_gp_log10_A':      [-18, -11],
        '(.*_)?dm_gp_gamma':        [0, 7],
        '(.*_)?chrom_gp_log10_A':   [-18, -11],
        '(.*_)?chrom_gp_gamma':     [0, 7],
        '(.*_)?chrom_gp_alpha':     [2.5, 14],
        '(.*_)?sw_gp_log10_A':      [-10, -2],
        '(.*_)?sw_gp_gamma':        [0, 4],
        '(.*_)?band_gp_log10_A':    [-18, -11],
        '(.*_)?band_gp_gamma':      [0, 7],
        '(.*_)?band_low_gp_fcutoff':    [856, 1712], # MeerKAT L-band
        '(.*_)?band_gp_flow':       [856, 1712], # MeerKAT L-band
        '(.*_)?band_gp_fhigh':      [856, 1712], # MeerKAT L-band
        '(.*_)?bandalpha_gp_log10_A':    [-18, -11],
        '(.*_)?bandalpha_gp_gamma':      [0, 7],
        '(.*_)?bandalpha_gp_alpha':      [0, 10],
        '(.*_)?bandalpha_gp_fcutoff':    [856, 1712], # MeerKAT L-band
        '(.*_)?bandalpha_gp_fhigh':    [856, 1712], # MeerKAT L-band
        '(.*_)?bandalpha_gp_flow':    [856, 1712], # MeerKAT L-band
        # common noise parameters
        'curn_log10_A':             [-18, -11],
        'curn_gamma':               [0, 7],
        # deterministic parameters
        '(.*_)?chrom_exp_t0': [58525, 60700], # MPTA 6-yr range
        '(.*_)?chrom_exp_log10_Amp': [-10, -4],
        '(.*_)?chrom_exp_log10_tau': [0, 4],
        '(.*_)?chrom_exp_sign_param': [-1, 1],
        '(.*_)?chrom_exp_alpha': [0, 7],
        '(.*_)?chrom_1yr_log10_Amp': [-10, -4],
        '(.*_)?chrom_1yr_phase': [0, 2 * np.pi],
        '(.*_)?chrom_1yr_alpha': [0, 7],
        '(.*_)?chrom_gauss_t0': [58525, 60700], # MPTA 6-yr range
        '(.*_)?chrom_gauss_log10_Amp': [-10, -4],
        '(.*_)?chrom_gauss_log10_sigma': [0, 4],
        '(.*_)?chrom_gauss_sign_param': [-1, 1],
        '(.*_)?chrom_gauss_alpha': [0, 7],
        r'(.*_)?timingmodel_coefficients\(\d+\)': [-20.0, 20.0],
        r'(.*_)?dm_sw_log10_rho\(\d+\)': [-10, 4],
        r'(.*_)?alpha_scaling\(\d+\)': [0.0, 100.0],
    })
    return

update_priordict_standard_mpta() # Ensure priordict_standard is updated on import, but also update when a model is created to catch any changes during likelihood/prior initialisation

def gps2commongp(gps):
    priors = [gp.Phi.getN for gp in gps]
    pmax = len(gps)
    ns = [gp.F.shape[1] for gp in gps]  # Does not work for callable gp.F (e.g. chromatic GP)
    nmax = max(ns)

    def prior(params):
        yp = matrix.jnp.full((pmax, nmax), 1e-40)
        for i,p in enumerate(priors):
            yp = yp.at[i, :ns[i]].set(p(params))

        return yp

    prior.params = sorted(set([par for p in priors for par in p.params]))
    Fs = [np.pad(gp.F, [(0,0), (0,nmax - gp.F.shape[1])]) for gp in gps]

    return matrix.VariableGP(matrix.VectorNoiseMatrix1D_var(prior), Fs)


def make_psr_gps_fourier(psr, max_cadence_days=14, Tspan=None, GlobalTspan = None,background=True, bkgrnd_fixed=False, bkgrnd_fixed_log10A=jnp.log10(2e-15), bkgrnd_fixed_gamma=13/3, curn=False, red=True, dm=True, chrom=True, sw=True, dm_sw_free=False, band=False, band_low=False, band_alpha=False):
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))

    psr_GlobalTspan = signals.getspan(psr) if GlobalTspan is None else GlobalTspan
    psr_Globalcomponents = int(psr_GlobalTspan / (max_cadence_days * 86400))

    def powerlaw_bkgrnd_fixed(f, df): # fixed amplitude and slope for a GWB

        A = 10**bkgrnd_fixed_log10A
        return (A**2) / 12.0 / jnp.pi**2 * const.fyr ** (bkgrnd_fixed_gamma - 3.0) * f ** (-bkgrnd_fixed_gamma) * df

    return (([signals.makegp_fourier(psr, signals.powerlaw_fixgam, components=psr_components, name='bkgrnd')] if background and not bkgrnd_fixed and not curn else []) + \
            ([signals.makegp_fourier(psr, powerlaw_bkgrnd_fixed, components=psr_components, name='bkgrnd_fixed')] if background and bkgrnd_fixed and not curn else []) + \
            #set up common process
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_Globalcomponents, T=psr_GlobalTspan, common=['curn_log10_A', 'curn_gamma'], name='curn')] if background and curn and not bkgrnd_fixed else []) + \
            #single pulsar noise processes
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, name='red_noise')] if red else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_dm, name='dm_gp')] if dm else [])+ \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_chrom, name='chrom_gp')] if chrom else [])+ \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=solar.fourierbasis_solar_dm, name='sw_gp')] if sw else []) + \
            ([signals.makegp_fourier(psr, signals.freespectrum, components=10, T=365.25*86400, fourierbasis=signals.fourierbasis_dm, name='dm_sw')] if dm_sw_free else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_band_range, name='band_gp')] if band else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_band, name='band_low_gp')] if band_low else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_band_range_alpha, name='bandalpha_gp')] if band_alpha else []))


def make_psr_gps_fftint(psr, max_cadence_days=14, Tspan=None, GlobalTspan = None , background=True, bkgrnd_fixed=False, bkgrnd_fixed_log10A=jnp.log10(2e-15), bkgrnd_fixed_gamma=13/3, curn=False, red=True, dm=True, chrom=True, sw=True, dm_sw_free=False, band=False, band_low=False, band_alpha=False):
    psr_Tspan = signals.getspan(psr) if Tspan is None else Tspan
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    psr_knots = 2 * psr_components + 1

    psr_GlobalTspan = signals.getspan(psr) if GlobalTspan is None else GlobalTspan
    psr_Globalcomponents = int(psr_GlobalTspan / (max_cadence_days * 86400))
    psr_Globalknots = 2 * psr_Globalcomponents + 1



    def powerlaw_bkgrnd_fixed(f, df): #log10_A=-14, gamma=13/3): # fixed amplitude and slope for a GWB 
        A = 10**bkgrnd_fixed_log10A
        return (A**2) / 12.0 / jnp.pi**2 * const.fyr ** (bkgrnd_fixed_gamma - 3.0) * f ** (-bkgrnd_fixed_gamma) * df

    return (([signals.makegp_fftcov(psr, signals.powerlaw_fixgam, components=psr_knots, name='bkgrnd')] if background  and not bkgrnd_fixed and not curn else []) + \
            ([signals.makegp_fftcov(psr, powerlaw_bkgrnd_fixed, components=psr_knots, name='bkgrnd_fixed')] if background and bkgrnd_fixed and not curn else []) + \
            #set up common process
            ([signals.makegp_fftcov(psr, signals.powerlaw, components=psr_Globalknots, T=psr_GlobalTspan, common=['curn_log10_A', 'curn_gamma'], name='curn')] if background and curn and not bkgrnd_fixed else []) + \
            #single pulsar noise processes
            ([signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, name='red_noise')] if red else []) + \
            ([signals.makegp_fftcov_dm(psr, signals.powerlaw, components=psr_knots, name='dm_gp')] if dm else [])+ \
            ([signals.makegp_fftcov_chrom(psr, signals.powerlaw, components=psr_knots, name='chrom_gp')] if chrom else [])+ \
            ([signals.makegp_fftcov_solar(psr, signals.powerlaw, components=psr_knots, name='sw_gp')] if sw else []) + \
            ([signals.makegp_fftcov_dm(psr, signals.freespectrum, components=21, T=365.25*86400, name='dm_sw')] if dm_sw_free else []) + \
            ([signals.makegp_fftcov_band_range(psr, signals.powerlaw, components=psr_knots, name='band_gp')] if band else []) + \
            ([signals.makegp_fftcov_band(psr, signals.powerlaw, components=psr_knots, name='band_low_gp')] if band_low else []) + \
            ([signals.makegp_fftcov_band_range_alpha(psr, signals.powerlaw, components=psr_knots, name='bandalpha_gp')] if band_alpha else []))


def single_pulsar_noise(psr, fftint=True, max_cadence_days=14, Tspan=None, GlobalTspan = None , noisedict={}, tm_variable=False, timing_inds=None, outliers=False, tnequad=True, ecorr=True, global_ecorr=False,
                        background=False, bkgrnd_fixed=False, bkgrnd_fixed_log10A=jnp.log10(2e-15), bkgrnd_fixed_gamma=13/3, curn=False, red=True, dm=True, chrom=True, sw=True, dm_sw_free=False, band=False, band_low=False, band_alpha=False, # GP models
                        chrom_annual=False, chrom_exponential=False, chrom_gaussian=False): # Deterministic chromatic models
    # Set up per-backend white noise
    measurement_noise = signals.makenoise_measurement(psr, tnequad=tnequad, noisedict=noisedict, outliers=outliers) 
    # Set up timing model
    tm = signals.makegp_timing(psr, svd=True, variable=tm_variable, timing_inds=timing_inds)
    if not isinstance(tm, list): # ensure the timing model is unpacked if returning a list
        tm = [tm]
    # Set up model components
    model_components = [psr.residuals]
    model_components += tm
    model_components += [measurement_noise]

    if ecorr: #add eccorr term
        model_components += [signals.makegp_ecorr(psr, noisedict=noisedict)]
    if global_ecorr: # add an additional global ECORR term
        model_components += [signals.makegp_ecorr_simple(psr, noisedict=noisedict)]
    # Add deterministic chromatic components
    if chrom_annual:
        model_components += [signals.makedelay(psr, deterministic.chromatic_annual(psr), name='chrom_1yr')]
    if chrom_exponential:
        model_components += [signals.makedelay(psr, deterministic.chromatic_exponential(psr), name='chrom_exp')]
    if chrom_gaussian:
        model_components += [signals.makedelay(psr, deterministic.chromatic_gaussian(psr), name='chrom_gauss')]
    # Add GP components
    if fftint:
        model_components += make_psr_gps_fftint(psr, max_cadence_days=max_cadence_days,Tspan=Tspan, GlobalTspan=GlobalTspan, background=background, bkgrnd_fixed=bkgrnd_fixed, bkgrnd_fixed_log10A=bkgrnd_fixed_log10A, bkgrnd_fixed_gamma=bkgrnd_fixed_gamma, curn=curn, red=red, dm=dm, chrom=chrom, sw=sw, dm_sw_free=dm_sw_free, band=band, band_low=band_low, band_alpha=band_alpha)
    else:
        model_components += make_psr_gps_fourier(psr, max_cadence_days=max_cadence_days, Tspan=Tspan, GlobalTspan=GlobalTspan, background=background, bkgrnd_fixed=bkgrnd_fixed, bkgrnd_fixed_log10A=bkgrnd_fixed_log10A, bkgrnd_fixed_gamma=bkgrnd_fixed_gamma, curn=curn, red=red, dm=dm, chrom=chrom, sw=sw, dm_sw_free=dm_sw_free, band=band, band_low=band_low, band_alpha=band_alpha)

    comp_params = []
    for comp in model_components:
        if hasattr(comp, 'params'):
            comp_params.extend(comp.params)

    m = likelihood.PulsarLikelihood(model_components)
    m.all_params.extend(comp_params)
    m.logL.params = sorted(set(m.all_params))

    return m



#single pulsar noise make noise measuramnt simple:
def single_pulsar_noise_simple(psr, fftint=True, max_cadence_days=14, Tspan=None, GlobalTspan = None , noisedict={}, tm_variable=False, timing_inds=None, add_equad=False, tnequad=False, ecorr=False, global_ecorr=False,
                        background=True, bkgrnd_fixed=True, bkgrnd_fixed_log10A= jnp.log10(5e-15) , bkgrnd_fixed_gamma=13/3, curn=False, red=True, dm=False, chrom=False, sw=False, dm_sw_free=False, band=False, band_low=False, band_alpha=False, # GP models
                        chrom_annual=False, chrom_exponential=False, chrom_gaussian=False): # Deterministic chromatic models
    # Set up per-backend white noise
    measurement_noise_simple = signals.makenoise_measurement_simple(psr,  noisedict=noisedict ,add_equad=add_equad, tnequad=tnequad) 
    # Set up timing model
    tm = signals.makegp_timing(psr, svd=True, variable=tm_variable, timing_inds=timing_inds)
    if not isinstance(tm, list): # ensure the timing model is unpacked if returning a list
        tm = [tm]
    # Set up model components
    model_components = [psr.residuals]
    model_components += tm
    model_components += [measurement_noise_simple]

    if ecorr: #add eccorr term
        model_components += [signals.makegp_ecorr(psr, noisedict=noisedict)]
    if global_ecorr: # add an additional global ECORR term
        model_components += [signals.makegp_ecorr_simple(psr, noisedict=noisedict)]
    # Add deterministic chromatic components
    if chrom_annual:
        model_components += [signals.makedelay(psr, deterministic.chromatic_annual(psr), name='chrom_1yr')]
    if chrom_exponential:
        model_components += [signals.makedelay(psr, deterministic.chromatic_exponential(psr), name='chrom_exp')]
    if chrom_gaussian:
        model_components += [signals.makedelay(psr, deterministic.chromatic_gaussian(psr), name='chrom_gauss')]
    # Add GP components
    if fftint:
        model_components += make_psr_gps_fftint(psr, max_cadence_days=max_cadence_days,Tspan=Tspan, GlobalTspan=GlobalTspan, background=background, bkgrnd_fixed=bkgrnd_fixed, bkgrnd_fixed_log10A=bkgrnd_fixed_log10A, bkgrnd_fixed_gamma=bkgrnd_fixed_gamma, curn=curn, red=red, dm=dm, chrom=chrom, sw=sw, dm_sw_free=dm_sw_free, band=band, band_low=band_low, band_alpha=band_alpha)
    else:
        model_components += make_psr_gps_fourier(psr, max_cadence_days=max_cadence_days, Tspan=Tspan, GlobalTspan=GlobalTspan, background=background, bkgrnd_fixed=bkgrnd_fixed, bkgrnd_fixed_log10A=bkgrnd_fixed_log10A, bkgrnd_fixed_gamma=bkgrnd_fixed_gamma, curn=curn, red=red, dm=dm, chrom=chrom, sw=sw, dm_sw_free=dm_sw_free, band=band, band_low=band_low, band_alpha=band_alpha)

    comp_params = []
    for comp in model_components:
        if hasattr(comp, 'params'):
            comp_params.extend(comp.params)

    m = likelihood.PulsarLikelihood(model_components)
    m.all_params.extend(comp_params)
    m.logL.params = sorted(set(m.all_params))

    return m


def GWB_simple_search_common(psrs, GlobalTspan=None, fftInt=False, max_cadence_days=14,name="curn"):
        
    GlobalTspan = signals.getspan(psrs) if GlobalTspan is None else GlobalTspan
    

    gbl = likelihood.GlobalLikelihood([single_pulsar_noise_simple(psr, fftint=fftInt, max_cadence_days=max_cadence_days, GlobalTspan=GlobalTspan, background=True, bkgrnd_fixed=False, curn=True, noisedict={f"{psr.name}_efac": 1.0}, global_ecorr=False, 
                        red=True, dm=False, chrom=False, sw=False, band=False, band_low=False, band_alpha=False) for psr in psrs])
    
    return gbl




def common_noise(psrs, chain_dfs, fftInt=False, max_cadence_days=14, name="gw_crn"):
    # Accepts a list of pulsars and their corresponding chain dataframes and constructs an ArrayLikelihood
    def has_param(df, param_string="red_noise"):
        return any(f"{param_string}" in col for col in list(df.columns))

    Tspan = signals.getspan(psrs)
    common_components = int(Tspan / (max_cadence_days * 86400))
    common_knots = 2 * common_components + 1

    psls = []

    for psr, df in zip(psrs, chain_dfs):
        if not any(psr.name in col for col in df.columns):
            raise ValueError("Chain data frames do not match pulsar names")
        # Get max-likelihood parameters for this pulsar
        ml_idx = df['logl'].idxmax()
        noisedict = {col: df.loc[ml_idx, col] for col in df.columns if col.startswith(psr.name)}

        # background = False, as we are including a common red noise process
        #m = single_pulsar_noise(psr, fftint=fftInt, max_cadence_days=max_cadence_days, Tspan=None, background=False, noisedict=noisedict, global_ecorr=has_param(df, f"{psr.name}_ecorr"),
        #                        red=has_param(df, "red_noise"), dm=has_param(df, "dm_gp"), chrom=has_param(df, "chrom_gp"), sw=has_param(df, "sw_gp"),
        #                        band=has_param(df, "band_gp"), band_low=has_param(df, "band_low_gp"), band_alpha=has_param(df, "bandalpha_gp"),
        #                        dm_sw_free=has_param(df, "dm_sw"), chrom_annual=has_param(df, "chrom_1yr"), chrom_exponential=has_param(df, "chrom_exp"), chrom_gaussian=has_param(df, "chrom_gauss"))

        m = single_pulsar_noise(psr, fftint=fftInt, max_cadence_days=max_cadence_days, Tspan=Tspan, background=False, noisedict=noisedict, global_ecorr=False,
                                red=True, dm=True, chrom=True, sw=False, band=False, band_low=False, band_alpha=False,
                                dm_sw_free=False, chrom_annual=False, chrom_exponential=False, chrom_gaussian=False) # Simplified model for testing

        print("Including pulsar", psr.name, "with model parameters:\n", m.logL.params)
        psls.append(m)

    if not fftInt:
        curn = signals.makeglobalgp_fourier(psrs, signals.powerlaw, signals.uncorrelated_orf, common_components, Tspan, common=['curn_log10_A', 'curn_gamma'], name='curn')
        return likelihood.GlobalLikelihood(psls, globalgp=curn)
        # return likelihood.ArrayLikelihood(psls, commongp=curn)

    else:
        curn = signals.makeglobalgp_fftcov(psrs, signals.powerlaw, signals.uncorrelated_orf, common_knots, Tspan, common=['curn_log10_A', 'curn_gamma'], name='curn')
        return likelihood.GlobalLikelihood(psls, globalgp=curn)
        # return likelihood.ArrayLikelihood(psls, commongp=curn)