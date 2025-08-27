import inspect

import pandas as pd

import numpyro
from numpyro import infer
from numpyro import distributions as dist

from .. import prior


def makemodel_transformed(mylogl, transform=prior.makelogtransform_uniform, priordict={}):
    logx = transform(mylogl, priordict=priordict)

    parlen = sum(int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1 for par in logx.params)

    def numpyro_model():
        pars = numpyro.sample('pars', dist.Normal(0, 10).expand([parlen]))
        logl = logx(pars)

        numpyro.factor('logl', logl)
    numpyro_model.to_df = lambda chain: logx.to_df(chain['pars'])

    return numpyro_model


def makemodel(mylogl, priordict={}):
    def numpyro_model():
        logl = mylogl({par: numpyro.sample(par, dist.Uniform(*prior.getprior_uniform(par, priordict)))
                       for par in mylogl.params})

        numpyro.factor('logl', logl)
    numpyro_model.to_df = lambda chain: pd.DataFrame(chain)

    return numpyro_model


def makesampler_nuts(numpyro_model, num_warmup=512, num_samples=1024, num_chains=1, **kwargs):
    nutsargs = dict(max_tree_depth=8, dense_mass=False,
                    forward_mode_differentiation=False, target_accept_prob=0.8,
                    **{arg: val for arg in kwargs.items() if arg in inspect.getfullargspec(infer.NUTS).args})

    mcmcargs = dict(num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                    chain_method='vectorized', progress_bar=True,
                    **{arg: val for arg in kwargs.items() if arg in inspect.getfullargspec(infer.MCMC).kwonlyargs})

    mcmc = infer.MCMC(infer.NUTS(numpyro_model, **nutsargs), **mcmcargs)

    class Sampler:
        def __init__(self, mcmc, model):
            self.mcmc = mcmc
            self.model = model
            self.samples = None

        def run(self, rng_key):
            self.mcmc.run(rng_key)
            self.samples = self.mcmc.get_samples()

        def make_plots(self, save_name=None, diagnostics=False):
            import matplotlib.pyplot as plt
            import corner
            if self.samples is None:
                raise RuntimeError("Run the sampler before making plots.")

            df = self.to_df()
            labels = list(df.columns)
            samples_array = df[labels].values

            fig = corner.corner(
                samples_array,
                labels=labels,
                show_titles=True,
                title_fmt=".2f",
                title_kwargs={"fontsize": 10},
                label_kwargs={"fontsize": 9},
                plot_datapoints=True,
                hist_kwargs={"color": "C0"},
                contour_kwargs={"colors": ["C0"]}
            )

            plt.tight_layout()
            if save_name:
                plt.savefig(save_name + "_corner.png")
            plt.close()

        def to_df(self):
            import numpy as np
            if self.samples is None:
                raise RuntimeError("Run the sampler before accessing results.")

            data = {}
            for k, v in self.samples.items():
                v_np = np.array(v)
                if v_np.ndim == 1:
                    data[k] = v_np
                else:
                    for j in range(v_np.shape[1]):
                        data[f"{k}[{j}]"] = v_np[:, j]
            return pd.DataFrame(data)

    return Sampler(mcmc, numpyro_model)

