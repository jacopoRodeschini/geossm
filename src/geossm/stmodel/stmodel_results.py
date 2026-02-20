from typing import Any, Optional
import numpy as np
from geossm.stmodel.param import ModelParams, FitOptions
import jax.numpy as jnp
from statsmodels.stats.stattools import jarque_bera, durbin_watson, omni_normtest
from statsmodels.iolib.summary import Summary

from geossm.ssm import StateSpaceResults
from types import SimpleNamespace


from scipy.stats import t, f, norm

import numpy as np
from scipy import stats
from dataclasses import dataclass

ArrayLike = Optional[Any]


class LRStateSpaceResults(StateSpaceResults):
    """
    Results container for LR State Space estimation.
    """

    def __init__(self, model=None, params:ModelParams=None, nstats:list=None, options:FitOptions=None, **kwargs):
        # Initialize base class
        super().__init__(model=model, **kwargs)

        # overwrite the params and nstats with the ones provided in the constructor
        # ---- Raw inputs ----
        self.params = params
        self.param_names = None  # will be processed from params
        self.param_dim = None  # will be processed from params
        self.param_names = None # will be processed from params

        self.nstats = nstats
        self.options = options

        # ---- Derived quantities (initialized empty) ----
        self.param_names = None
        self.param_values = None
        self.param_dim = None

        self.iterations = 0
        self.runtime_tot = 0.0
        self.runtime_tot_estep = 0.0
        self.runtime_tot_mstep = 0.0

        self.llf_path = None  # log-likelihood across EM iterations

        # Inference (see attributes and methods below)
        self._cov_params = None
        self._tvalues = None
        self._pvalues = None
        self._aic = None
        self._bic = None
        self._n_params = None
        self._hessian = None

        

        self._p_block = None
        self._q_block = None

        # ---- Process inputs explicitly ----
        if self.params is not None:
            self._process_params()

        if self.nstats is not None:
            self._process_nstats()
    

    def _process_params(self):
        """
        Extract parameter names, values, and dimensions from ModelParams dataclass.
        """

        param_fields = self.params.__dataclass_fields__

        names = []
        values = []
        dims = []

        for field in param_fields:
            obj = getattr(self.params, field)
            names.append(obj.name)
            
            # name is x0 of Sigma0, get the average
            if obj.name in ['x0']:
                values.append(np.mean(obj.value))
                temp_dim = 1
            
            elif obj.name in ['Sigma0']:
                values.append(np.mean(np.diag(obj.value)))
                temp_dim = 1
            
            else:
                values.append(obj.value.flatten())
                temp_dim =obj.value.flatten().size
            
            dims.append(temp_dim)

        self.param_names = names
        self.param_values = values
        self.param_dim = dims
        self.n_params = sum(dims)

    def _process_nstats(self):
        """
        Extract iteration statistics from EM output.
        """

        if not self.nstats:
            return

        self.iterations = self.nstats[-1]["niter"]

        runtime_each = [v["time_tot"] for v in self.nstats]
        runtime_estep = [v["tdelta_E"] for v in self.nstats]
        runtime_mstep = [v["tdelta_M"] for v in self.nstats]

        self.runtime_tot = sum(runtime_each)
        self.runtime_tot_estep = sum(runtime_estep)
        self.runtime_tot_mstep = sum(runtime_mstep)

        self.llf_path = [v["logL"] for v in self.nstats]
        self.llf = self.llf_path[-1]



    @property
    def bse(self):
        se = np.sqrt(np.diag(self.cov_params))
        return se

    @property
    def tvalues(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.params / self.bse

    @property
    def pvalues(self):
        tv = self.tvalues
        if self.df_resid is not None:
            pv = 2 * stats.t.sf(np.abs(tv), df=self.df_resid)
        else:
            pv = 2 * stats.norm.sf(np.abs(tv))
        return pv

    def conf_int(self, alpha=0.05):
        z = stats.norm.ppf(1 - alpha / 2.0)
        se = self.bse
        lower = self.params - z * se
        upper = self.params + z * se
        return np.vstack([lower, upper]).T

    @property
    def aic(self):
        return self.compute_aic()

    @property
    def bic(self):
        return self.compute_bic()
    
        # Compute AIC and BIC
    def compute_aic(self):
        """Computes the AIC (Akaike Information Criterion)."""
        llf = self.llf
        
        # number of estimated parameters: try to infer from model
        k = getattr(self, 'n_params', None)
        if k is None and hasattr(self.model, 'beta'):
            k = np.size(np.array(self.model.beta))
        k = int(k) if k is not None else 0
        self.aic = 2 * k - 2 * llf
        return self.aic
    
    def compute_bic(self):
        """Computes the BIC (Bayesian Information Criterion)."""
        llf = self.llf
        
        # number of estimated parameters: try to infer from model
        k = getattr(self, 'n_params', None)
        if k is None and hasattr(self.model, 'beta'):
            k = np.size(np.array(self.model.beta))
        k = int(k) if k is not None else 0
        
        n = self.nobs if self.nobs is not None else 1
        self.bic = np.log(n) * k - 2 * llf
        return self.bic
    

    def summary(self):
        
        # self.results = np.array([0])
        # self.params = self.beta
        # self.param_names = self.xbeta_names
        # self.bse = np.zeros(len(self.beta))
        # self.tvalues = np.zeros(len(self.beta))
        # self.pvalues = np.zeros(len(self.beta))

        smry = super().summary()
        # Add parameter estimates table

        # todo: Add the parameters table (measurement equation parameters)
        self.measurement = [SimpleNamespace() for _ in range(self.nvar)]  # Create a list with one SimpleNamespace for compatibility with summary structure
        
        for i, m in enumerate(self.measurement):
            m.results = np.array([0])  # Dummy results for compatibility

            # Get the parameter values for this variable and assign them to the model namespace
            m.params_beta = self.beta[i]
            m.error_params = self.s2e[i]
            m.linear_params = self.A[:, i]

            # Get the parameter names for this variable and assign them to the model namespace
            m.beta_names = self.xbeta_names[i]
            m.error_name = []
            m.linear_names = [f"a_({i},{j+1})" for j in range(self.nlat)]
           
            # Combine all parameters and names into a single list for the summary       
            m.params = np.concatenate([m.params_beta, m.linear_params, m.error_params])
            m.param_names = m.beta_names + m.linear_names + m.error_name
        
            m.bse = np.full(len(m.params), np.nan)  # Set standard errors to NaN (not available)
            m.tvalues = np.full(len(m.params), np.nan)  # Set t-values to NaN (not available)
            m.pvalues = np.full(len(m.params), np.nan)  # Set p-values to NaN (not available)

        
        # todo: add the parameters table (state equation parameters)
        self.state = [SimpleNamespace() for _ in range(self.nlat)]  # Create a list with one SimpleNamespace for compatibility with summary structure
        
        for i, m in enumerate(self.state):
            m.results = np.array([0])  # Dummy results for compatibility

            # Get the parameter values for this variable and assign them to the model namespace
            m.matern_params = [cov.rescale for cov in self.cov_function]
            m.markov_params = [self.f]

            m.matern_names = []
            m.latent_names = []

            # Combine all parameters and names into a single list for the summary       
            m.params = np.concatenate([m.matern_params, m.markov_params])
            m.param_names = m.matern_names + m.latent_names
        
            m.bse = np.full(len(m.params), np.nan)  # Set standard errors to NaN (not available)
            m.tvalues = np.full(len(m.params), np.nan)  # Set t-values to NaN (not available)
            m.pvalues = np.full(len(m.params), np.nan)  # Set p-values to NaN (not available)



       
        return smry
    
    
    
    
