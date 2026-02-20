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


# %% Class definition
@dataclass
class LRStateSpaceResults(StateSpaceResults):

    # In a dataclass, fields must be declared using type annotations
    params : ModelParams # parameter estimates ModelParams object
    nstats: list # EM iterations until convergence
    options: FitOptions  # options used for the estimation (for summary and other methods)
    
    # formulas : list = None
    # cov_function: list = None    # Covariance function for the innovations
    # points : list[np.ndarray] = None          # points used for training (for prediction)
    
    # Attribute of the results class
    params_names = None     # Updated in __setattr__ when params is set (array-like)
    params_dim = None       # Updated in __setattr__ when params is set (array-like)                    
    cov_params = None       # Updated in __setattr__ when cov_function is set (array-like)
    
    
    tvalues = None
    pvalues = None
    aic = None
    bic = None
    n_params = None # Number of parameters in the model (for AIC/BIC)

    p_block = None # indecis for the multivariate case (measurement equation) 
    q_block = None # indecis for the multivariate case (state equation)

    # Add attibute on inference results
    hessian = None
    llf = None
    

    def __setattr__(self, name, value):
        if name == 'params': 
            # Model parameters (ModelParams object) - extract the parameter names and dimensions for summary and other methods
            
            self.param_name = [getattr(value, par).name for par in value.__dataclass_fields__]
            self.param_value = [getattr(value, par).value for par in value.__dataclass_fields__]
            self.param_dim = [v.size if hasattr(v, 'size') else 1 for v in self.param_value]
            
        # em iterations statistics
        if name == 'nstats':
            self.iterations = value[-1]['niter'] if value is not None and len(value) > 0 else 0
            self.runtime_tot_each = [v['time_tot'] for v in value] 
            self.runtime_tot_estep = [v['tdelta_E'] for v in value]
            self.runtime_tot_mstep = [v['tdelta_M'] for v in value]
            self.runtime_tot = sum(self.runtime_tot_each)
            self.runtime_tot_estep = sum(self.runtime_tot_estep)
            self.runtime_tot_mstep = sum(self.runtime_tot_mstep) 

            self.llf = [v['logL'] for v in value]

        super().__setattr__(name, value)

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
    

    def summary(self):
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
    
    
    
