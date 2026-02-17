import numpy as np
import jax.numpy as jnp
from statsmodels.stats.stattools import jarque_bera, durbin_watson, omni_normtest
from statsmodels.iolib.summary import Summary

from .statespace_results import StateSpaceResults


from scipy.stats import t, f, norm

from __future__ import annotations
import numpy as np
from scipy import stats
from dataclasses import dataclass


# %% Class definition
@dataclass
class stModelResults(StateSpaceResults):

    params = None
    cov_params = None
    tvalues = None
    pvalues = None
    aic = None
    bic = None


    # Convert all quantity to numpy array 
    def __post_init__(self):
        self = self.to_numpy()


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
    
    
    
