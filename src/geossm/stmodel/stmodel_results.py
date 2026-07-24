from statsmodels.iolib.summary import Summary
from typing import Any, Optional
import numpy as np
from geossm.stmodel.param import Param, ModelParams, FitOptions
from geossm.ssm import StateSpaceResults
from types import SimpleNamespace
from scipy import stats
import time
import jax
import jax.numpy as jnp

ArrayLike = Optional[Any]

# %% Results class for LR State Space Model


class LRStateSpaceResults(StateSpaceResults):
    """
    Results container for LR State Space estimation.
    """

    def __init__(
        self,
        model=None,
        params: ModelParams = None,
        nstats: list = None,
        options: FitOptions = None,
        **kwargs,
    ):
        # Initialize base class
        super().__init__(model=model, **kwargs)

        # overwrite the params and nstats with the ones provided in the constructor
        # ---- Raw inputs ----
        self.params = params
        self.param_names = None  # will be processed from params
        self.param_dim = None  # will be processed from params
        self.param_names = None  # will be processed from params

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
        self._tvalues = None
        self._pvalues = None
        self._aic = None
        self._bic = None
        self._n_params = None
        self._hessian = None
        self._cov_params = None

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
            if obj.name in ["x0"]:
                values.append(np.mean(obj.value))
                temp_dim = 1

            elif obj.name in ["Sigma0"]:
                values.append(np.mean(np.diag(obj.value)))
                temp_dim = 1

            else:
                values.append(obj.value.flatten())
                temp_dim = obj.value.flatten().size

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

    
    def _compute_hessian(self):
        """
        Compute the Hessian matrix of the log-likelihood function at the estimated parameters.
        """
        if self._hessian is not None:
            return self._hessian

        params = self.params.copy()  # Create a copy of the parameters to avoid modifying the original

        ts = time.time()

        # Compute the Hessian using JAX, of the observed log-likelihood function at the argument 0 (params)
        hesfun = jax.hessian(self.model._observed_logL)

        # Evaluate the Hessian at the estimated parameters
        # the Information matrix, which is the negative of the second derivative of the log-likelihood 
        # function
        hessian = hesfun(params)
        jax.block_until_ready(hessian)
        tdelta = time.time() - ts

        self._hessian = hessian
        return hessian, tdelta
        
    def compute_cov_params(self):
        """
        Compute the standard errors of the estimated parameters based on the Hessian matrix.
        """
        if self._cov_params is not None:
            return self._cov_params

        self.model._log("Computing Hessian and standard errors of the parameters", verbose=True)
        hessian, tdelta = self._compute_hessian()
        self.model._log(f"Hessian computed in {tdelta:.3f} seconds", verbose=True)

        # Compute the covariance matrix as the inverse of the Hessian
        cov_params = jnp.linalg.inv(hessian)

        # check if it is postive definte
        # chol = jnp.linalg.cholesky(cov_params)

        self._cov_params = cov_params
        return cov_params
    
    @property
    def bse(self):
        se = np.sqrt(np.diag(self.cov_params))
        return se

    @property
    def tvalues(self):
        with np.errstate(divide="ignore", invalid="ignore"):
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
        k = getattr(self, "n_params", None)
        if k is None and hasattr(self.model, "beta"):
            k = np.size(np.array(self.model.beta))
        k = int(k) if k is not None else 0
        self.aic = 2 * k - 2 * llf
        return self.aic

    def compute_bic(self):
        """Computes the BIC (Bayesian Information Criterion)."""
        llf = self.llf

        # number of estimated parameters: try to infer from model
        k = getattr(self, "n_params", None)
        if k is None and hasattr(self.model, "beta"):
            k = np.size(np.array(self.model.beta))
        k = int(k) if k is not None else 0

        n = self.nobs if self.nobs is not None else 1
        self.bic = np.log(n) * k - 2 * llf
        return self.bic

    def predict(self, df, verbose=True):
        """
        Compute predicted observations (y_hat) based on smoothed states and model parameters.
        """  
        points, y_hat, Sigma_y_hat, tdelta = self.model.predict(df, modelresults=self, verbose=verbose)

        return points, y_hat, Sigma_y_hat, tdelta
    

    def generate_summary(self):

        # update the computational time
        time_e = self.runtime_tot_estep
        time_m = self.runtime_tot_mstep

        # Generate the parent summary table (with model info)
        gen_top_left, gen_top_right = super().generate_summary()
        len_empty = len(gen_top_left)- len(gen_top_right)
        if len_empty > 0:
            gen_top_right = gen_top_right + [("", [""])] * len_empty
        elif len_empty < 0:
            gen_top_left = gen_top_left + [("", [""])] * (-len_empty)
        

        # Add the EM iteration statistics table
        top_left_em = dict(
            [
                ("EM iters :", lambda: [f"{self.iterations}"]),
                ("Runtime total (s):", lambda: [f"{self.runtime_tot:.3g}"]),
            ]
        )

        top_right_em = {
            "Runtime E-step (s):": lambda: [f"{time_e:.3g}"],
            "Runtime M-step (s):": lambda: [f"{time_m:.3g}"],
        }

        # Generate the dictionaly
        gen_top_left_em = []
        for item in top_left_em.keys():
            gen_top_left_em.append((item, list(top_left_em[item]())))

        gen_top_right_em = []
        for item in top_right_em.keys():
            gen_top_right_em.append((item, list(top_right_em[item]())))


        
        gen_top_left += gen_top_left_em
        gen_top_right += gen_top_right_em


        return gen_top_left, gen_top_right
    def summary(self):

        # self.results = np.array([0])
        # self.params = self.beta
        # self.param_names = self.xbeta_names
        # self.bse = np.zeros(len(self.beta))
        # self.tvalues = np.zeros(len(self.beta))
        # self.pvalues = np.zeros(len(self.beta))


        gen_top_left, gen_top_right = self.generate_summary()
        
        # Generate the summary
        smry = Summary()
        smry.add_table_2cols(
          
            self,
            title="LR State Space Model Results",
            gleft=gen_top_left,
            gright=gen_top_right,
            yname=None,
            xname=None,
        )

        # todo: Add the parameters table (measurement equation parameters)
        self.measurement = [
            SimpleNamespace() for _ in range(self.model.nvar)
        ]  # Create a list with one SimpleNamespace for compatibility with summary structure

        def conf_int_params(params, bse, alpha=0.05):
            lower = params - 1.96 * bse
            upper = params + 1.96 * bse
            return np.column_stack([lower, upper])

        # fixed effect
        for m in self.measurement:
            m.results = np.array([0])  # Dummy results for compatibility
            m.model = None

            # Get the parameter values for this variable and assign them to the model namespace
            m.params = self.params.beta.value

            # Get the parameter names for this variable and assign them to the model namespace
            xnames_stack = [item for sublist in self.model.xbeta_names for item in sublist]
            m.params_name = xnames_stack

            m.bse = np.full(
                len(m.params), np.nan
            )  # Set standard errors to NaN (not available)
            m.tvalues = np.full(
                len(m.params), np.nan
            )  # Set t-values to NaN (not available)
            m.pvalues = np.full(
                len(m.params), np.nan
            )  # Set p-values to NaN (not available)

            m.conf_int = lambda alpha=0.05: conf_int_params(m.params, m.bse, alpha)

            smry.add_table_params(m, xname=m.params_name, alpha=0.05)

        # Measrement error
        temp = SimpleNamespace()
        temp.results = np.array([0])
        temp.model = None
        temp.params = np.hstack((self.params.s2e.value, self.params.A.value.flatten()))
        temp.params_name = [f"s2e_{i}" for i in range(self.params.s2e.size)] + [
            f"A_{i,j}"
            for i in range(self.params.A.shape[0])
            for j in range(self.params.A.shape[1])
        ]
        temp.bse = np.full(
            len(temp.params), np.nan
        )  # Set standard errors to NaN (not available)
        temp.tvalues = np.full(
            len(temp.params), np.nan
        )  # Set t-values to NaN (not available)
        temp.pvalues = np.full(
            len(temp.params), np.nan
        )  # Set p-values to NaN (not available)
        temp.conf_int = lambda alpha=0.05: conf_int_params(temp.params, temp.bse, alpha)
        smry.add_table_params(temp, xname=temp.params_name, alpha=0.05)

        
        # todo: add the parameters table (state equation parameters)
        temp = (
            SimpleNamespace()
        )  # Create a list with one SimpleNamespace for compatibility with summary structure
        temp.results = np.array([0])  # Dummy results for compatibility
        temp.model = None

        # Get the parameter values for this variable and assign them to the model namespace
        temp.matern_params = np.array([cov.rescale for cov in self.model.cov_function])
        temp.markov_params = self.params.f.value
        temp.params = np.hstack((temp.matern_params, temp.markov_params))

        temp.matern_names = [
            f"rescale_{j}" for j in range(len(self.model.cov_function))
        ]
        temp.latent_names = [f"f_{i}" for i in range(self.params.f.value.size)]

        # Combine all parameters and names into a single list for the summary
        temp.param_names = temp.matern_names + temp.latent_names

        temp.bse = np.full(
            len(temp.params), np.nan
        )  # Set standard errors to NaN (not available)
        temp.tvalues = np.full(
            len(temp.params), np.nan
        )  # Set t-values to NaN (not available)
        temp.pvalues = np.full(
            len(temp.params), np.nan
        )  # Set p-values to NaN (not available)

        temp.conf_int = lambda alpha=0.05: conf_int_params(temp.params, temp.bse, alpha)
        smry.add_table_params(temp, xname=temp.param_names, alpha=0.05)

        return smry
