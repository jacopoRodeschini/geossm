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
from dataclasses import replace, fields


ArrayLike = Optional[Any]

# %% [Utils] Define pack/unpack helpers to flatten params to a 1D array

def _pack_params(params):
    """Flatten free parameter values into one 1D JAX array."""
    parts = []
    meta = []
    start = 0

    for f in fields(params):
        p = getattr(params, f.name)

        if p is None or p.value is None or p.fixed:
            continue

        arr = jnp.asarray(p.value).ravel()
        stop = start + arr.size
        parts.append(arr)
        meta.append((f.name, p.value.shape, start, stop))
        start = stop

    if parts:
        vec = jnp.concatenate(parts)
    else:
        vec = jnp.zeros((0,), dtype=jnp.float32)

    return vec, meta


def _unpack_params(vec, template_params, meta):
    """Rebuild ModelParams from flat vector using template_params as template."""
    meta_map = {name: (shape, start, stop) for name, shape, start, stop in meta}
    updated = {}

    for f in fields(template_params):
        p = getattr(template_params, f.name)

        if p is None or p.value is None or p.fixed or f.name not in meta_map:
            updated[f.name] = p
            continue

        shape, start, stop = meta_map[f.name]
        updated[f.name] = replace(p, value=vec[start:stop].reshape(shape))

    return ModelParams(**updated)


def _vector_to_bse_params(vec, template_params, meta):
    """Attach a flat BSE vector back into a ModelParams object."""
    meta_map = {name: (shape, start, stop) for name, shape, start, stop in meta}
    updated = {}

    for f in fields(template_params):
        p = getattr(template_params, f.name)

        if p is None or p.value is None:
            updated[f.name] = p
            continue

        if p.fixed or f.name not in meta_map:
            updated[f.name] = replace(p, bse=None)
            continue

        shape, start, stop = meta_map[f.name]
        updated[f.name] = replace(p, bse=vec[start:stop].reshape(shape))

    return ModelParams(**updated)

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
            return self._hessian, 0.0  # Return cached Hessian and zero time delta

        params = self.params.copy()  # Create a copy of the parameters to avoid modifying the original

        # fix x0, Sigma0, so that no derivatives are computed
        params.x0.fixed = True
        params.Sigma0.fixed = True
        x0 = params.x0.value
        Sigma0 = params.Sigma0.value
        
        # Get the scalar positive log-likelihood function
        logL = self.model._observed_logL(self.y_obs, self.Xbeta, x0, Sigma0)
        # fun(params)
        
        # Pack free params into a flat array
        vec0, meta = _pack_params(params)
        self._free_meta = meta

        
        # Wrap fun to accept a flat vector
        def fun_flat(vec):
            p = _unpack_params(vec, params, meta)
            return logL(p)

        ts = time.time()
        # Compute the Hessian using JAX, of the observed log-likelihood function at the argument 0 (params)
        hesfun = jax.hessian(fun_flat)
        # Evaluate the Hessian at the estimated parameters
        hessian = hesfun(vec0)
        jax.block_until_ready(hessian)
        tdelta = time.time() - ts

        # the Information matrix, which is the negative of the second derivative of the log-likelihood 
        # function
        self._hessian = hessian
        return hessian, tdelta
        
    def compute_cov_params(self):
        """
        Compute the standard errors of the estimated parameters based on the Hessian matrix.
        """
        if self._cov_params is not None:
            return self._cov_params

        self.model._log("Computing Hessian and standard errors of the parameters", verbose=True)
        H, tdelta = self._compute_hessian()
        self.model._log(f"Hessian computed in {tdelta:.3f} seconds", verbose=True)

        # Compute the covariance matrix as the inverse of the Hessian (of the positive log-likelihood)
        # Sigma = -jnp.linalg.pinv(H)
        # cov_params = -jnp.linalg.inv(H)
        H = 0.5 * (H + H.T)
        # eps = 1e-6
        # H += eps * jnp.eye(H.shape[0], dtype=H.dtype)
        cov_params = -jnp.linalg.solve(H, jnp.eye(H.shape[0], dtype=H.dtype))

        # check if it is postive definte
        # chol = jnp.linalg.cholesky(cov_params)

        self._cov_params = cov_params
        return cov_params
    
    @property
    def bse(self):
        
        bse_vec = jnp.sqrt(jnp.clip(jnp.diag(self._cov_params), a_min=0.0))
        # se = np.sqrt(np.diag(self.cov_params))

        params  = _unpack_params(bse_vec, self.params, 
            {name: (getattr(self.params, name).value.shape, 
                getattr(self.params, name).value.size) 
                    for name in self.params.__dataclass_fields__})
        return params

    @property
    def bse_vector(self):
        cov = self.compute_cov_params()
        return np.sqrt(np.clip(np.asarray(jnp.diag(cov)), a_min=0.0, a_max=None))

    @property
    def bse(self):
        # structured ModelParams with bse stored in each Param
        if getattr(self, "_bse_params", None) is None:
            self._bse_params = _vector_to_bse_params(self.bse_vector, self.params, self._free_meta)
        return self._bse_params

       
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
        llf = self.llf
        k = getattr(self, "n_params", 0)
        k = int(k) if k is not None else 0
        self._aic = 2 * k - 2 * llf
        return self._aic

    def compute_bic(self):
        llf = self.llf
        k = getattr(self, "n_params", 0)
        k = int(k) if k is not None else 0
        n = self.nobs if self.nobs is not None else 1
        self._bic = np.log(n) * k - 2 * llf
        return self._bic

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
