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
        meta.append((f.name, p.value.shape, start, stop, p.fixed))
        start = stop

    if parts:
        vec = jnp.concatenate(parts)
    else:
        vec = jnp.zeros((0,), dtype=jnp.float32)

    return vec, meta


def _unpack_params(vec, template_params, meta):
    """Rebuild ModelParams from flat vector using template_params as template."""
    meta_map = {name: (shape, start, stop, fixed) for name, shape, start, stop, fixed in meta}
    updated = {}

    for f in fields(template_params):
        p = getattr(template_params, f.name)

        if p is None or p.value is None or p.fixed or f.name not in meta_map:
            updated[f.name] = p
            continue

        shape, start, stop, fixed = meta_map[f.name]
        updated[f.name] = replace(p, value=vec[start:stop].reshape(shape))
        updated[f.name] = replace(updated[f.name], fixed=fixed)

    return ModelParams(**updated)


def _vector_to_bse_params(vec, template_params, meta):
    """Attach a flat BSE vector back into a ModelParams object."""
    meta_map = {name: (shape, start, stop, fixed) for name, shape, start, stop, fixed in meta}
    updated = {}

    for f in fields(template_params):
        p = getattr(template_params, f.name)

        if p is None or p.value is None:
            updated[f.name] = p
            continue

        if p.fixed or f.name not in meta_map:
            updated[f.name] = replace(p, bse=None)
            continue

        shape, start, stop, _fixed = meta_map[f.name]
        updated[f.name] = replace(p, bse=vec[start:stop].reshape(shape))

    return ModelParams(**updated)


def _equalize_row_widths(text: str) -> str:
    """Pad every line of a rendered Summary to the same width.

    statsmodels sizes each sub-table's borders independently, so the top
    info table and the parameter tables can end up with different total
    widths and look misaligned when printed together. This re-pads every
    line to the overall max width: pure rule lines ('===' / '---') are
    extended with their own character, everything else with trailing
    spaces, so the borders line up without touching column contents.
    """
    lines = text.split("\n")
    width = max((len(line) for line in lines), default=0)

    padded = []
    for line in lines:
        if line and len(set(line)) == 1 and line[0] in "=-":
            padded.append(line[0] * width)
        else:
            padded.append(line.ljust(width))
    return "\n".join(padded)

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
        self._free_meta = None
        self._frozen_params = None

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
        self.param_fixed = [getattr(self.params, name).fixed for name in names]

    @property
    def n_params(self):
        if self._n_params is None:
            self._n_params = sum(d for d, f in zip(self.param_dim, self.param_fixed) if not f) if self.param_dim is not None and self.param_fixed else 0
        return self._n_params
    
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
        runtime_filter = [v["tdelta_E_detail"][0] for v in self.nstats]
        runtime_smoother = [v["tdelta_E_detail"][1] for v in self.nstats]
        runtime_expectation = [v["tdelta_E_detail"][2] for v in self.nstats]

        self.runtime_tot_estep = sum(runtime_estep)
        self.runtime_tot_mstep = sum(runtime_mstep)

        self.time_filter = sum(runtime_filter)
        self.time_smoother = sum(runtime_smoother)
        self.time_expectation = sum(runtime_expectation)
        self.time_total = sum(runtime_each)
        
        self.llf_path = [v["logL"] for v in self.nstats]
        self.llf = self.llf_path[-1]
    
    def _inference_params(self):
        """
        Params used for Hessian/inference: a copy of self.params with x0 and
        Sigma0 forced fixed (no derivatives are computed for the initial state).

        theta_hat, bse_vector, tvalues, pvalues and conf_int all derive their
        flat-vector layout from this same frozen copy, so they stay aligned
        even if self.params.x0 / self.params.Sigma0 are marked free.
        """
        if self._frozen_params is None:
            params = self.params.copy()
            params.x0.fixed = True
            params.Sigma0.fixed = True
            self._frozen_params = params
        return self._frozen_params

    def _nan_bse_params(self):
        """
        ModelParams copy with every `.bse` replaced by NaN placeholders,
        used by summary(hessian=False) to print point estimates without
        forcing the (possibly slow) Hessian computation.
        """
        updated = {}
        for f in fields(self.params):
            p = getattr(self.params, f.name)
            if p is None or p.value is None:
                updated[f.name] = p
                continue
            nan_bse = jnp.full(p.value.shape, jnp.nan, dtype=jnp.asarray(p.value).dtype)
            updated[f.name] = replace(p, bse=nan_bse)
        return ModelParams(**updated)

    def _compute_hessian(self):
        """
        Compute the Hessian matrix of the log-likelihood function at the estimated parameters.
        """
        if self._hessian is not None:
            return self._hessian, 0.0  # Return cached Hessian and zero time delta

        params = self._inference_params()
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
    
        
    # @property
    # def bse(self):
        
    #     bse_vec = jnp.sqrt(jnp.clip(jnp.diag(self._cov_params), a_min=0.0))
    #     # se = np.sqrt(np.diag(self.cov_params))

    #     params  = _unpack_params(bse_vec, self.params, 
    #         {name: (getattr(self.params, name).value.shape, 
    #             getattr(self.params, name).value.size) 
    #                 for name in self.params.__dataclass_fields__})
    #     return params

    
    @property
    def df_resid(self):
        """
        Compute the degrees of freedom of the residuals.
        """
        nobs = self.nobs if self.nobs is not None else 0
        n_params = self.n_params if self.n_params is not None else 0
        return nobs - n_params

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
    
    def _stats_from_arrays(self, params, bse, alpha=0.05):
        params = np.asarray(params, dtype=float).ravel()
        bse = np.asarray(bse, dtype=float).ravel()
    
        with np.errstate(divide="ignore", invalid="ignore"):
            t = params / bse
    
        if self.df_resid is not None:
            p = 2 * stats.t.sf(np.abs(t), df=self.df_resid)
            crit = stats.t.ppf(1 - alpha / 2.0, df=self.df_resid)
        else:
            p = 2 * stats.norm.sf(np.abs(t))
            crit = stats.norm.ppf(1 - alpha / 2.0)
    
        ci = np.column_stack([params - crit * bse, params + crit * bse])
        return t, p, ci

       
    @property
    def theta_hat(self):
        vec, _ = _pack_params(self._inference_params())
        return np.asarray(vec)
    
    @property
    def tvalues(self):
        t, _, _ = self._stats_from_arrays(self.theta_hat, self.bse_vector)
        return t
    
    @property
    def pvalues(self):
        _, p, _ = self._stats_from_arrays(self.theta_hat, self.bse_vector)
        return p
    
    def conf_int(self, alpha=0.05):
        _, _, ci = self._stats_from_arrays(self.theta_hat, self.bse_vector, alpha=alpha)
        return ci


    @property
    def aic(self):
        return self.compute_aic()

    @property
    def bic(self):
        return self.compute_bic()

    # Compute AIC and BIC
    def compute_aic(self):
        llf = getattr(self, "llf", None)
        if llf is None:
            raise AttributeError(
                "AIC requires the log-likelihood, which is only set when "
                "`nstats` is provided to LRStateSpaceResults."
            )
        k = getattr(self, "n_params", 0)
        k = int(k) if k is not None else 0
        self._aic = 2 * k - 2 * llf
        return self._aic

    def compute_bic(self):
        llf = getattr(self, "llf", None)
        if llf is None:
            raise AttributeError(
                "BIC requires the log-likelihood, which is only set when "
                "`nstats` is provided to LRStateSpaceResults."
            )
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
                
                # ("Runtime total (s):", lambda: [f"{self.runtime_tot:.3g}"]),
                ("AIC:", lambda: [f"{self.aic:.4g}"]),
                ("BIC:", lambda: [f"{self.bic:.4g}"]),
            ]
        )

        top_right_em = {
            "Runtime E-step (s):": lambda: [f"{time_e:.3g}"],
            "Runtime M-step (s):": lambda: [f"{time_m:.3g}"],
            "EM iters :": lambda: [f"{self.iterations}"], 
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
    def summary(self, hessian=False, alpha=0.05):

        # self.results = np.array([0])
        # self.params = self.beta
        # self.param_names = self.xbeta_names
        # self.bse = np.zeros(len(self.beta))
        # self.tvalues = np.zeros(len(self.beta))
        # self.pvalues = np.zeros(len(self.beta))
        
        if hessian:
            bse_struct = self.bse  # structured ModelParams with bse fields
        else:
            if self._hessian is not None:
                bse_struct = self.bse
            else:
                # Hessian/SE not requested yet (e.g. deferred because it's slow) -
                # show point estimates with NaN placeholders for bse/t/p/CI.
                bse_struct = self._nan_bse_params()

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

        # fixed effect
        for m in self.measurement:
            m.results = np.array([0])  # Dummy results for compatibility
            m.model = None

            # Get the parameter names for this variable and assign them to the model namespace
            xnames_stack = [item for sublist in self.model.xbeta_names for item in sublist]

            
            # Get the fixed effect block statistics
            beta_vals = np.asarray(self.params.beta.value).ravel()
            beta_bse = np.asarray(bse_struct.beta.bse).ravel()
            beta_t, beta_p, _ = self._stats_from_arrays(beta_vals, beta_bse)

            m.params = beta_vals
            m.bse = beta_bse
            m.tvalues = beta_t
            m.pvalues = beta_p
            m.params_name = xnames_stack
            m.conf_int = lambda alpha=alpha, v=beta_vals, s=beta_bse: self._stats_from_arrays(v, s, alpha)[2]

            smry.add_table_params(m, xname=m.params_name, alpha=alpha)

        # Measrement error
        temp = SimpleNamespace()
        temp.results = np.array([0])
        temp.model = None
        temp.params_name = [f"s2e_{i}" for i in range(self.params.s2e.size)] + [
            f"A_{i}_{j}"
            for i in range(self.params.A.shape[0])
            for j in range(self.params.A.shape[1])
        ]
        
        
        s2e_vals = np.asarray(self.params.s2e.value).ravel()
        A_vals = np.asarray(self.params.A.value).ravel()
        
        s2e_bse = np.asarray(bse_struct.s2e.bse).ravel()
        A_bse = np.asarray(bse_struct.A.bse).ravel()
        
        temp.params = np.hstack((s2e_vals, A_vals))
        temp.bse = np.hstack((s2e_bse, A_bse))
        temp.tvalues, temp.pvalues, _ = self._stats_from_arrays(temp.params, temp.bse)
        temp.conf_int = lambda alpha=alpha, v=temp.params, s=temp.bse: self._stats_from_arrays(v, s, alpha)[2]
        
        smry.add_table_params(temp, xname=temp.params_name, alpha=alpha)

        
        # todo: add the parameters table (state equation parameters)
        temp = (
            SimpleNamespace()
        )  # Create a list with one SimpleNamespace for compatibility with summary structure
        temp.results = np.array([0])  # Dummy results for compatibility
        temp.model = None

        # Get the parameter values for this variable and assign them to the model namespace
        ks_val = np.asarray(self.params.ks.value).ravel()
        ks_bse = np.asarray(bse_struct.ks.bse).ravel()
        f_val = np.asarray(self.params.f.value).ravel()
        f_bse = np.asarray(bse_struct.f.bse).ravel()

        matern_names = [
            f"rescale_{j}" for j in range(len(self.model.cov_function))
        ]
        latent_names = [f"f_{i}" for i in range(self.params.f.value.size)]

        # Combine all parameters and names into a single list for the summary
        temp.param_names = matern_names + latent_names

        temp.params = np.hstack((ks_val, f_val))
        temp.bse = np.hstack((ks_bse, f_bse))
        temp.tvalues, temp.pvalues, _ = self._stats_from_arrays(temp.params, temp.bse)
        temp.conf_int = lambda alpha=alpha, v=temp.params, s=temp.bse: self._stats_from_arrays(v, s, alpha)[2]
        
        smry.add_table_params(temp, xname=temp.param_names, alpha=alpha)

        # Re-pad every rendered row to a common width so the top info table
        # and the parameter tables line up when printed together.
        smry.as_text = lambda _orig=smry.as_text: _equalize_row_widths(_orig())

        return smry
