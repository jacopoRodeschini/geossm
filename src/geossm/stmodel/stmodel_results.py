from statsmodels.iolib.summary import Summary
from typing import Any, Optional
import numpy as np
from geossm.stmodel.param import Param, ModelParams, FitOptions
from geossm.ssm import StateSpaceResults, _safe_sqrt_variance
from types import SimpleNamespace
from scipy import stats
import time
import jax
import jax.numpy as jnp
from dataclasses import replace, fields
from geossm.utils import _on_device, split_by_block, format_info_table


ArrayLike = Optional[Any]

# %% [Utils] Define pack/unpack helpers to flatten params to a 1D array

def _pack_params(params, dtype=jnp.float32):
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
        vec = jnp.zeros((0,), dtype=dtype)

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
    Results of fitting a :class:`geossm.stmodel.LRStateSpaceModel`.

    Subclass of :class:`geossm.ssm.StateSpaceResults` returned by
    `LRStateSpaceModel.fit()`, carrying the estimated `ModelParams`
    (`.params`), the EM iteration history (`.nstats`/`.llf_path`), and the
    in-sample fitted values/smoothed states, in addition to everything the
    base class already provides (residuals, `mse`/`rmse`,
    `conf_int_y`, ...). It also adds:

    - Parameter inference: `compute_cov_params` (observed-information
      Hessian, via `jax.hessian` on the model's log-likelihood), `bse`,
      `tvalues`, `pvalues`, `conf_int`, `aic`/`bic`.
    - Out-of-sample prediction: `predict` (thin wrapper around
      `LRStateSpaceModel.predict`, storing `y_pred`/`Sigma_y_pred` on this
      object).
    - Back-transformation of predictions/fitted values to the response's
      original scale via the delta method (`back_transform`), when the
      model formula applies a transform to the response (e.g. `np.log`).
    - Export to `geopandas.GeoDataFrame` (`to_geo`).
    - A `statsmodels`-style textual report (`summary`).

    Per-variable ("_list") views: `y_hat_list`/`Sigma_y_hat_list`
    (in-sample fitted), `y_obs_list` (training observations),
    `residuals_list`, and -- after `.predict()` -- `y_pred_list`/
    `Sigma_y_pred_list`, are all lazy `@property` views splitting the
    corresponding stacked array (`y_hat`, `y_obs`, `residuals`, `y_pred`)
    one entry per response variable. After `.back_transform()`, their
    original-scale counterparts `y_hat_back_list`, `y_obs_back_list`,
    `residuals_back_list`, and `y_pred_back_list` are plain attributes
    (genuine computations, not free slices of a stacked array).

    Typical usage, continuing from `LRStateSpaceModel.fit()`::

        results = model.fit()
        results.compute_cov_params()      # optional, needed for bse/t/p/CI
        print(results.summary())
        results = results.predict(new_df) # out-of-sample prediction
        results = results.back_transform(g_inv=jnp.exp)  # if y = np.log(...)
        geo = results.to_geo()
    """

    def __init__(
        self,
        model=None,
        params: ModelParams = None,
        nstats: list = None,
        options: FitOptions = None,
        block_p=None,
        points_hat: list = None,
        timestamps_hat: list = None,
        crs_hat=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : LRStateSpaceModel, optional
            The model instance that produced these results.
        params : ModelParams, optional
            Estimated parameters (`beta`, `s2e`, `f`, `A`, `ks`, `x0`,
            `Sigma0`); processed into `param_names`/`param_values`/
            `param_dim`/`param_fixed` via `_process_params`.
        nstats : list of dict, optional
            Per-EM-iteration statistics produced by
            `LRStateSpaceModel._log_iteration`; processed via
            `_process_nstats` into `iterations`, `llf`/`llf_path`, and the
            per-phase runtime totals.
        options : FitOptions, optional
            The `FitOptions` the model was fit with (stored as-is, for
            reference).
        block_p : array-like, optional
            Cumulative index boundaries splitting the stacked training
            arrays (`y_hat`, `Sigma_y_hat`, `y_obs`, `residuals`, all from
            the base class) into one block per response variable
            (`block_p[i]:block_p[i+1]` selects variable `i`); snapshotted
            here rather than read live off `model.block_p`, which is
            mutable. Drives the `y_hat_list`/`Sigma_y_hat_list`/
            `y_obs_list`/`residuals_list` properties (see
            `geossm.utils.split_by_block`).
        points_hat, timestamps_hat, crs_hat : list / list / CRS, optional
            Training grid (sites, timestamps) and CRS, one entry per
            response variable, snapshotted for `to_geo()`.
        **kwargs
            Forwarded to `StateSpaceResults.__init__` (e.g. `y_hat`,
            `Sigma_y_hat`, `x_smoothed`, `P_smoothed`, `S11`/`S10`/`S00`,
            `llf`, ...).
        """
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

        # Cumulative per-response-variable index boundaries for the
        # stacked training arrays (`y_hat`, `Sigma_y_hat`, `y_obs`,
        # `residuals`); snapshotted here (not read live off
        # `self.model.block_p`) since that attribute is mutable and would
        # go stale for this results object the next time `fit()`/`setup()`
        # runs on this model instance. Drives the `y_hat_list`/
        # `Sigma_y_hat_list`/`y_obs_list`/`residuals_list` properties below.
        self.block_p = block_p

        # Training grid (one entry per response variable), snapshotted for
        # the same reason as block_p above -- used by `.to_geo()`.
        self.points_hat = points_hat
        self.timestamps_hat = timestamps_hat
        self.crs_hat = crs_hat

        # ---- Derived quantities (initialized empty) ----
        self.param_names = None
        self.param_values = None
        self.param_dim = None

        self.iterations = 0
        self.runtime_tot_estep = 0.0
        self.runtime_tot_mstep = 0.0

        # Out-of-sample prediction (populated by .predict()): stacked
        # arrays, distinct from `y_hat`/`Sigma_y_hat`, which hold the
        # in-sample filtered/fitted values used for residuals. Their own
        # `block_p_pred` is snapshotted separately from `block_p` (the
        # training grid's) since the prediction grid can have a different
        # number of points per response variable. `y_pred_list`/
        # `Sigma_y_pred_list` (below) are derived properties from these.
        self.points_pred = None
        self.block_p_pred = None
        self.y_pred = None
        self.Sigma_y_pred = None
        self.tdelta_pred = None
        self.timestamps_pred = None
        self.crs_pred = None

        # Original-scale (back-transformed) counterparts of y_hat_list/
        # y_obs_list/y_pred_list/residuals_list, populated by
        # .back_transform() -- see that method's docstring. Per-variable
        # views only (no stacked *_back counterparts): unlike the plain
        # `_list` views, these are genuine computations (delta method, or
        # a direct transform), not free slices of a stacked array, so they
        # are plain attributes rather than properties.
        self.y_hat_back_list = None
        self.Sigma_y_hat_back_list = None
        self.y_obs_back_list = None
        self.y_pred_back_list = None
        self.Sigma_y_pred_back_list = None
        self.residuals_back_list = None

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
        """
        int : Total number of *free* (not `fixed`) scalar parameters across
        `beta`, `s2e`, `f`, `A`, `ks`, `x0`, `Sigma0`, used as the degrees
        of freedom / penalty term `k` in `df_resid`, `compute_aic`, and
        `compute_bic`.
        """
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

    @_on_device
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
        vec0, meta = _pack_params(params, dtype=self.dtype)
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
        
    @_on_device
    def compute_cov_params(self):
        """
        Compute the asymptotic covariance matrix of the free parameters
        from the observed-information Hessian of the log-likelihood
        (`-inverse(Hessian)`, symmetrised), and cache it.

        This drives `bse`/`bse_vector`, `tvalues`, `pvalues`, and
        `conf_int`, and its presence (together with `summary()`) decides
        whether `summary()` shows real standard errors or NaN placeholders
        (see `_nan_bse_params`). Computing the Hessian
        (`_compute_hessian`, via `jax.hessian`) can be relatively slow for
        models with many free parameters, so it is not run automatically by
        `fit()`; call this explicitly when standard errors are needed.

        Returns
        -------
        jax.numpy.ndarray
            The covariance matrix of the free (not `fixed`) parameters,
            in the flat order given by `_pack_params`/`theta_hat`. Cached
            on `self._cov_params` after the first call.
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
    def df_resid(self):
        """
        int : Residual degrees of freedom, `nobs - n_params` (number of
        observed values minus the number of free parameters). Used as the
        Student-t degrees of freedom in `_stats_from_arrays` (p-values and
        confidence intervals).
        """
        nobs = self.nobs if self.nobs is not None else 0
        n_params = self.n_params if self.n_params is not None else 0
        return nobs - n_params

    @property
    def bse_vector(self):
        """
        ndarray : Standard errors of the free parameters, as a flat 1D
        array (`sqrt` of the diagonal of `compute_cov_params()`, via
        `_safe_sqrt_variance` to guard against small negative/NaN values
        from numerical noise), in the same flat order as `theta_hat`.

        Does *not* trigger the Hessian computation: if `compute_cov_params()`
        hasn't been called yet, returns a NaN-filled array of the same
        shape instead. Call `compute_cov_params()` explicitly to get real
        standard errors.
        """
        if self._cov_params is None:
            return np.full(self.theta_hat.shape, np.nan)
        return _safe_sqrt_variance(np.asarray(jnp.diag(self._cov_params)), context="bse_vector")

    @property
    def bse(self):
        """
        ModelParams : Standard errors of the free parameters, reshaped back
        into a `ModelParams` structure (one array per parameter, stored in
        each `Param.bse`) via `_vector_to_bse_params`, mirroring the shape
        of `self.params`. Fixed parameters get `bse=None`.

        Does *not* trigger the Hessian computation: if `compute_cov_params()`
        hasn't been called yet, returns NaN placeholders (see
        `_nan_bse_params`) instead. Call `compute_cov_params()` explicitly
        to get real standard errors.
        """
        if self._cov_params is None:
            return self._nan_bse_params()

        # structured ModelParams with bse stored in each Param
        if getattr(self, "_bse_params", None) is None:
            self._bse_params = _vector_to_bse_params(self.bse_vector, self.params, self._free_meta)
        return self._bse_params

    def _stats_from_arrays(self, params, bse, alpha=0.05):
        """Compute t-values, two-sided p-values, and `alpha`-level confidence
        intervals for flat `params`/`bse` arrays, using the Student-t
        distribution with `df_resid` degrees of freedom (or the Normal
        distribution if `df_resid` is unavailable)."""
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
        """
        ndarray : Point estimates of the free (not `fixed`) parameters,
        flattened into a single 1D array in the order produced by
        `_pack_params` (`x0`/`Sigma0` always treated as fixed here, see
        `_inference_params`) -- the same layout as `bse_vector`, `tvalues`,
        `pvalues`, and `conf_int`.
        """
        vec, _ = _pack_params(self._inference_params(), dtype=self.dtype)
        return np.asarray(vec)

    @property
    def tvalues(self):
        """ndarray : t-statistics (`theta_hat / bse_vector`) for the free
        parameters, in the same flat order as `theta_hat`. NaN until
        `compute_cov_params()` has been called (see `bse_vector`)."""
        t, _, _ = self._stats_from_arrays(self.theta_hat, self.bse_vector)
        return t

    @property
    def pvalues(self):
        """ndarray : Two-sided p-values for the free parameters (Student-t
        with `df_resid` degrees of freedom), in the same flat order as
        `theta_hat`. NaN until `compute_cov_params()` has been called (see
        `bse_vector`)."""
        _, p, _ = self._stats_from_arrays(self.theta_hat, self.bse_vector)
        return p

    def conf_int(self, alpha=0.05):
        """
        Confidence intervals for the free parameters. NaN until
        `compute_cov_params()` has been called (see `bse_vector`).

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level; the returned interval has coverage
            `1 - alpha` (e.g. `alpha=0.05` gives a 95% confidence interval).

        Returns
        -------
        ndarray, shape (n_free_params, 2)
            Lower/upper confidence bounds, in the same flat parameter order
            as `theta_hat`.
        """
        _, _, ci = self._stats_from_arrays(self.theta_hat, self.bse_vector, alpha=alpha)
        return ci


    @property
    def aic(self):
        """float : Akaike Information Criterion, `compute_aic()`."""
        return self.compute_aic()

    @property
    def bic(self):
        """float : Bayesian Information Criterion, `compute_bic()`."""
        return self.compute_bic()

    # Compute AIC and BIC
    def compute_aic(self):
        """
        Akaike Information Criterion: `2 * k - 2 * llf`, with `k = n_params`
        (free parameters) and `llf` the fitted log-likelihood.

        Raises
        ------
        AttributeError
            If `llf` is not set (only available when `nstats` was provided
            to `__init__`, i.e. after `LRStateSpaceModel.fit()`).
        """
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
        """
        Bayesian Information Criterion: `log(nobs) * k - 2 * llf`, with
        `k = n_params` (free parameters), `nobs` the number of observed
        values, and `llf` the fitted log-likelihood.

        Raises
        ------
        AttributeError
            If `llf` is not set (only available when `nstats` was provided
            to `__init__`, i.e. after `LRStateSpaceModel.fit()`).
        """
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
        Compute out-of-sample predictions based on smoothed states and model
        parameters. `self.model.predict` stores them directly on this
        results object (`points_pred`, `block_p_pred`, `y_pred`,
        `Sigma_y_pred`, `tdelta_pred`, `timestamps_pred`, `crs_pred`; the
        per-variable views `y_pred_list`/`Sigma_y_pred_list` are then
        derived properties) and returns `self`, so predictions travel with
        the fitted model and can be reused by other methods (e.g.
        `.to_geo()`, plotting, summaries) without re-running prediction.

        Parameters
        ----------
        df : geopandas.GeoDataFrame
            New locations/times to predict at; see
            `LRStateSpaceModel.predict` for the exact requirements.
        verbose : bool, default True
            Verbosity for the underlying grid/design-matrix construction.

        Returns
        -------
        LRStateSpaceResults
            `self`, with the prediction attributes populated.
        """
        return self.model.predict(df, modelresults=self, verbose=verbose)

    # ---- Per-variable ("_list") views -----------------------------------
    #
    # Every `*_list` property below is a lazy, uncached view of an
    # already-stored stacked array, split one entry per response variable
    # via `geossm.utils.split_by_block`. Row-slicing a contiguous array
    # returns a view rather than a copy, so these are effectively free in
    # memory/compute -- no caching is needed, and none is done.
    #
    # In-sample:   y_hat / Sigma_y_hat   -> y_hat_list / Sigma_y_hat_list
    #              y_obs                 -> y_obs_list
    #              residuals             -> residuals_list
    # Out-of-sample (after .predict()):
    #              y_pred / Sigma_y_pred -> y_pred_list / Sigma_y_pred_list
    #
    # Original-scale counterparts (y_hat_back_list, y_obs_back_list,
    # y_pred_back_list, residuals_back_list, ...) are genuine computations
    # (delta method / direct transform) rather than free slices, so they
    # are plain attributes populated by `back_transform()` instead.

    @property
    def y_hat_list(self):
        """list of ndarray : Per-response-variable view of the base
        class's stacked `y_hat` (in-sample fitted mean), split along
        `block_p`."""
        if self.y_hat is None or self.block_p is None:
            return None
        y_list, _ = split_by_block(self.y_hat, self.block_p, self.Sigma_y_hat)
        return y_list

    @property
    def Sigma_y_hat_list(self):
        """list of ndarray : Per-response-variable view of the base
        class's stacked `Sigma_y_hat` (in-sample fitted covariance),
        split along `block_p` (diagonal blocks only)."""
        if self.y_hat is None or self.Sigma_y_hat is None or self.block_p is None:
            return None
        _, Sigma_list = split_by_block(self.y_hat, self.block_p, self.Sigma_y_hat)
        return Sigma_list

    @property
    def y_obs_list(self):
        """list of ndarray : Per-response-variable view of the base
        class's stacked `y_obs` (training observations), split along
        `block_p`."""
        if self.y_obs is None or self.block_p is None:
            return None
        return split_by_block(self.y_obs, self.block_p)

    @property
    def residuals_list(self):
        """list of ndarray : Per-response-variable view of the base
        class's stacked `residuals` (`y_obs - y_hat`), split along
        `block_p`."""
        if self.residuals is None or self.block_p is None:
            return None
        return split_by_block(self.residuals, self.block_p)

    @property
    def y_pred_list(self):
        """list of ndarray : Per-response-variable view of `y_pred`
        (out-of-sample prediction mean, set by `.predict()`), split along
        `block_p_pred` -- the prediction grid's own block boundaries,
        since the prediction grid can have a different number of points
        per response variable than the training grid."""
        if self.y_pred is None or self.block_p_pred is None:
            return None
        y_list, _ = split_by_block(self.y_pred, self.block_p_pred, self.Sigma_y_pred)
        return y_list

    @property
    def Sigma_y_pred_list(self):
        """list of ndarray : Per-response-variable view of `Sigma_y_pred`
        (out-of-sample prediction covariance, set by `.predict()`), split
        along `block_p_pred` (diagonal blocks only)."""
        if self.y_pred is None or self.Sigma_y_pred is None or self.block_p_pred is None:
            return None
        _, Sigma_list = split_by_block(self.y_pred, self.block_p_pred, self.Sigma_y_pred)
        return Sigma_list

    def _pred_summary_stats(self):
        """
        Small numeric summary of the last computed prediction (`.predict()`),
        used by `generate_summary()` to populate the top_left_pred /
        top_right_pred summary tables.
        """
        y_all = np.concatenate([np.asarray(y).ravel() for y in self.y_pred_list])
        n_points = sum(np.asarray(p).shape[0] for p in self.points_pred)

        std_all = []
        for sigma in self.Sigma_y_pred_list:
            sigma = np.asarray(sigma)
            var = np.diagonal(sigma, axis1=0, axis2=1)  # (T, n_i)
            std_all.append(_safe_sqrt_variance(var, context="_pred_summary_stats").ravel())
        std_all = np.concatenate(std_all) if std_all else np.array([np.nan])

        return {
            "n_points": n_points,
            "n_pred": y_all.size,
            "n_missing": int(np.sum(np.isnan(y_all))),
            "y_min": np.nanmin(y_all),
            "y_median": np.nanmedian(y_all),
            "y_max": np.nanmax(y_all),
            "y_mean_std": np.nanmean(std_all),
        }

    @staticmethod
    def _build_geo_dataframe(
        y_names, points_list, timestamps_list, y_list, Sigma_list, crs,
        prefix, y_back_list=None, Sigma_back_list=None,
    ):
        """
        Build one GeoDataFrame, one row per (point, timestamp), from a set
        of per-variable mean/covariance lists that share a common grid
        (`points_list[i]`/`timestamps_list[i]` must be the same across `i`
        -- true for both `y_hat_list`/`Sigma_y_hat_list` and
        `y_pred_list`/`Sigma_y_pred_list`, each built from a single input
        dataframe). `prefix` (`"hat"` or `"pred"`) names the value columns:
        `y_<prefix>_<var>`/`std_<prefix>_<var>`, plus `y_<prefix>_back_<var>`/
        `std_<prefix>_back_<var>` when `y_back_list`/`Sigma_back_list` are
        given.
        """
        import geopandas as geopd
        from shapely.geometry import Point

        points = np.asarray(points_list[0])
        ts = timestamps_list[0]
        n, T = points.shape[0], ts.shape[0]

        for name, p, t in zip(y_names, points_list, timestamps_list):
            if np.asarray(p).shape[0] != n or np.asarray(t).shape[0] != T:
                raise ValueError(
                    f"to_geo() requires every response variable to share the same "
                    f"grid, but '{name}' has {np.asarray(p).shape[0]} points "
                    f"and {np.asarray(t).shape[0]} timestamps, versus {n} points and "
                    f"{T} timestamps for '{y_names[0]}'."
                )

        data = {
            "point_id": np.tile(np.arange(n), T),
            "timestamp": np.repeat(ts, n),
        }

        def add_columns(names, ys, sigmas, col_prefix):
            for name, y, sigma in zip(names, ys, sigmas):
                y = np.asarray(y)
                var = np.diagonal(np.asarray(sigma), axis1=0, axis2=1)  # (T, n)
                std = _safe_sqrt_variance(var, context="_build_geo_dataframe")
                data[f"y_{col_prefix}_{name}"] = y.T.ravel()  # (T, n) row-major: matches geoms below
                data[f"std_{col_prefix}_{name}"] = std.ravel()

        add_columns(y_names, y_list, Sigma_list, prefix)
        if y_back_list is not None:
            add_columns(y_names, y_back_list, Sigma_back_list, f"{prefix}_back")

        geoms = [Point(xy) for xy in points]

        return geopd.GeoDataFrame(
            data,
            geometry=np.tile(geoms, T),
            crs=crs,
        )

    def to_geo(self):
        """
        Package the in-sample fitted values and, if `.predict()` has been
        run, the out-of-sample prediction into GeoDataFrames, one row per
        (point, timestamp), ready to be exported (e.g. `.to_file("out.shp")`).

        Returns a dict with two keys:

        - `"hat"`: GeoDataFrame over the *training* grid, from
          `y_hat_list`/`Sigma_y_hat_list` (plus `y_hat_back_<var>`/
          `std_hat_back_<var>` if `.back_transform()` has been run). Always
          present.
        - `"pred"`: GeoDataFrame over the *prediction* grid, from
          `y_pred_list`/`Sigma_y_pred_list` (plus `y_pred_back_<var>`/
          `std_pred_back_<var>` if `.back_transform()` has been run), or
          `None` if `.predict()` hasn't been run yet.

        These are two separate GeoDataFrames, not one, because the training
        and prediction grids generally have a different number of
        points/timestamps.

        Columns: `point_id`, `timestamp`, then `y_<hat|pred>_<var>`/
        `std_<hat|pred>_<var>` for every response variable.

        results = results.predict(grid, verbose=True)
        geo = results.to_geo()
        geo["hat"].to_file("fitted.shp")
        geo["pred"].to_file("predictions.shp")
        """
        y_names = self.model.y_name

        gdf_hat = self._build_geo_dataframe(
            y_names, self.points_hat, self.timestamps_hat,
            self.y_hat_list, self.Sigma_y_hat_list, self.crs_hat,
            prefix="hat",
            y_back_list=self.y_hat_back_list, Sigma_back_list=self.Sigma_y_hat_back_list,
        )

        gdf_pred = None
        if self.y_pred_list is not None:
            gdf_pred = self._build_geo_dataframe(
                y_names, self.points_pred, self.timestamps_pred,
                self.y_pred_list, self.Sigma_y_pred_list, self.crs_pred,
                prefix="pred",
                y_back_list=self.y_pred_back_list, Sigma_back_list=self.Sigma_y_pred_back_list,
            )

        return {"hat": gdf_hat, "pred": gdf_pred}

    @staticmethod
    def _delta_method(g_inv, mu, Sigma):
        """
        Second-order delta method: back-transform a mean `mu` (shape
        `(n, T)`) and its full covariance `Sigma` (shape `(n, n, T)`,
        `Sigma[:, :, t]` symmetric) through `h = g_inv`, the inverse of the
        response transform applied by the model formula.

        `h` must be invertible with `h' != 0` everywhere it's evaluated
        (required for the linearization below to be valid) and is
        differentiated automatically via JAX autodiff, so `g_inv` only
        needs to be a plain scalar -> scalar JAX-traceable function (e.g.
        `jnp.exp` for `np.log(y)`) -- no analytic derivative required.

        Returns
        -------
        mean_back : ndarray, shape (n, T)
            E[h(Y)] ~= h(mu) + 0.5 * h''(mu) * Var(Y), a second-order Taylor
            expansion of h around mu (the first-order/naive term h(mu)
            alone is biased whenever h is curved).
        Sigma_back : ndarray, shape (n, n, T)
            Cov[h(Y)] ~= diag(h'(mu)) @ Sigma @ diag(h'(mu)), the standard
            (first-order) multivariate delta method, applied per time step;
            for the diagonal this is the familiar Var[h(Y)] ~= h'(mu)^2 * Var(Y).
        """
        mu = jnp.asarray(mu)
        Sigma = jnp.asarray(Sigma)
        flat_mu = mu.ravel()

        h = jax.vmap(g_inv)(flat_mu).reshape(mu.shape)
        h1 = jax.vmap(jax.grad(g_inv))(flat_mu).reshape(mu.shape)
        h2 = jax.vmap(jax.grad(jax.grad(g_inv)))(flat_mu).reshape(mu.shape)

        var_diag = jnp.diagonal(Sigma, axis1=0, axis2=1).T  # (T, n) -> (n, T)
        mean_back = h + 0.5 * h2 * var_diag
        Sigma_back = h1[:, None, :] * Sigma * h1[None, :, :]

        return np.asarray(mean_back), np.asarray(Sigma_back)

    @staticmethod
    def _direct_transform(g_inv, y):
        """
        Apply `g_inv` elementwise to an observed value `y` (shape `(n,
        T)`), with no delta-method correction -- unlike `_delta_method`,
        `y` here is a point (an actual observation), not the mean of an
        estimated distribution, so there is no variance to propagate.
        """
        y = jnp.asarray(y)
        h = jax.vmap(g_inv)(y.ravel()).reshape(y.shape)
        return np.asarray(h)

    def back_transform(self, g_invs):
        """
        Map `y_hat_list`/`y_obs_list`/`y_pred_list` back to the response's
        original scale, and store the results as `y_hat_back_list`/
        `Sigma_y_hat_back_list`, `y_obs_back_list`, and -- if `.predict()`
        has been run -- `y_pred_back_list`/`Sigma_y_pred_back_list`.
        Also derives `residuals_back_list` (`y_obs_back_list -
        y_hat_back_list`, elementwise per response variable) once both are
        available.

        `y_hat_list`/`y_pred_list` are the mean of an estimated
        distribution (with a model-implied covariance), so they go through
        the second-order delta method (see `_delta_method`), applied per
        response variable. `y_obs_list` is instead a point observation, so
        it is transformed directly (`g_inv(y_obs)`, see `_direct_transform`)
        -- there is no variance to propagate, and no meaningful way to
        transform `residuals_list` (`y_obs - y_hat`) directly, since
        `g_inv` of a residual is not itself a residual on the original
        scale; `residuals_back_list` is instead built by back-transforming
        `y_obs`/`y_hat` separately and then differencing, as is standard
        practice.

        This relies on the transformed response being asymptotically
        Normal (the model's own assumption), so the delta method's local,
        second-order Taylor expansion around the mean is a reasonable
        approximation -- but it is *not* the only option. Alternatives,
        roughly in order of how much they trade simplicity for accuracy:

        - Naive plug-in `g_inv(y_hat)`: what the first-order term alone
          gives; biased whenever `g_inv` is curved (Jensen's inequality),
          which is exactly what the second-order correction here fixes.
        
        - Closed-form formulas for a specific `g_inv`, when known -- e.g.
          the lognormal mean `exp(mu + sigma^2/2)` for `log`, which is the
          delta method's answer *and* the exact one for that case.

        - Duan's smearing estimator: replace `h(mu)` by the empirical
          average of `h(mu + residual_i)` over the in-sample residuals.
          Distribution-free (no normality needed) but needs those residuals
          kept around, and is usually applied to the mean only.
        
        - Gauss-Hermite quadrature / Monte Carlo against `Normal(mu, Var)`:
          numerically integrates `h` under the same normality assumption
          used here, so it stays exact as curvature or `Var` grows large
          (where a second-order Taylor expansion starts to break down),
          at the cost of a handful of extra evaluations of `h` per point.

        The delta method is the right default when `Var` is small relative
        to `h`'s curvature (the usual case for a well-identified model);
        if predictions look off for locations/times with large predictive
        variance, quadrature is the natural drop-in replacement since it
        reuses the same `mu`/`Sigma` this method already computes.

        Parameters
        ----------
        g_invs : Callable[[float], float] or list/tuple of callables
            The inverse of the response transform in the model formula
            (e.g. `jnp.exp` for a `np.log(y)` response), as a scalar ->
            scalar JAX-traceable function, or a list/tuple of such functions
            matching the number of response variables.
        """

        # check the len of g_invs and if it matches the number of response variables
        if isinstance(g_invs, (list, tuple)):
            if len(g_invs) != len(self.model.y_name):
                raise ValueError(
                    f"g_invs must be a single callable or a list/tuple of callables "
                    f"matching the number of response variables ({len(self.model.y_name)}), "
                    f"but got {len(g_invs)}."
                )
        else:
            g_invs = [g_invs] * len(self.model.y_name)


        if self.y_hat_list is not None and self.Sigma_y_hat_list is not None:
            y_hat_back_list, Sigma_hat_back_list = [], []
            for mu_i, Sigma_i, g_i in zip(self.y_hat_list, self.Sigma_y_hat_list, g_invs):
                m, S = self._delta_method(g_i, mu_i, Sigma_i)
                y_hat_back_list.append(m)
                Sigma_hat_back_list.append(S)

            self.y_hat_back_list = y_hat_back_list
            self.Sigma_y_hat_back_list = Sigma_hat_back_list

        if self.y_obs_list is not None:
            self.y_obs_back_list = [
                self._direct_transform(g_i, y_i) for g_i, y_i in zip(g_invs, self.y_obs_list)
            ]

        if self.y_obs_back_list is not None and self.y_hat_back_list is not None:
            self.residuals_back_list = [
                y_i - yhat_i
                for y_i, yhat_i in zip(self.y_obs_back_list, self.y_hat_back_list)
            ]

        if self.y_pred_list is not None:
            y_pred_back_list, Sigma_pred_back_list = [], []
            for mu_i, Sigma_i, g_i in zip(self.y_pred_list, self.Sigma_y_pred_list, g_invs):
                m, S = self._delta_method(g_i, mu_i, Sigma_i)
                y_pred_back_list.append(m)
                Sigma_pred_back_list.append(S)
            
            self.y_pred_back_list = y_pred_back_list
            self.Sigma_y_pred_back_list = Sigma_pred_back_list

        return self

    
    def generate_summary_header(self):
        """
        Cheap identity/fit rows: the base class's header (see
        `StateSpaceResults.generate_summary_header`) plus EM iteration
        count and AIC/BIC -- all pure attribute lookups/cheap arithmetic,
        none of `generate_summary()`'s residual diagnostics, coverage
        probability, or (if `.predict()` has run) prediction summary.
        Used by `__repr__` (inherited from the base class, via
        `format_info_table`).

        Returns
        -------
        gen_top_left, gen_top_right : list of (str, list)
            Two lists of `(label, [value])` rows, of equal length.
        """
        gen_top_left, gen_top_right = super().generate_summary_header()

        # aic/bic raise AttributeError before llf is set (i.e. before
        # LRStateSpaceModel.fit() has run) -- __repr__ must never raise,
        # so fall back to "N/A" rather than propagate that.
        
        if not hasattr(self, "aic") or not hasattr(self, "bic"):
            aic_bic = "N/A"
        else:
            aic_bic = f"{self.aic:.4g}, {self.bic:.4g}"

        gen_top_left = gen_top_left + [("EM iterations:", [f"{self.iterations}"])]
        gen_top_right = gen_top_right + [("AIC, BIC:", [aic_bic])]

        return gen_top_left, gen_top_right

    def generate_summary(self):
        """
        Build the left/right key-value rows used by `summary()`'s header
        table: the base class's rows (model info, fit quality) plus the EM
        iteration count/runtime and AIC/BIC, and -- if `.predict()` has
        been run -- a small summary of the out-of-sample predictions (see
        `_pred_summary_stats`).

        Returns
        -------
        gen_top_left, gen_top_right : list of (str, list)
            Two lists of `(label, [value])` rows, of equal length.
        """

        # update the computational time
        time_e = self.runtime_tot_estep
        time_m = self.runtime_tot_mstep

        # Generate the parent summary table (with model info). The base
        # class always returns left/right of equal length, so no padding
        # is needed here to keep the two columns aligned.
        gen_top_left, gen_top_right = super().generate_summary()

        # Add the EM iteration statistics table. Left mirrors the parent's
        # "execution" rows (log-likelihood, runtime); right mirrors its
        # "fit quality" rows (MSE, RMSE, ...).
        runtime_label, runtime_value = self._format_runtime_summary(
            [("E step", time_e), ("M step", time_m)]
        )
        top_left_em = {
            "EM iters :": lambda: [f"{self.iterations}"],
            runtime_label: lambda v=runtime_value: [v],
        }

        top_right_em = dict(
            [
                ("AIC:", lambda: [f"{self.aic:.4g}"]),
                ("BIC:", lambda: [f"{self.bic:.4g}"]),
            ]
        )

        # Generate the dictionaly
        gen_top_left_em = []
        for item in top_left_em.keys():
            gen_top_left_em.append((item, list(top_left_em[item]())))

        gen_top_right_em = []
        for item in top_right_em.keys():
            gen_top_right_em.append((item, list(top_right_em[item]())))


        
        gen_top_left += gen_top_left_em
        gen_top_right += gen_top_right_em

        # Add the out-of-sample prediction summary, if .predict() has been run
        if self.y_pred_list is not None:
            pstats = self._pred_summary_stats()

            top_left_pred = dict(
                [
                    ("Pred. points (values):", lambda: [f"{pstats['n_points']} ({pstats['n_pred']}, missing {pstats['n_missing']})"]),
                    ("Pred. y (min, med, max):", lambda: [f"{pstats['y_min']:.3g}, {pstats['y_median']:.3g}, {pstats['y_max']:.3g}"]),
                ]
            )

            top_right_pred = {
                "Pred. mean std:": lambda: [f"{pstats['y_mean_std']:.3g}"],
                "Pred. runtime (s):": lambda: [f"{self.tdelta_pred:.3g}"],
            }

            gen_top_left_pred = [(item, list(fn())) for item, fn in top_left_pred.items()]
            gen_top_right_pred = [(item, list(fn())) for item, fn in top_right_pred.items()]

            gen_top_left += gen_top_left_pred
            gen_top_right += gen_top_right_pred

        return gen_top_left, gen_top_right

    def summary(self, alpha=0.05):
        """
        Build a `statsmodels`-style textual report of the fitted model:
        the header table from `generate_summary` (model info, EM/fit
        diagnostics, AIC/BIC, and prediction summary if available),
        followed by one parameter table per group -- fixed effects
        (`beta`), measurement-equation parameters (`s2e`, `A`), and
        state-equation parameters (`ks`, `f`) -- each with point estimate,
        standard error, t-value, p-value, and `alpha`-level confidence
        interval.

        If `compute_cov_params()` has not been called yet, standard
        errors/t-values/p-values/confidence intervals are shown as NaN
        placeholders (see `_nan_bse_params`) rather than triggering the
        (potentially slow) Hessian computation implicitly.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for the parameter tables' confidence
            intervals (see `conf_int`).

        Returns
        -------
        statsmodels.iolib.summary.Summary
            Printable summary object (`str(...)`/`print(...)`).
        """
        name_width = 15

        # structured ModelParams with bse fields; NaN placeholders if
        # compute_cov_params() hasn't been called yet (see `bse`).
        bse_struct = self.bse

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
        smry.add_extra_txt([self._DIAGNOSTICS_NOTE])

        # Parameter names for every table below, padded to a common width so
        # the name column lines up across the (independently-sized) tables
        # produced by add_table_params.
        xnames_stack = [item for sublist in self.model.xbeta_names for item in sublist]
        meas_err_names = [f"s2e_{i}" for i in range(self.params.s2e.size)] + [
            f"A_{i}_{j}"
            for i in range(self.params.A.shape[0])
            for j in range(self.params.A.shape[1])
        ]
        matern_names = [f"rescale_{j}" for j in range(len(self.model.cov_function))]
        latent_names = [f"f_{i}" for i in range(self.params.f.value.size)]
        state_names = matern_names + latent_names

        max_name_len = max(
            [len(n) for n in xnames_stack + meas_err_names + state_names] + [name_width]
        )
        xnames_stack = [n.ljust(max_name_len) for n in xnames_stack]
        meas_err_names = [n.ljust(max_name_len) for n in meas_err_names]
        state_names = [n.ljust(max_name_len) for n in state_names]

        
        # fixed effect
        m = SimpleNamespace()
        m.results = np.array([0])  # Dummy results for compatibility
        m.model = None

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
        temp.params_name = meas_err_names

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
        temp = SimpleNamespace()
        temp.results = np.array([0])  # Dummy results for compatibility
        temp.model = None

        # Get the parameter values for this variable and assign them to the model namespace
        ks_val = np.asarray(self.params.ks.value).ravel()
        ks_bse = np.asarray(bse_struct.ks.bse).ravel()
        f_val = np.asarray(self.params.f.value).ravel()
        f_bse = np.asarray(bse_struct.f.bse).ravel()

        # Combine all parameters and names into a single list for the summary
        temp.param_names = state_names

        temp.params = np.hstack((ks_val, f_val))
        temp.bse = np.hstack((ks_bse, f_bse))
        temp.tvalues, temp.pvalues, _ = self._stats_from_arrays(temp.params, temp.bse)
        temp.conf_int = lambda alpha=alpha, v=temp.params, s=temp.bse: self._stats_from_arrays(v, s, alpha)[2]
        
        smry.add_table_params(temp, xname=temp.param_names, alpha=alpha)

        # Re-pad every rendered row to a common width so the top info table
        # and the parameter tables line up when printed together.
        smry.as_text = lambda _orig=smry.as_text: _equalize_row_widths(_orig())

        return smry
