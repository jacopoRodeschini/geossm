from statsmodels.iolib.summary import Summary
from typing import Any, Optional
import numpy as np
from geossm.stmodel.param import Param, ModelParams, FitOptions
from geossm.ssm.statespace_results import StateSpaceResults, _safe_sqrt_variance
from geossm.ssm.statespace import _highest_matmul_precision
from types import SimpleNamespace
from scipy import stats
import time
import warnings
import jax
import jax.numpy as jnp
from dataclasses import dataclass, replace, fields
from geossm.utils import _on_device, split_by_block, format_info_table, _summary_table_source
from geossm.utils import _print_info
from geossm.stmodel._fitted import FittedState, _read_only_copy, observed_loglik, predict_from_fit
from geossm.stmodel._output import build_geo_dataframe, delta_method, diag_variance, direct_transform, normalize_g_invs
from geossm.stmodel.prediction_results import LRPredictionResults


ArrayLike = Optional[Any]

# %% [Utils] Define pack/unpack helpers to flatten params to a 1D array

# Scale of the confidence intervals of bounded parameters (see
# `_stats_from_arrays`): log for the positive ones, atanh for f.
_CI_SCALE = {"s2e": "log", "ks": "log", "f": "atanh"}


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


def _state_property(name, doc):
    """Read-only view of the `FittedState` field `name` (see
    `LRStateSpaceResults.fitted_state`)."""
    def get(self):
        state = getattr(self, "_fitted_state", None)
        return None if state is None else getattr(state, name)
    return property(get, doc=doc)


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
      Hessian, via `jax.hessian` of the observed log-likelihood), `bse`,
      `tvalues`, `pvalues`, `conf_int`, `aic`/`bic`.
    - Out-of-sample prediction: `predict` (mean and point-wise variance as
      arrays) and `get_prediction` (an `LRPredictionResults` object with the
      covariance, back-transformation, export and change of support to
      polygons, `to_poly`). Prediction does not modify these results.
    - Back-transformation of the fitted values to the response's original
      scale via the delta method (`back_transform`), when the model formula
      applies a transform to the response (e.g. `np.log`).
    - Export of the fitted values to a `geopandas.GeoDataFrame` (`to_geo`).
    - A `statsmodels`-style textual report (`summary`).

    A results object is a snapshot of the fit that produced it and does not
    reference the model: the configuration and the estimates of that fit are
    recorded in `fitted_state` (a `FittedState`) when it is created, and the
    Hessian, predictions and summaries are computed from it, so a later
    `fit()` or `setup()` on the model does not change it (the new fit
    returns its own results). The covariance objects and formula builders
    are kept by reference and should not be re-`setup()` afterwards.

    Per-variable ("_list") views: `y_hat_list`/`Sigma_y_hat_list`
    (in-sample fitted), `y_obs_list` (training observations),
    and `residuals_list` are lazy `@property` views splitting the
    corresponding stacked array (`y_hat`, `y_obs`, `residuals`) one entry
    per response variable. After `.back_transform()`, their original-scale
    counterparts `y_hat_back_list`, `y_obs_back_list` and
    `residuals_back_list` are plain attributes (genuine computations, not
    free slices of a stacked array).

    Typical usage, continuing from `LRStateSpaceModel.fit()`::

        results = model.fit()
        results.compute_cov_params()      # optional, needed for bse/t/p/CI
        print(results.summary())
        y_pred, var_pred = results.predict(new_df)   # arrays
        pred = results.get_prediction(new_df)       # LRPredictionResults
        pred.back_transform(jnp.exp)                # if y = np.log(...)
        geo = pred.to_geo()
        areal = pred.to_poly("regions.shp", std=True)
    """

    def __init__(
        self,
        model=None,
        params: ModelParams = None,
        nstats: list = None,
        options: FitOptions = None,
        convergence: bool = None,
        termination: str = None,
        fitted_state=None,
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
            The model instance that produced these results; read once, for
            the defaults of the base class (see `StateSpaceResults`), and
            not stored.
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
        convergence : bool, optional
            `True` if the EM algorithm stopped because the per-observation
            log-likelihood change was within `tol_relat`; `False` if it
            stopped for any other reason (see `termination`).
        termination : str, optional
            Why the EM algorithm stopped: ``"tolerance"`` (converged; this
            includes a decrease within `tol_relat`, i.e. the iterates
            oscillating on the plateau), ``"decrease"`` (a larger decrease
            with `force_iter=False`), ``"invalid"`` (NaN/Inf
            log-likelihood) or ``"max_iter"``. On ``"decrease"``,
            ``"invalid"`` and a converged decrease, the highest-likelihood
            iterate seen is returned.
        fitted_state : FittedState, optional
            Configuration snapshot of the model (`_get_fitted_state()`),
            completed here with the estimates (`params`, `x_smoothed`,
            `P_smoothed`, `timestamps_hat`); required by the Hessian,
            `predict` and the summary.
        block_p : array-like, optional
            Cumulative index boundaries splitting the stacked training
            arrays into one block per response variable; used only without
            a `fitted_state` (otherwise `block_p` is derived from
            `fitted_state.pdim`).
        points_hat, timestamps_hat, crs_hat : list / list / CRS, optional
            Training grid (sites, timestamps) and CRS, one entry per
            response variable. The timestamps are recorded in the fitted
            state; `points_hat`/`timestamps_hat` are then read-only views of
            it (used as given only without a `fitted_state`).
        **kwargs
            Forwarded to `StateSpaceResults.__init__` (e.g. `y_hat`,
            `Sigma_y_hat`, `llf`, ...). The latent state (`x_smoothed`,
            `P_smoothed`, `P_pred_smoothed`, `S11`/`S10`/`S00`) is normally in
            `fitted_state`; arrays passed here replace its fields.
        """
        # The arrays of the fit are held by the fitted state, not here (see
        # `_DELEGATED_FIELDS`): those passed (and the model's training data,
        # by default) are added to it below.
        if model is not None:
            for name, attr in (("y_obs", "y_t"), ("Xbeta", "Xbeta")):
                if kwargs.get(name) is None:
                    kwargs[name] = getattr(model, attr, None)
        delegated = {k: kwargs.pop(k) for k in list(kwargs)
                     if k in self._DELEGATED_FIELDS and kwargs[k] is not None}
        kwargs.setdefault("y_hat", None)  # required by the base class; held by the fitted state

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
        self.convergence = convergence
        self.termination = termination
        # Fit-time state (configuration from the model, completed with the
        # estimates below): everything computed after fitting depends on it,
        # not on the model (F09, P1); see `FittedState`.
        self._fitted_state = fitted_state

        # Cumulative per-response-variable index boundaries for the
        # stacked training arrays (`y_hat`, `Sigma_y_hat`, `y_obs`,
        # `residuals`); snapshotted here (not read live off the model's
        # `block_p`) since that attribute is mutable and would
        # go stale for this results object the next time `fit()`/`setup()`
        # runs on this model instance. Drives the `y_hat_list`/
        # `Sigma_y_hat_list`/`y_obs_list`/`residuals_list` properties below.
        self._block_p = block_p  # fallback without a fitted state; see `block_p`

        # Training grid (one entry per response variable), snapshotted for
        # the same reason as block_p above -- used by `.to_geo()`.
        self._points_hat = points_hat
        self._timestamps_hat = timestamps_hat
        self.crs_hat = crs_hat
        self._complete_fitted_state()
        if delegated and self._fitted_state is not None:
            self._fitted_state = replace(self._fitted_state, **delegated)
        self._set_observation_counts()  # y_obs is now available from the fitted state

        # ---- Derived quantities (initialized empty) ----
        self.param_names = None
        self.param_values = None
        self.param_dim = None

        self.iterations = 0
        self.runtime_tot_estep = 0.0
        self.runtime_tot_mstep = 0.0

        # Original-scale (back-transformed) counterparts of y_hat_list/
        # y_obs_list/residuals_list, populated by
        # .back_transform() -- see that method's docstring. Per-variable
        # views only (no stacked *_back counterparts): unlike the plain
        # `_list` views, these are genuine computations (delta method, or
        # a direct transform), not free slices of a stacked array, so they
        # are plain attributes rather than properties.
        self.y_hat_back_list = None
        self.Sigma_y_hat_back_list = None
        self.y_obs_back_list = None
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
        `beta`, `s2e`, `f`, `A`, `ks`, used as the degrees of freedom /
        penalty term `k` in `df_resid`, `compute_aic`, and `compute_bic`.
        The initial state (`x0`, `Sigma0`) is a nuisance parameter and is
        not counted, whether fixed or estimated (as in the Hessian, see
        `_inference_params`).
        """
        if self._n_params is None:
            self._n_params = sum(d for n, d, f in zip(self.param_names, self.param_dim, self.param_fixed)
                                 if not f and n not in ("x0", "Sigma0")) if self.param_dim is not None and self.param_fixed else 0
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
        
        self.llf_path = [float(v["logL"]) for v in self.nstats]
        # fit() passes the llf of the returned estimate, which is not the
        # last logged one when an earlier best iterate is restored.
        if self.llf is None:
            self.llf = self.llf_path[-1]
    
    def _after_update(self, keys):
        """`update` hook: also clear the inference caches and reprocess
        `params`/`nstats` when they are replaced."""
        super()._after_update(keys)
        self._n_params = self._aic = self._bic = None
        self._hessian = self._cov_params = self._free_meta = self._frozen_params = None
        self._bse_params = None
        if "params" in keys and self.params is not None:
            self._process_params()
        if "params" in keys:
            self._complete_fitted_state()
        if "nstats" in keys and self.nstats is not None and "llf" not in keys:
            self.llf = None  # taken from the new nstats
            self._process_nstats()

    def _snapshot(self, name):
        """Fit-time value `name` of the model state (see `FittedState`)."""
        return getattr(self._fitted_state, name)

    def _complete_fitted_state(self):
        """Add `params` (the same object) and the training timestamps
        (read-only copies) of these results to the fitted state built by
        the model, which already holds the latent state."""
        state = getattr(self, "_fitted_state", None)
        if state is None:
            return
        self._fitted_state = replace(
            state,
            params=self.params,
            timestamps_hat=(state.timestamps_hat if self._timestamps_hat is None
                            else tuple(_read_only_copy(t) for t in self._timestamps_hat)),
        )

    def _update_delegated(self, values):
        """`update` of latent-state fields: replaced in the fitted state
        (converted to read-only arrays in the results' dtype)."""
        self._fitted_state = replace(self._fitted_state, **values)

    @property
    def fitted_state(self):
        """FittedState: State of the model at the end of the fit -- the
        configuration (covariance functions, formula builders, sites, block
        sizes, names, dtypes) and the estimates (`params`, smoothed states
        `x_smoothed` and covariances `P_smoothed`, lag-one covariances,
        sufficient statistics `S11`/`S10`/`S00`, training timestamps).
        Read-only. The latent-state attributes of these results are views of
        it, and the Hessian, predictions and summaries are computed from it."""
        return self._fitted_state

    # The training grid lives in the fitted state (one source, read-only).

    @property
    def points_hat(self):
        """list of ndarray: Training sites of each response variable
        (read-only, from `fitted_state.points`)."""
        state = self._fitted_state
        return self._points_hat if state is None else list(state.points)

    @property
    def timestamps_hat(self):
        """list of ndarray: Training timestamps of each response variable
        (read-only, from `fitted_state.timestamps_hat`)."""
        state = self._fitted_state
        if state is None or state.timestamps_hat is None:
            return self._timestamps_hat
        return list(state.timestamps_hat)

    @property
    def block_p(self):
        """ndarray: Row boundaries of each response variable in the stacked
        training arrays (`block_p[i]:block_p[i+1]` selects variable `i`),
        derived from `fitted_state.pdim`; read-only NumPy array."""
        state = self._fitted_state
        if state is None:
            return self._block_p
        bp = np.concatenate([[0], np.cumsum(np.asarray(state.pdim))])
        bp.setflags(write=False)
        return bp

    # The arrays of the fit live in the fitted state; these read-only
    # properties keep the results' attribute API (`results.y_hat`, ...).
    _DELEGATED_FIELDS = frozenset(FittedState.ARRAY_FIELDS)
    y_obs = _state_property("y_obs", "ndarray, shape (p, T): Training observations (NaN where missing); see `fitted_state`.")
    Xbeta = _state_property("Xbeta", "ndarray, shape (p, b, T): Training design matrix; see `fitted_state`.")
    y_hat = _state_property("y_hat", "ndarray, shape (p, T): In-sample fitted mean; see `fitted_state`.")
    Sigma_y_hat = _state_property(
        "Sigma_y_hat", "ndarray, shape (p, p, T), or None: In-sample state-driven covariance; see `fitted_state`.")
    x_smoothed = _state_property(
        "x_smoothed", "ndarray, shape (q, T+1): Smoothed state means (index 0: initial state); see `fitted_state`.")
    P_smoothed = _state_property(
        "P_smoothed", "ndarray, shape (q, q, T+1): Smoothed state covariances; see `fitted_state`.")
    P_pred_smoothed = _state_property(
        "P_pred_smoothed", "ndarray or None: Smoothed lag-one state covariances; see `fitted_state`.")
    S11 = _state_property("S11", "ndarray, shape (q, q): EM sufficient statistic S11; see `fitted_state`.")
    S10 = _state_property("S10", "ndarray, shape (q, q): EM sufficient statistic S10; see `fitted_state`.")
    S00 = _state_property("S00", "ndarray, shape (q, q): EM sufficient statistic S00; see `fitted_state`.")

    @property
    def _y_names(self):
        """Response-variable names of the fit (snapshot, see `_snapshot`)."""
        return self._snapshot("y_name")

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
        logL = observed_loglik(self._fitted_state, self.y_obs, self.Xbeta, x0, Sigma0)
        # fun(params)

        # Pack free params into a flat array
        vec0, meta = _pack_params(params, dtype=self.dtype)
        self._free_meta = meta

        
        # Wrap fun to accept a flat vector
        def fun_flat(vec):
            p = _unpack_params(vec, params, meta)
            return logL(p)

        ts = time.time()
        # Compute the Hessian using JAX, of the observed log-likelihood function at the argument 0 (params).
        # Jitted as one program (instead of ~265 eagerly compiled operations),
        # at full matmul precision like the other kernels (F21, F31).
        hesfun = jax.jit(_highest_matmul_precision(jax.hessian(fun_flat)))
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

        _print_info("Computing Hessian and standard errors of the parameters")
        H, tdelta = self._compute_hessian()
        _print_info(f"Hessian computed in {tdelta:.3f} seconds")

        # Compute the covariance matrix as the inverse of the Hessian (of the positive log-likelihood)
        # Sigma = -jnp.linalg.pinv(H)
        # cov_params = -jnp.linalg.inv(H)
        H = 0.5 * (H + H.T)
        # eps = 1e-6
        # H += eps * jnp.eye(H.shape[0], dtype=H.dtype)

        # At a maximum the Hessian of the log-likelihood is negative definite;
        # otherwise -H^-1 is not a covariance matrix and no SE is reported.
        eig = np.asarray(jnp.linalg.eigvalsh(H))
        n_bad = int(np.sum(~(eig < 0)))
        if n_bad:
            warnings.warn(
                "The Hessian of the log-likelihood is not negative definite "
                "({} of {} eigenvalues are non-negative or not finite; largest {:.3g}), "
                "so the estimate is not at a strict maximum or the Hessian is "
                "inaccurate. Standard errors are set to NaN.".format(
                    n_bad, eig.size, float(np.max(eig)) if eig.size else float("nan")),
                RuntimeWarning,
                stacklevel=2,
            )
            cov_params = jnp.full(H.shape, jnp.nan, dtype=H.dtype)
        else:
            cov_params = -jnp.linalg.solve(H, jnp.eye(H.shape[0], dtype=H.dtype))

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
        var = np.asarray(jnp.diag(self._cov_params))
        if np.all(np.isnan(var)):  # not a covariance matrix: compute_cov_params warned
            return np.full(var.shape, np.nan)
        return _safe_sqrt_variance(var, context="bse_vector")

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

    def _stats_from_arrays(self, params, bse, alpha=0.05, scale=""):
        """Compute t-values, two-sided p-values, and `alpha`-level confidence
        intervals for flat `params`/`bse` arrays, using the Student-t
        distribution with `df_resid` degrees of freedom (or the Normal
        distribution if `df_resid` is unavailable).

        `scale` (a string, or one per element of `params`) sets the scale
        of the interval; it is computed there, with the standard error from
        the delta method, and mapped back:

        * `""`: natural scale, `theta -+ crit * bse`;
        * `"log"`: `log(theta)`, standard error `bse / theta`, giving
          `theta * exp(-+ crit * bse / theta)` (positive);
        * `"atanh"`: `atanh(theta)`, standard error `bse / (1 - theta**2)`,
          giving `tanh(atanh(theta) -+ crit * bse / (1 - theta**2))`, inside
          (-1, 1). For `|theta| >= 1` atanh is undefined and the natural
          scale is used.

        t-values and p-values are always on the natural scale."""
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
    
        lo, hi = params - crit * bse, params + crit * bse
        scale = np.broadcast_to(np.asarray(scale, dtype=str), params.shape)
        log, ath = scale == "log", (scale == "atanh") & (np.abs(params) < 1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = np.exp(crit * bse / params)
            z, dz = np.arctanh(params), crit * bse / (1 - params**2)
            lo = np.where(log, params / r, np.where(ath, np.tanh(z - dz), lo))
            hi = np.where(log, params * r, np.where(ath, np.tanh(z + dz), hi))
        return t, p, np.column_stack([lo, hi])

    def _ci_scale(self):
        """Confidence-interval scale of each free parameter, in the order of
        `theta_hat` (`_CI_SCALE`, `""` for the natural scale)."""
        _, meta = _pack_params(self._inference_params(), dtype=self.dtype)
        scale = np.full(self.theta_hat.size, "", dtype="<U5")
        for name, _shape, start, stop, _fixed in meta:
            scale[start:stop] = _CI_SCALE.get(name, "")
        return scale

       
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

        Wald intervals `theta +- crit * bse`, except for the bounded
        parameters, whose intervals are computed on a transformed scale and
        mapped back (delta method, see `_stats_from_arrays`): log scale for
        `s2e` and `ks`, so they stay positive, and atanh scale for `f`, so
        it stays inside (-1, 1) (natural scale if `|f| >= 1`).

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
        _, _, ci = self._stats_from_arrays(self.theta_hat, self.bse_vector, alpha=alpha,
                                           scale=self._ci_scale())
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

    @_on_device
    def predict(self, df, variance=True, verbose=True):
        """
        Predicted mean and point-wise variance of the response at new
        locations/times, as arrays (see `get_prediction` for an object with
        the full covariance, per-variable views, back-transformation and
        export).

        Computed from the fitted state (`fitted_state`: smoothed states,
        parameters, covariance functions and formula builders of the fit;
        see `geossm.stmodel._fitted.predict_from_fit`): mean `H x_t + X
        beta` and variance `diag(H P_t H')`, the state-driven uncertainty
        without the measurement error. The variance is computed without
        building the `(p, p, T)` covariance. These results are not modified.

        Parameters
        ----------
        df : geopandas.GeoDataFrame or list of geopandas.GeoDataFrame
            New locations/times (a subset of the training timestamps), with
            the covariates of the formulas and the training CRS; one per
            formula to predict each response variable at its own sites.
        variance : bool, default True
            Also compute the point-wise variance.
        verbose : bool, default True
            Verbosity of the design-matrix builders. Progress messages follow
            the model's verbosity at the end of the fit.

        Returns
        -------
        y_pred : numpy.ndarray, shape (p, T)
            Predicted mean, response variables stacked as in the fit.
        var_pred : numpy.ndarray, shape (p, T), or None
            Point-wise predictive variance (`None` if `variance=False`).
        """
        state = self._fitted_state
        log = _print_info if state.verbose else None
        pred = predict_from_fit(state, df, verbose=verbose, cov="diag" if variance else None, log=log)
        return pred["y_pred"], pred["var_pred"]

    @_on_device
    def get_prediction(self, df, std=True, verbose=True):
        """
        Out-of-sample prediction at new locations/times, as an
        `LRPredictionResults` object: predicted mean, full predictive
        covariance (if `std`) and point-wise variance, prediction grid,
        per-variable views, `back_transform`, `to_geo`, and the change of
        support `to_poly`. These results are not modified.

        Parameters
        ----------
        df : geopandas.GeoDataFrame or list of geopandas.GeoDataFrame
            New locations/times; see `predict`.
        std : bool, default True
            Compute the predictive covariance `Sigma_y_pred` `(p, p, T)`
            (needed by `to_poly(std=True)` and the back-transformed
            covariance). If `False`, only the mean is computed -- useful when
            the covariance does not fit in memory (`predict` gives the
            point-wise variance without it).
        verbose : bool, default True
            Verbosity of the design-matrix builders.

        Returns
        -------
        LRPredictionResults
        """
        state = self._fitted_state
        log = _print_info if state.verbose else None
        pred = predict_from_fit(state, df, verbose=verbose, cov="full" if std else None, log=log)
        return LRPredictionResults(
            state, pred["y_pred"], Sigma_y_pred=pred["Sigma_y_pred"], var_pred=pred["var_pred"],
            points=pred["points"], block_p=pred["block_p"], timestamps=pred["timestamps"],
            crs=pred["crs"], tdelta=pred["tdelta"],
        )

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
    #
    # Original-scale counterparts (y_hat_back_list, y_obs_back_list,
    # residuals_back_list, ...) are genuine computations
    # (delta method / direct transform) rather than free slices, so they
    # are plain attributes populated by `back_transform()` instead.

    @property
    def y_hat_list(self):
        """list of ndarray : Per-response-variable view of the base
        class's stacked `y_hat` (in-sample fitted mean), split along
        `block_p`."""
        if self.y_hat is None or self.block_p is None:
            return None
        return split_by_block(self.y_hat, self.block_p)

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

    def to_geo(self):
        """
        The in-sample fitted values as a GeoDataFrame over the training
        grid, one row per (site, timestamp), ready to be exported (e.g.
        `.to_file("fitted.shp")`). Predictions are exported by
        `LRPredictionResults.to_geo` (see `get_prediction`).

        Columns: `point_id`, `timestamp`, then `y_hat_<var>`/`std_hat_<var>`
        for every response variable (plus `y_hat_back_<var>`/
        `std_hat_back_<var>` after `.back_transform()`), in the dtype of the
        fit. `std_*` is the standard deviation of the latent mean
        (`sqrt(diag(H P H'))`, plug-in parameters), without the measurement
        error: for a prediction interval of a new observation, add `s2e` to
        its square (as `conf_int_y(prediction=True)` does).

        Returns
        -------
        geopandas.GeoDataFrame
        """
        var_list = None if self.Sigma_y_hat_list is None else [diag_variance(S) for S in self.Sigma_y_hat_list]
        back_var = (None if self.Sigma_y_hat_back_list is None
                    else [diag_variance(S) for S in self.Sigma_y_hat_back_list])
        return build_geo_dataframe(
            self._y_names, self.points_hat, self.timestamps_hat, self.y_hat_list, var_list, self.crs_hat,
            prefix="hat", y_back_list=self.y_hat_back_list, var_back_list=back_var,
        )

    def back_transform(self, g_invs):
        """
        Map `y_hat_list`/`y_obs_list` back to the response's original scale,
        and store the results as `y_hat_back_list`/`Sigma_y_hat_back_list`
        and `y_obs_back_list`. Also derives `residuals_back_list`
        (`y_obs_back_list - y_hat_back_list`, elementwise per response
        variable) once both are available. Predictions are back-transformed
        by `LRPredictionResults.back_transform` (see `get_prediction`).

        `y_hat_list` is the mean of an estimated
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

        What is estimated: `mu` and `Sigma` are the plug-in mean and the
        state-driven covariance of the latent signal (`Sigma_y_hat`, and
        `Sigma_y_pred` for predictions, `H P H'`, without the measurement error `s2e` and
        with the parameters treated as known). `y_*_back_list` therefore
        approximates the original-scale mean of the signal, `E[h(signal)]`,
        not of a new noisy observation (which would add `s2e` to the
        variance in the second-order term), and `Sigma_*_back_list` its
        covariance. `residuals_back_list` is `h(y_obs) - E[h(signal)]`.

        A fitted or predicted mean without its covariance (`std=False`)
        cannot be back-transformed this way: it is left as `None`, with a
        warning.

        Parameters
        ----------
        g_invs : Callable[[float], float] or list/tuple of callables
            The inverse of the response transform in the model formula
            (e.g. `jnp.exp` for a `np.log(y)` response), as a scalar ->
            scalar JAX-traceable function, or a list/tuple of such functions
            matching the number of response variables.

        Returns
        -------
        LRStateSpaceResults
            `self`, with the `*_back_list` attributes set.
        """

        g_invs = normalize_g_invs(g_invs, len(self._y_names))

        if self.y_hat_list is not None and self.Sigma_y_hat_list is not None:
            y_hat_back_list, Sigma_hat_back_list = [], []
            for mu_i, Sigma_i, g_i in zip(self.y_hat_list, self.Sigma_y_hat_list, g_invs):
                m, S = delta_method(g_i, mu_i, Sigma_i)
                y_hat_back_list.append(m)
                Sigma_hat_back_list.append(S)

            self.y_hat_back_list = y_hat_back_list
            self.Sigma_y_hat_back_list = Sigma_hat_back_list
        elif self.y_hat_list is not None:
            warnings.warn("back_transform(): the fitted values have no covariance (Sigma_y_hat), "
                          "so they are not back-transformed.", RuntimeWarning, stacklevel=2)

        if self.y_obs_list is not None:
            self.y_obs_back_list = [
                direct_transform(g_i, y_i) for g_i, y_i in zip(g_invs, self.y_obs_list)
            ]

        if self.y_obs_back_list is not None and self.y_hat_back_list is not None:
            self.residuals_back_list = [
                y_i - yhat_i
                for y_i, yhat_i in zip(self.y_obs_back_list, self.y_hat_back_list)
            ]

        return self

    
    def generate_summary_header(self):
        """
        Cheap identity/fit rows: the base class's header (see
        `StateSpaceResults.generate_summary_header`) plus EM iteration
        count and AIC/BIC -- all pure attribute lookups/cheap arithmetic,
        none of `generate_summary()`'s residual diagnostics or coverage
        probability.
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

        gen_top_left = gen_top_left + [
            ("EM iterations:", [f"{self.iterations}"]),
            ("Converged:", [f"{self.convergence}"]),
        ]
        gen_top_right = gen_top_right + [
            ("AIC, BIC:", [aic_bic]),
            ("", [""]),
        ]

        return gen_top_left, gen_top_right

    def generate_summary(self):
        """
        Build the left/right key-value rows used by `summary()`'s header
        table: the base class's rows (model info, fit quality) plus the EM
        iteration count/runtime and AIC/BIC.

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
            _summary_table_source(),
            title="LR State Space Model Results",
            gleft=gen_top_left,
            gright=gen_top_right,
            yname=None,
            xname=None,
        )
        smry.add_extra_txt(list(self._DIAGNOSTICS_NOTE))

        # Parameter names for every table below, padded to a common width so
        # the name column lines up across the (independently-sized) tables
        # produced by add_table_params.
        xnames_stack = [item for sublist in self._snapshot("xbeta_names") for item in sublist]
        meas_err_names = [f"s2e_{i}" for i in range(self.params.s2e.size)] + [
            f"A_{i}_{j}"
            for i in range(self.params.A.shape[0])
            for j in range(self.params.A.shape[1])
        ]
        matern_names = [f"rescale_{j}" for j in range(len(self._snapshot("cov_function")))]
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
        scale = ["log"] * s2e_vals.size + [""] * A_vals.size
        temp.tvalues, temp.pvalues, _ = self._stats_from_arrays(temp.params, temp.bse)
        temp.conf_int = lambda alpha=alpha, v=temp.params, s=temp.bse, m=scale: self._stats_from_arrays(v, s, alpha, m)[2]
        
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
        scale = ["log"] * ks_val.size + ["atanh"] * f_val.size
        temp.tvalues, temp.pvalues, _ = self._stats_from_arrays(temp.params, temp.bse)
        temp.conf_int = lambda alpha=alpha, v=temp.params, s=temp.bse, m=scale: self._stats_from_arrays(v, s, alpha, m)[2]
        
        smry.add_table_params(temp, xname=temp.param_names, alpha=alpha)

        # Re-pad every rendered row to a common width so the top info table
        # and the parameter tables line up when printed together.
        smry.as_text = lambda _orig=smry.as_text: _equalize_row_widths(_orig())

        return smry
