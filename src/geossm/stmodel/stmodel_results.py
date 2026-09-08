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
from geossm.utils import _on_device


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
    Results container for LR State Space estimation.
    """

    def __init__(
        self,
        model=None,
        params: ModelParams = None,
        nstats: list = None,
        options: FitOptions = None,
        y_hat_list: list = None,
        Sigma_y_hat_list: list = None,
        block_p=None,
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

        # Per-variable views (one entry per response variable, i.e. length
        # `model.nvar`) of the base class's stacked `y_hat`/`Sigma_y_hat`,
        # split along `block_p` -- see `_split_by_block`. `block_p` is
        # snapshotted here (not read live off `self.model.block_p`) since
        # that attribute is mutable and would go stale for this results
        # object the next time `fit()`/`setup()` runs on this model instance.
        self.y_hat_list = y_hat_list
        self.Sigma_y_hat_list = Sigma_y_hat_list
        self.block_p = block_p
        self._residuals_list = None  # cache for the residuals_list property

        # ---- Derived quantities (initialized empty) ----
        self.param_names = None
        self.param_values = None
        self.param_dim = None

        self.iterations = 0
        self.runtime_tot = 0.0
        self.runtime_tot_estep = 0.0
        self.runtime_tot_mstep = 0.0

        # Out-of-sample prediction (populated by .predict(); per-variable
        # views, distinct from `y_hat`/`y_hat_list`, which hold the
        # in-sample filtered/fitted values used for residuals).
        self.points_pred = None
        self.y_pred_list = None
        self.Sigma_y_pred_list = None
        self.tdelta_pred = None
        self.timestamps_pred = None
        self.crs_pred = None

        # Original-scale (back-transformed) counterparts of y_hat_list/
        # y_pred_list, populated by .back_transform() -- see that method's
        # docstring. Per-variable views only, mirroring y_hat_list/y_pred_list
        # (no stacked y_hat_back/Sigma_y_hat_back).
        self.y_hat_back_list = None
        self.Sigma_y_hat_back_list = None
        self.y_pred_back_list = None
        self.Sigma_y_pred_back_list = None

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
        vec, _ = _pack_params(self._inference_params(), dtype=self.dtype)
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
        Compute out-of-sample predictions based on smoothed states and model
        parameters. `self.model.predict` stores them directly on this
        results object (`points_pred`, `y_pred_list`, `Sigma_y_pred_list`,
        `tdelta_pred`, `timestamps_pred`, `crs_pred` -- one entry per
        response variable) and returns `self`, so predictions travel with
        the fitted model and can be reused by other methods (e.g.
        `.to_geo()`, plotting, summaries) without re-running prediction.
        """
        return self.model.predict(df, modelresults=self, verbose=verbose)


    @property
    def residuals_list(self):
        """
        Per-variable view of the base class's stacked `residuals`
        (`y_obs - y_hat`), split along `block_p` -- the in-sample
        counterpart of `y_hat_list`/`Sigma_y_hat_list`.
        """
        if self._residuals_list is None:
            res = self.residuals
            if res is None or self.block_p is None:
                return None
            block_p = np.asarray(self.block_p)
            self._residuals_list = [
                res[block_p[i]:block_p[i + 1], :] for i in range(len(block_p) - 1)
            ]
        return self._residuals_list

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
            var = np.clip(np.diagonal(sigma, axis1=0, axis2=1), 0.0, None)  # (T, n_i)
            std_all.append(np.sqrt(var).ravel())
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

    def to_geo(self):
        """
        Package the last computed prediction (`.predict()`) into a single
        GeoDataFrame, one row per (point, timestamp), ready to be exported
        (e.g. `.to_file("out.shp")`).

        Columns: `point_id`, `timestamp`, then `y_pred_<var>`/`std_pred_<var>`
        for every response variable (predicted mean, and predictive standard
        deviation from the diagonal of `Sigma_y_pred_list`), plus, if
        `.back_transform()` has been run, `y_pred_back_<var>`/
        `std_pred_back_<var>` (original-scale counterparts). Geometry is the
        prediction point, repeated once per timestamp; CRS is `self.crs_pred`.

        Only prediction-grid quantities (`y_pred_list`/`Sigma_y_pred_list`
        and their back-transformed counterparts) are included here --
        `y_hat_list`/`Sigma_y_hat_list` live on the *training* grid, which
        generally has a different number of points/timestamps than the
        prediction grid this GeoDataFrame is indexed by.

        All response variables must share the same prediction grid (same
        points and timestamps) -- true whenever they were all predicted from
        the same input dataframe, as `.predict()` guarantees.

        results = results.predict(grid, verbose=True)
        gdf = results.to_geo()
        gdf.to_file("predictions.shp")

        """
        if self.y_pred_list is None:
            raise ValueError(
                "No prediction available: call `.predict(df)` before `.to_geo()`."
            )

        import geopandas as geopd
        from shapely.geometry import Point

        y_names = self.model.y_name
        points = np.asarray(self.points_pred[0])
        ts = self.timestamps_pred[0]
        n, T = points.shape[0], ts.shape[0]

        for name, p, t in zip(y_names, self.points_pred, self.timestamps_pred):
            if np.asarray(p).shape[0] != n or np.asarray(t).shape[0] != T:
                raise ValueError(
                    f"to_geo() requires every response variable to share the same "
                    f"prediction grid, but '{name}' has {np.asarray(p).shape[0]} points "
                    f"and {np.asarray(t).shape[0]} timestamps, versus {n} points and "
                    f"{T} timestamps for '{y_names[0]}'."
                )

        data = {
            "point_id": np.tile(np.arange(n), T),
            "timestamp": np.repeat(ts, n),
        }

        for name, y, sigma in zip(y_names, self.y_pred_list, self.Sigma_y_pred_list):
            y = np.asarray(y)
            var = np.clip(np.diagonal(np.asarray(sigma), axis1=0, axis2=1), 0.0, None)  # (T, n)
            std = np.sqrt(var)

            data[f"y_pred_{name}"] = y.T.ravel()  # (T, n) row-major: matches geoms below
            data[f"std_pred_{name}"] = std.ravel()

        if self.y_pred_back_list is not None:
            for name, y_pred, sigma_pred in zip(
                y_names, self.y_pred_back_list, self.Sigma_y_pred_back_list
            ):
                var_pred = np.clip(np.diagonal(np.asarray(sigma_pred), axis1=0, axis2=1), 0.0, None)  # (T, n)
                std_pred = np.sqrt(var_pred)

                data[f"y_pred_back_{name}"] = np.asarray(y_pred).T.ravel()
                data[f"std_pred_back_{name}"] = std_pred.ravel()

        geoms = [Point(xy) for xy in points]


        return geopd.GeoDataFrame(
            data,
            geometry=np.tile(geoms, T),
            crs=self.crs_pred,
        )

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

    def back_transform(self, g_inv):
        """
        Map `y_hat_list`/`y_pred_list` (and their model-implied covariance)
        back to the response's original scale, via the second-order delta
        method (see `_delta_method`), applied per response variable, and
        store the results as `y_hat_back_list`/`Sigma_y_hat_back_list` and --
        if `.predict()` has been run -- `y_pred_back_list`/
        `Sigma_y_pred_back_list`.

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
        g_inv : Callable[[float], float]
            The inverse of the response transform in the model formula
            (e.g. `jnp.exp` for a `np.log(y)` response), as a scalar ->
            scalar JAX-traceable function.
        """

        if self.y_hat_list is not None and self.Sigma_y_hat_list is not None:
            y_hat_back_list, Sigma_hat_back_list = [], []
            for mu_i, Sigma_i in zip(self.y_hat_list, self.Sigma_y_hat_list):
                m, S = self._delta_method(g_inv, mu_i, Sigma_i)
                y_hat_back_list.append(m)
                Sigma_hat_back_list.append(S)

            self.y_hat_back_list = y_hat_back_list
            self.Sigma_y_hat_back_list = Sigma_hat_back_list

        if self.y_pred_list is not None:
            y_pred_back_list, Sigma_pred_back_list = [], []
            for mu_i, Sigma_i in zip(self.y_pred_list, self.Sigma_y_pred_list):
                m, S = self._delta_method(g_inv, mu_i, Sigma_i)
                y_pred_back_list.append(m)
                Sigma_pred_back_list.append(S)
            
            self.y_pred_back_list = y_pred_back_list
            self.Sigma_y_pred_back_list = Sigma_pred_back_list

        return self

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

        # Add the out-of-sample prediction summary, if .predict() has been run
        if self.y_pred_list is not None:
            pstats = self._pred_summary_stats()

            top_left_pred = dict(
                [
                    ("Pred. points:", lambda: [f"{pstats['n_points']}"]),
                    ("Pred. values (# missing):", lambda: [f"{pstats['n_pred']} ({pstats['n_missing']})"]),
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

        # self.results = np.array([0])
        # self.params = self.beta
        # self.param_names = self.xbeta_names
        # self.bse = np.zeros(len(self.beta))
        # self.tvalues = np.zeros(len(self.beta))
        # self.pvalues = np.zeros(len(self.beta))
        name_width=15
        
        if self._hessian is not None or self._cov_params is not None:
            bse_struct = self.bse  # structured ModelParams with bse fields
        else:
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

        # todo: Add the parameters table (measurement equation parameters)
        self.measurement = [
            SimpleNamespace() for _ in range(self.model.nvar)
        ]  # Create a list with one SimpleNamespace for compatibility with summary structure

        # fixed effect
        for m in self.measurement:
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
