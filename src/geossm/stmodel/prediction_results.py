"""Out-of-sample predictions of a fitted spatio-temporal model."""

import warnings

import numpy as np
from statsmodels.iolib.summary import Summary

from geossm.ssm.statespace_results import _safe_sqrt_variance
from geossm.stmodel._fitted import _read_only_array
from geossm.stmodel._output import (areal_weights, as_polygon_gdf, build_geo_dataframe, delta_method,
                                    diag_variance, normalize_g_invs)
from geossm.utils import split_by_block, format_info_table, _summary_table_source


class LRPredictionResults:
    """
    Out-of-sample predictions of a fitted `LRStateSpaceModel`, returned by
    `LRStateSpaceResults.get_prediction(df)`.

    The predictive mean is `H x_t + X beta` and the uncertainty is the
    state-driven covariance `H P_t H'` (plug-in parameters, without the
    measurement error `s2e`), at the prediction sites and timestamps, for
    every response variable stacked as in the fit (`block_p` splits them).
    All arrays are read-only NumPy arrays in the dtype of the fit.

    Attributes
    ----------
    y_pred : numpy.ndarray, shape (p, T)
        Predicted mean.
    Sigma_y_pred : numpy.ndarray, shape (p, p, T), or None
        Predictive covariance (`None` with `std=False`).
    var_pred : numpy.ndarray, shape (p, T), or None
        Point-wise predictive variance, the diagonal of `Sigma_y_pred`.
    points : list of numpy.ndarray
        Prediction sites of each response variable, shape ``(n_i, 2)``.
    block_p : numpy.ndarray
        Row boundaries of each response variable in the stacked arrays.
    timestamps : list of numpy.ndarray
        Prediction timestamps of each response variable.
    crs : pyproj.CRS
        CRS of the prediction sites.
    tdelta : float
        Wall-clock time of the prediction kernels (seconds).
    y_pred_back_list, Sigma_y_pred_back_list : list of numpy.ndarray or None
        Original-scale predictions (delta method), set by `back_transform`.

    Typical usage::

        pred = results.get_prediction(grid)
        pred.back_transform(jnp.exp)            # if y = np.log(...)
        gdf = pred.to_geo()
        areal = pred.to_poly("regions.shp", std=True)
    """

    def __init__(self, fitted_state, y_pred, Sigma_y_pred=None, var_pred=None, points=None, block_p=None,
                 timestamps=None, crs=None, tdelta=None):
        dtype = fitted_state.dtype
        self._fitted_state = fitted_state
        self.y_pred = _read_only_array(y_pred, dtype)
        self.Sigma_y_pred = _read_only_array(Sigma_y_pred, dtype)
        if var_pred is None and self.Sigma_y_pred is not None:
            var_pred = diag_variance(self.Sigma_y_pred)
        self.var_pred = _read_only_array(var_pred, dtype)
        self.points = list(points)
        self.block_p = np.asarray(block_p)
        self.timestamps = list(timestamps)
        self.crs = crs
        self.tdelta = tdelta
        self.y_pred_back_list = None
        self.Sigma_y_pred_back_list = None

    @property
    def fitted_state(self):
        """FittedState: State of the fit these predictions were computed from
        (see `LRStateSpaceResults.fitted_state`)."""
        return self._fitted_state

    @property
    def y_name(self):
        """tuple of str: Response-variable names."""
        return self._fitted_state.y_name

    _y_names = y_name

    # ---- Per-variable views (split along block_p) ----

    @property
    def y_pred_list(self):
        """list of ndarray: Per-response-variable view of `y_pred`."""
        return split_by_block(self.y_pred, self.block_p)

    @property
    def var_pred_list(self):
        """list of ndarray or None: Per-response-variable view of `var_pred`."""
        return None if self.var_pred is None else split_by_block(self.var_pred, self.block_p)

    @property
    def Sigma_y_pred_list(self):
        """list of ndarray or None: Per-response-variable diagonal blocks of
        `Sigma_y_pred` (cross-variable covariance dropped)."""
        if self.Sigma_y_pred is None:
            return None
        _, Sigma_list = split_by_block(self.y_pred, self.block_p, self.Sigma_y_pred)
        return Sigma_list

    # ---- Back-transformation and export ----

    def back_transform(self, g_invs):
        """
        Map the predictions back to the response's original scale with the
        second-order delta method, per response variable, and store them as
        `y_pred_back_list`/`Sigma_y_pred_back_list` (see
        `LRStateSpaceResults.back_transform` for the method and what is
        estimated: the original-scale mean of the latent signal).

        Parameters
        ----------
        g_invs : Callable[[float], float] or list/tuple of callables
            Inverse of the response transform (e.g. `jnp.exp` for a
            `np.log(y)` response), scalar -> scalar and JAX-traceable, or one
            per response variable.

        Returns
        -------
        LRPredictionResults
            `self`. Without the covariance (`std=False`) the predictions are
            not back-transformed (warning).
        """
        g_invs = normalize_g_invs(g_invs, len(self._y_names))
        if self.Sigma_y_pred is None:
            warnings.warn("back_transform(): the predictions have no covariance (std=False), "
                          "so they are not back-transformed; use get_prediction(df, std=True).",
                          RuntimeWarning, stacklevel=2)
            return self
        y_back, Sigma_back = [], []
        for mu_i, Sigma_i, g_i in zip(self.y_pred_list, self.Sigma_y_pred_list, g_invs):
            m, S = delta_method(g_i, mu_i, Sigma_i)
            y_back.append(m)
            Sigma_back.append(S)
        self.y_pred_back_list = y_back
        self.Sigma_y_pred_back_list = Sigma_back
        return self

    def to_geo(self):
        """
        The predictions as a GeoDataFrame, one row per (site, timestamp),
        ready to be exported (e.g. `.to_file("predictions.shp")`).

        Columns: `point_id`, `timestamp`, then `y_pred_<var>`/`std_pred_<var>`
        for every response variable (plus `y_pred_back_<var>`/
        `std_pred_back_<var>` after `back_transform`), in the dtype of the
        fit. `std_*` is the standard deviation of the latent mean (no
        measurement error).
        """
        back_var = (None if self.Sigma_y_pred_back_list is None
                    else [diag_variance(S) for S in self.Sigma_y_pred_back_list])
        return build_geo_dataframe(
            self._y_names, self.points, self.timestamps, self.y_pred_list, self.var_pred_list, self.crs,
            prefix="pred", y_back_list=self.y_pred_back_list, var_back_list=back_var,
        )

    def to_poly(self, polys, back_transform=False, std=False):
        """
        Change of support: aggregate the point predictions of this object
        to areal (polygon) predictions, approximating the block mean
        `Y(B) = 1/|B| * integral_B Y(s) ds` of each polygon `B` by the
        average of the predicted values at the prediction points inside it.

        The intended workflow is to predict on a dense grid first, then
        aggregate::

            pred = results.get_prediction(fine_grid)
            gdf = pred.to_poly("regions.shp", std=True)

        Since the average is linear, with `W` the `(m, n)` averaging
        matrix (`W[k, j] = 1/n_k` for the `n_k` points in polygon `k`), the
        areal predictive mean and covariance are, per time step `t`:

            y_poly[:, t]       = W @ y_pred[:, t]
            Sigma_poly[:, :, t] = W @ Sigma_pred[:, :, t] @ W.T

        The full point covariance is required for the areal variance: the
        std of a polygon mean is `sqrt(w_k' Sigma w_k)`, which includes the
        cross-covariances between points. Averaging the point stds instead
        assumes perfect correlation (overestimating it), and averaging the
        variances divided by `n_k` assumes independence (underestimating
        it, increasingly so for denser grids).

        Only the diagonal of `Sigma_poly` is computed and returned (the
        `std_<var>` columns): the cross-covariances between polygons (its
        off-diagonal) are not provided. The per-polygon std is therefore
        correct for each polygon on its own, but not sufficient for
        quantities combining several polygons (e.g. the uncertainty of a
        sum/difference of regions, or a joint confidence region), which
        would need the full `W Sigma_t W'`.

        Notes
        -----
        - `Sigma_y_pred` is the state-driven covariance `H P_t H'`: it
          excludes the measurement error (whose contribution to a block
          mean vanishes as `s2e / n_k`, so it is correctly left out of the
          areal signal) and the estimation uncertainty of `beta`.
        - The discrete average approximates the areal integral only when
          the prediction points are dense and roughly uniform within each
          polygon; polygons containing no point get NaN (with a warning).
        - Only the per-time-step covariance is available, so the
          uncertainty of temporal aggregates (e.g. an annual polygon mean)
          cannot be derived from these outputs.
        - Cross-variable covariance is dropped (as in `Sigma_y_pred_list`):
          each response variable is aggregated independently.
        - With `back_transform=True` the original-scale point predictions
          (`y_pred_back_list`/`Sigma_y_pred_back_list`, from the delta
          method) are averaged, i.e. the polygon mean of the original-scale
          process -- not `g_inv` of the polygon mean on the model scale.

        Parameters
        ----------
        polys : list of shapely (Multi)Polygon, GeoSeries, GeoDataFrame, or path
            The target polygons, or a path to a vector file (e.g. a
            shapefile) readable by `geopandas.read_file`. Reprojected to the
            prediction CRS (`crs`) if needed; geometries without a CRS
            are assumed to already be in it. The attribute columns of a
            GeoDataFrame/file are carried over to the output.
        back_transform : bool, default False
            Aggregate the original-scale predictions instead. Requires
            `.back_transform(g_invs)` to have been run.
        std : bool, default False
            Also compute the areal predictive std of each polygon mean
            (`std_<var>` columns), `sqrt(diag(W Sigma_t W'))`.

        Returns
        -------
        geopandas.GeoDataFrame
            One row per (polygon, timestamp), with columns `poly_id`,
            `timestamp`, the input polygon attributes, then, per response
            variable `<var>` in `y_name`: `n_points_<var>`
            (prediction points in the polygon), `<var>` (areal mean), and
            `std_<var>` (if `std=True`) -- the last two suffixed
            `_back` if `back_transform=True` -- and `geometry` last.
        """
        import geopandas as geopd

        if back_transform:
            if self.y_pred_back_list is None:
                raise ValueError(
                    "to_poly(back_transform=True) requires back-transformed "
                    "predictions: run `.back_transform(g_invs)` first."
                )
            y_list, Sigma_list, suffix = self.y_pred_back_list, self.Sigma_y_pred_back_list, "_back"
        else:
            y_list, Sigma_list, suffix = self.y_pred_list, self.Sigma_y_pred_list, ""

        if Sigma_list is None:
            if std:
                raise ValueError(
                    "to_poly(std=True) requires the prediction covariance: "
                    "use `results.get_prediction(df, std=True)`."
                )
            Sigma_list = [None] * len(y_list)

        y_names = self._y_names
        ts = self.timestamps[0]
        for name, t in zip(y_names, self.timestamps):
            if len(t) != len(ts) or not np.array_equal(np.asarray(t), np.asarray(ts)):
                raise ValueError(
                    f"to_poly() requires every response variable to share the same "
                    f"prediction timestamps, but '{name}' differs from '{y_names[0]}'."
                )

        polygons = as_polygon_gdf(polys, self.crs)
        m, T = len(polygons), len(ts)

        # Long format (row-major over (T, m)), matching to_geo()'s layout
        rows = np.tile(np.arange(m), T)
        out = polygons.iloc[rows].reset_index(drop=True)
        out.insert(0, "poly_id", rows)
        out.insert(1, "timestamp", np.repeat(ts, m))

        for name, points, y, Sigma in zip(y_names, self.points, y_list, Sigma_list):
            col = f"{name}{suffix}"
            W, counts = areal_weights(points, polygons)
            empty = counts == 0
            if empty.any():
                warnings.warn(
                    f"to_poly(): {int(empty.sum())} of {m} polygon(s) contain no "
                    f"prediction point for '{name}'; their values are set to NaN. "
                    f"Use a denser prediction grid.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            y_poly = np.asarray(W @ np.asarray(y), dtype=float)  # (m, T)
            y_poly[empty, :] = np.nan

            out[f"n_points_{name}"] = np.tile(counts, T)
            out[col] = y_poly.T.ravel()

            if std:
                # diag(W Sigma_t W') = rowsum(W * (W Sigma_t)), without
                # forming the (m, m) cross-polygon covariance
                Sigma = np.asarray(Sigma)
                var = np.empty((T, m))
                for t in range(T):
                    WS = W @ Sigma[:, :, t]  # (m, n)
                    var[t] = np.asarray(W.multiply(WS).sum(axis=1)).ravel()
                sd = _safe_sqrt_variance(var, context="to_poly")
                sd[:, empty] = np.nan  # after the sqrt: already warned above
                out[f"std_{col}"] = sd.ravel()

        geom = polygons.geometry.name
        out = out[[c for c in out.columns if c != geom] + [geom]]
        return geopd.GeoDataFrame(out, geometry=geom, crs=polygons.crs)

    # ---- Summary ----

    def _summary_stats(self):
        y_all = np.concatenate([np.asarray(y).ravel() for y in self.y_pred_list])
        n_points = sum(np.asarray(p).shape[0] for p in self.points)
        if self.var_pred is None:
            mean_std = np.nan
        else:
            std = _safe_sqrt_variance(self.var_pred, context="LRPredictionResults.summary")
            mean_std = np.nanmean(std) if not np.all(np.isnan(std)) else np.nan
        return {
            "n_points": n_points,
            "n_pred": y_all.size,
            "n_missing": int(np.sum(np.isnan(y_all))),
            "y_min": np.nanmin(y_all),
            "y_median": np.nanmedian(y_all),
            "y_max": np.nanmax(y_all),
            "y_mean_std": mean_std,
        }

    def summary(self):
        """
        statsmodels-style summary of the predictions: response names, number
        of sites, timestamps and values, range of the predicted mean, mean
        predictive standard deviation, runtime.

        Returns
        -------
        statsmodels.iolib.summary.Summary
        """
        s = self._summary_stats()
        left = [
            ("Dep. Variables:", [", ".join(self._y_names)]),
            ("Pred. points (values):", [f"{s['n_points']} ({s['n_pred']}, missing {s['n_missing']})"]),
            ("Timestamps:", [f"{len(self.timestamps[0])}"]),
        ]
        right = [
            ("Pred. y (min, med, max):", [f"{s['y_min']:.3g}, {s['y_median']:.3g}, {s['y_max']:.3g}"]),
            ("Pred. mean std:", [f"{s['y_mean_std']:.3g}"]),
            ("Pred. runtime (s):", [f"{self.tdelta:.3g}"]),
        ]
        smry = Summary()
        smry.add_table_2cols(_summary_table_source(), title="LR State Space Model Predictions",
                             gleft=left, gright=right, yname=None, xname=None)
        return smry

    def __str__(self):
        return str(self.summary())

    def __repr__(self):
        rows = [("Dep. Variables:", [", ".join(self._y_names)]),
                ("Shape (p, T):", [f"{self.y_pred.shape}"]),
                ("Covariance:", ["full" if self.Sigma_y_pred is not None else
                                 "variance only" if self.var_pred is not None else "none"])]
        return f"{self.__class__.__name__}\n" + format_info_table(rows, indent=2)
