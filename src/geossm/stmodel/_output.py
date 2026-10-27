"""Output helpers shared by the fit and prediction results (internal).

GeoDataFrame export, polygons and areal weights for the change of support,
and the delta-method back-transformation.
"""

import jax
import jax.numpy as jnp
import numpy as np

from geossm.ssm.statespace_results import _safe_sqrt_variance


def diag_variance(Sigma):
    """Point-wise variances `(n, T)` from covariances `(n, n, T)`."""
    return np.diagonal(np.asarray(Sigma), axis1=0, axis2=1).T


def build_geo_dataframe(y_names, points_list, timestamps_list, y_list, var_list, crs,
                        prefix, y_back_list=None, var_back_list=None):
    """
    Build one GeoDataFrame, one row per (point, timestamp), from per-variable
    means `y_list` (each `(n, T)`) and point-wise variances `var_list` (each
    `(n, T)`, or `None`) that share a common grid (`points_list[i]`/
    `timestamps_list[i]` must be the same across `i`). `prefix` (`"hat"` or
    `"pred"`) names the value columns: `y_<prefix>_<var>`/`std_<prefix>_<var>`,
    plus `y_<prefix>_back_<var>`/`std_<prefix>_back_<var>` when
    `y_back_list`/`var_back_list` are given.
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

    def add_columns(names, ys, variances, col_prefix):
        # variances is None when fit()/the prediction ran with std=False: no std columns
        if variances is None:
            variances = [None] * len(names)
        for name, y, var in zip(names, ys, variances):
            y = np.asarray(y)
            data[f"y_{col_prefix}_{name}"] = y.T.ravel()  # (T, n) row-major: matches geoms below
            if var is not None:
                std = _safe_sqrt_variance(np.asarray(var).T, context="to_geo")  # (T, n)
                data[f"std_{col_prefix}_{name}"] = std.ravel().astype(y.dtype)

    add_columns(y_names, y_list, var_list, prefix)
    if y_back_list is not None:
        add_columns(y_names, y_back_list, var_back_list, f"{prefix}_back")

    geoms = [Point(xy) for xy in points]

    return geopd.GeoDataFrame(
        data,
        geometry=np.tile(geoms, T),
        crs=crs,
    )


def normalize_g_invs(g_invs, nvar):
    """`g_invs` (a callable or one per response variable) as a list of `nvar`
    callables."""
    if isinstance(g_invs, (list, tuple)):
        if len(g_invs) != nvar:
            raise ValueError(
                f"g_invs must be a single callable or a list/tuple of callables "
                f"matching the number of response variables ({nvar}), "
                f"but got {len(g_invs)}."
            )
        return list(g_invs)
    return [g_invs] * nvar


def as_polygon_gdf(polys, crs):
    """
    Normalize the `polys` argument of `to_poly` -- a list of shapely
    (Multi)Polygons, a single (Multi)Polygon, a GeoSeries/GeoDataFrame,
    or a path to a vector file (e.g. a shapefile) -- into a
    GeoDataFrame with a 0..m-1 index, reprojected to `crs` (the
    prediction grid's CRS). Geometries without a CRS (plain shapely
    objects, or a file/frame with no CRS set) are assumed to already
    be in `crs`.
    """
    import geopandas as geopd
    from os import PathLike
    from shapely.geometry.base import BaseGeometry

    if isinstance(polys, (str, PathLike)):
        gdf = geopd.read_file(polys)
    elif isinstance(polys, geopd.GeoDataFrame):
        gdf = polys.copy()
    elif isinstance(polys, geopd.GeoSeries):
        gdf = geopd.GeoDataFrame(geometry=polys)
    elif isinstance(polys, BaseGeometry):
        gdf = geopd.GeoDataFrame(geometry=[polys])
    else:
        gdf = geopd.GeoDataFrame(geometry=list(polys))

    if len(gdf) == 0:
        raise ValueError("to_poly() requires at least one polygon.")

    geom_type = gdf.geom_type
    if not geom_type.isin(["Polygon", "MultiPolygon"]).all():
        bad = sorted(set(geom_type[~geom_type.isin(["Polygon", "MultiPolygon"])].astype(str)))
        raise ValueError(
            f"to_poly() requires Polygon/MultiPolygon geometries, but got {bad}."
        )

    if gdf.crs is None:
        gdf = gdf.set_crs(crs) if crs is not None else gdf
    elif crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    return gdf.reset_index(drop=True)


def areal_weights(points, polygons):
    """
    Sparse averaging matrix `W` (shape `(m, n)`) mapping the `n`
    prediction points to the `m` polygons: `W[k, j] = 1 / n_k` if point
    `j` falls in polygon `k` (boundary included), else 0, with `n_k`
    the number of points in polygon `k`. A point on a boundary shared
    by two polygons contributes to both. Rows of polygons containing
    no point are all-zero; `n_k` (returned alongside) flags them.
    """
    import geopandas as geopd
    from scipy import sparse

    points = np.asarray(points)
    pts = geopd.points_from_xy(points[:, 0], points[:, 1], crs=polygons.crs)
    # (2, k) positional pairs: row 0 -> point, row 1 -> polygon
    pt_idx, poly_idx = polygons.sindex.query(pts, predicate="intersects")

    m, n = len(polygons), points.shape[0]
    counts = np.bincount(poly_idx, minlength=m)
    W = sparse.csr_matrix(
        (1.0 / counts[poly_idx], (poly_idx, pt_idx)), shape=(m, n)
    )
    return W, counts


def delta_method(g_inv, mu, Sigma):
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


def direct_transform(g_inv, y):
    """
    Apply `g_inv` elementwise to an observed value `y` (shape `(n,
    T)`), with no delta-method correction -- unlike `delta_method`,
    `y` here is a point (an actual observation), not the mean of an
    estimated distribution, so there is no variance to propagate.
    """
    y = jnp.asarray(y)
    h = jax.vmap(g_inv)(y.ravel()).reshape(y.shape)
    return np.asarray(h)
