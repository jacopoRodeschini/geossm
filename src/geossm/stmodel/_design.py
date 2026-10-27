"""Design-matrix assembly of the spatio-temporal model (internal).

Functions of the fitted formula builders (`DesignMatricesBuilder`), shared by
`LRStateSpaceModel` (training, `getDesignMatrix`, simulation) and the
prediction of fitted results (`geossm.stmodel._fitted.predict_from_fit`).
"""

import numpy as np
import jax.numpy as jnp

from geossm.utils import block_diag_3D


def as_prediction_dataframes(df, builders):
    """
    Normalize `df` (as accepted by `predict`/`getDesignMatrix`) to a list of
    GeoDataFrames, one per formula/builder.

    A single GeoDataFrame is broadcast to every formula (the same
    locations/times are used to predict every response variable, e.g. a
    shared regular grid). A list/tuple is used as-is, one entry per formula
    -- e.g. to predict/compare each response variable at its own, different
    set of held-out locations -- and must have exactly one entry per formula.
    """
    if isinstance(df, (list, tuple)):
        if len(df) != len(builders):
            raise ValueError(
                f"Number of dataframes ({len(df)}) must match number of "
                f"formulas ({len(builders)})"
            )
        return list(df)
    return [df] * len(builders)


def prediction_grid(df, builders, domain, verbose=True, include_response=False):
    """
    Build the design matrices for new (prediction) locations/times, one per
    formula, reusing each formula's fitted `DesignMatricesBuilder` so
    stateful transforms (e.g. standardize()) reuse training statistics
    instead of being recomputed on `df`.

    `df` may be a single GeoDataFrame (broadcast to every formula) or a list
    with one GeoDataFrame per formula (see `as_prediction_dataframes`).
    `include_response`, if `True`, also evaluates each formula's response on
    its dataframe (see `DesignMatricesBuilder.build_predict`) -- that
    dataframe must then contain the response column.

    Returns `(points, grids, ndim, pdim, block_p, T)`.
    """
    dfs_in = as_prediction_dataframes(df, builders)

    dfs = [
        b.build_predict(d, verbose=verbose, domain=dom, include_response=include_response)
        for b, d, dom in zip(builders, dfs_in, domain)
    ]

    T = [gr.T for gr in dfs]
    points = [gr.points for gr in dfs]

    pdim = [grid.N for grid in dfs]
    block_p = np.hstack((0, np.cumsum(pdim)))
    ndim = block_p[-1]

    return points, dfs, ndim, pdim, block_p, T


def split_design(grids):
    """Per-variable responses and fixed-effect design matrices of `grids`
    (`DesignMatrices`, one per formula), skipping missing ones."""
    y_list = [grid.y for grid in grids if grid.y is not None]
    Xbeta_list = [grid.X for grid in grids if grid.X is not None]
    return y_list, Xbeta_list


def stack_design(y_list, Xbeta_list):
    """Stack the per-variable responses (vertically, `(p, T)`) and design
    matrices (block-diagonal across variables, `(p, b, T)`, via
    `block_diag_3D`) into the model-wide arrays consumed by
    `StateSpaceModel`; `None` for an empty list."""
    y = jnp.vstack(y_list) if len(y_list) > 0 else None
    Xbeta = block_diag_3D(*Xbeta_list) if len(Xbeta_list) > 0 else None
    return y, Xbeta


def prediction_design(df, builders, domain, verbose=True, include_response=False, log=None):
    """
    Design matrix for new (out-of-sample) locations/times, reusing each
    formula's fitted `DesignMatricesBuilder` and the measurement-equation
    `domain` of the fit.

    Returns `(points, grids, ndim, pdim, block_p, T, Xbeta)`; with
    `include_response=True`, `(points, T, Xbeta_list, y_list)` instead (see
    `LRStateSpaceModel.getDesignMatrix`). `log(msg)` receives progress
    messages.
    """
    log = log or (lambda msg: None)
    log("Building observation grid...")
    points, grids, ndim, pdim, block_p, T = prediction_grid(
        df, builders, domain, verbose=verbose, include_response=include_response,
    )
    log("Building observation grid... Done.")

    log("Building the design matrix...")
    y_list, Xbeta_list = split_design(grids)
    log("Building the design matrix... Done.")

    if include_response:
        return points, T, Xbeta_list, y_list

    _, Xbeta = stack_design(y_list, Xbeta_list)
    return points, grids, ndim, pdim, block_p, T, Xbeta
