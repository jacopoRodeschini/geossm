"""Fitted state of the spatio-temporal model and what is computed from it (internal).

`FittedState` is the interface between a fitted `LRStateSpaceModel` and its
results: everything needed after `fit()` -- the observed log-likelihood (for
the Hessian), predictions at new locations, names in summaries -- is a
function of it, so results never call back into the model (Section B, P1).
"""

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from geossm.ssm.statespace import (
    _compute_predict_mean_kernel_JAX,
    _compute_predict_sigma_kernel_JAX,
    _filter_kernelJAX,
)
from geossm.stmodel._design import prediction_design


def _read_only_copy(a):
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class FittedState:
    """
    State of an `LRStateSpaceModel` at the end of a `fit()`: its configuration
    and the estimates. Everything computed after fitting -- the Hessian,
    predictions at new locations, names in summaries -- is a function of it,
    so a later `fit()`/`setup()`/`set()` on the model does not change those
    results (findings F09, F27). Available as `LRStateSpaceResults.fitted_state`.

    Built by `LRStateSpaceModel._get_fitted_state()` (configuration) and
    completed with the estimates by `LRStateSpaceResults`. Fields cannot be
    reassigned; sequences are tuples and arrays are read-only (the estimate
    arrays are the results' own arrays, not copies). The covariance
    functions, formula builders and domains are kept by reference: they are
    configuration and must not be modified (e.g. re-`setup()`) after fitting.

    Attributes
    ----------
    cov_function : tuple of spdeAppoxCov
        Latent covariance functions of the fit (FEM mesh and basis).
    points : tuple of numpy.ndarray
        Observation sites of each response variable, shape ``(n_i, 2)``.
    pdim : numpy.ndarray
        Number of sites of each response variable.
    nvar, nlat : int
        Number of response variables and of latent factors.
    y_name : tuple of str
        Response-variable names, one per response variable.
    xbeta_names : tuple of tuple of str
        Covariate names of each response variable.
    builders : tuple of DesignMatricesBuilder
        Fitted formula builders, used to build the design matrix at new
        locations with the training transforms (e.g. `standardize()`).
    domain : tuple
        Measurement-equation domain of each response variable.
    dtype, itype : numpy.dtype
        Float and integer dtype of the model.
    verbose : bool
        Verbosity of the model at the end of the fit (progress messages of
        `predict`).
    params : ModelParams or None
        Estimated parameters (the results' `params`).
    x_smoothed, P_smoothed : numpy.ndarray or None
        Smoothed states ``(q, T+1)`` and their covariances ``(q, q, T+1)``;
        index 0 is the initial state.
    timestamps_hat : tuple of numpy.ndarray or None
        Training timestamps of each response variable.

    The last four are `None` in a configuration-only snapshot (used
    internally by the model).
    """

    cov_function: tuple
    points: tuple
    pdim: np.ndarray
    nvar: int
    nlat: int
    y_name: tuple
    xbeta_names: tuple
    builders: tuple
    domain: tuple
    dtype: np.dtype
    itype: np.dtype
    x_smoothed: np.ndarray
    P_smoothed: np.ndarray
    verbose: bool = False
    params: Optional[Any] = None
    P_pred_smoothed : Optional[np.ndarray] = None,
    # sufficient statistics
    S11: Optional[np.ndarray] = None
    S10: Optional[np.ndarray] = None
    S00: Optional[np.ndarray] = None
    timestamps_hat: Optional[tuple] = None



# %% Spatial system matrices

def _spde_marginal_precision(ks, mass_list, stiff_list, perm_list, n_inner):
    """Block-diagonal marginal precision of the inner FEM nodes, one block
    per covariance function, differentiable in `ks`.

    Same math as `spdeAppoxCov.precision(marginal=True)` (NumPy/SciPy, used by
    `fit()`): Q = sigma2(k) K C^-1 K with K = k^2 C + G (nu = 1), then the
    Schur complement Q_11 - Q_12 Q_22^-1 Q_12' of the inner nodes, solving for
    Q_22^-1 Q_12' directly instead of inverting Q_22.

    `perm_list` holds the inner-then-outer node permutations (integer
    arrays) and `n_inner` the number of inner nodes (Python ints, static
    under `jit`), so that the block sizes are known at trace time. Shared by
    the M-step `ks` objective and the Hessian of the log-likelihood.
    """
    blocks = []
    for ki, C, G, perm, ni in zip(ks, mass_list, stiff_list, perm_list, n_inner):
        Ci = jnp.diag(1.0 / C.diagonal())
        K = ki**2 * C + G
        # gamma() of a Python float is float64 under x64: use ki's dtype (F03)
        sigma2k = (jax.scipy.special.gamma(jnp.asarray(1.0, dtype=ki.dtype)) /
                   (jax.scipy.special.gamma(jnp.asarray(2.0, dtype=ki.dtype)) * 4 * jnp.pi * ki**2))
        Qi = sigma2k * (K @ Ci @ K)

        Qperm = Qi[perm][:, perm]
        Q_11 = Qperm[:ni, :ni]
        Q_12 = Qperm[:ni, ni:]
        Q_22 = Qperm[ni:, ni:]
        blocks.append(Q_11 - Q_12 @ jnp.linalg.solve(Q_22, Q_12.T))

    return jax.scipy.linalg.block_diag(*blocks)


def mesh_arrays(cov_function, dtype, itype):
    """
    Mesh inputs of `_spde_marginal_precision` for `cov_function`: mass and
    stiffness matrices (dense, `dtype`), the inner-then-outer node
    permutations (`itype`), and the inner-node counts (Python ints, static
    under `jit`).
    """
    mass_list, stiff_list, perm_list, n_inner = [], [], [], []
    for cov in cov_function:
        mass, stiff, inner_mask = cov.fem_arrays()
        mass_list.append(jnp.array(mass.toarray(), dtype=dtype))
        stiff_list.append(jnp.array(stiff.toarray(), dtype=dtype))
        perm_list.append(jnp.asarray(np.concatenate([np.where(inner_mask)[0], np.where(~inner_mask)[0]]),
                                     dtype=itype))
        n_inner.append(int(inner_mask.sum()))
    return mass_list, stiff_list, perm_list, tuple(n_inner)


def latent_basis(cov_function, points, dtype):
    """
    Each latent factor's FEM basis at each response variable's sites,
    row-normalised (see `spdeAppoxCov.latent_basis`): a nested list
    `basis[p][q]` of dense arrays of shape `(n_p, q_inner)`, used by `build_H`.
    """
    return [[jnp.asarray(cov.latent_basis(pts).toarray(), dtype=dtype) for cov in cov_function]
            for pts in points]


def build_H(A, basis):
    """Dense measurement matrix `H`: each (variable, latent factor) basis
    block of `latent_basis` scaled by the loading `A[p, q]`, stacked into one
    block matrix."""
    nvar, nlat = A.shape
    return jnp.block([[A[p, q] * basis[p][q] for q in range(nlat)] for p in range(nvar)])


def build_RF(s2e, f, pdim, qdim, dtype):
    """
    Diagonals of R and F (1D vectors): `s2e` repeated over the `pdim` sites
    of each response variable, `f` over the `qdim` nodes of each latent
    factor. R and F are only used through their diagonal (see
    `StateSpaceModel._prepare_diag_array`).
    """
    # NumPy repeat counts stay concrete when traced under jit (Hessian)
    rdiag = jnp.repeat(s2e, repeats=np.asarray(pdim)).astype(dtype=dtype)
    fdiag = jnp.repeat(f, repeats=np.asarray(qdim)).astype(dtype=dtype)
    return rdiag, fdiag


# %% Observed log-likelihood

def _observed_loglik_value(params, y_t, Xbeta, basis, x0, Sigma0, pdim, qdim, mesh, nvar, nlat, dtype):
    """Observed log-likelihood at `params` (differentiable with JAX), with the
    initial state `x0`, `Sigma0` fixed and known."""
    beta0 = params.beta.value
    A0 = params.A.value.reshape((nvar, nlat))
    s2e0 = params.s2e.value
    f0 = params.f.value
    ks0 = params.ks.value

    H = build_H(A0, basis)
    R, F = build_RF(s2e0, f0, pdim, qdim, dtype)

    invQ = _spde_marginal_precision(ks0, *mesh)
    Q = jnp.linalg.solve(invQ, jnp.eye(invQ.shape[0], dtype=invQ.dtype))

    _, _, _, _, _, logL = _filter_kernelJAX(
        y_t, H, R, F, Q,
        jnp.array(x0, dtype=y_t.dtype),
        jnp.array(Sigma0, dtype=y_t.dtype),
        Xbeta, beta0
    )
    return logL


def observed_loglik(state, y_obs, Xbeta, x0, Sigma0):
    """
    `ModelParams -> scalar` closure computing the observed log-likelihood of
    the model described by `state` at fixed `y_obs`/`Xbeta`/`x0`/`Sigma0`;
    differentiated by `LRStateSpaceResults._compute_hessian` with
    `jax.hessian`.
    """
    basis = latent_basis(state.cov_function, state.points, state.dtype)
    # Block sizes as NumPy arrays: they must stay concrete (jnp.repeat sizes)
    # when the closure is traced under jit (F21).
    pdim = np.asarray(state.pdim)
    qdim = np.asarray([cov.n_latent for cov in state.cov_function])
    return partial(
        _observed_loglik_value,
        y_t=y_obs, Xbeta=Xbeta, basis=basis, x0=x0, Sigma0=Sigma0, pdim=pdim, qdim=qdim,
        mesh=mesh_arrays(state.cov_function, state.dtype, state.itype),
        nvar=state.nvar, nlat=state.nlat, dtype=state.dtype,
    )


# %% Prediction

def predict_from_fit(state, df, verbose=True, std=True, log=None):
    """
    Out-of-sample prediction from a fitted state (`state.params`,
    `state.x_smoothed`/`P_smoothed`): design matrix at `df` with the fitted
    builders, basis at the new sites, `y = H x + X beta` and, if `std`, the
    state-driven covariance `H P H'`, at the prediction timestamps (a subset
    of the training ones).

    `verbose` is passed to the design-matrix builders; `log(msg)` receives
    progress messages. Returns a dict with `points`, `block_p`, `y_pred`,
    `Sigma_y_pred` (NumPy, `state.dtype`; `None` if not `std`), `tdelta`,
    `timestamps` and `crs`.
    """
    log = log or (lambda msg: None)
    log("Predicting response variable...")

    points, grids, ndim, pdim, block_p, T, Xbeta_predict = prediction_design(
        df, state.builders, state.domain, verbose=verbose, log=log,
    )

    log("Parsing the model results (LRStateSpaceResults)...")
    beta = state.params.beta.value
    A = state.params.A.value

    log("Get the filtered state")
    x_T = state.x_smoothed
    P_T = state.P_smoothed

    # Restrict the smoothed states to the prediction timestamps: the
    # prediction df may cover only a (contiguous) sub-window of the training
    # period, e.g. when covariates are available only there. The latent state
    # is shared across response variables, so every prediction grid must
    # share one time axis.
    log("Filtering the valid timestamps...")
    timestamps_pred = grids[0].timestamps
    if any(not np.array_equal(g.timestamps, timestamps_pred) for g in grids[1:]):
        raise ValueError("All prediction grids must share the same timestamps")

    timestamps_train = state.timestamps_hat[0]
    idx = np.flatnonzero(np.isin(timestamps_train, timestamps_pred))
    if idx.size != timestamps_pred.size:
        raise ValueError("Prediction timestamps must be a subset of the training timestamps")
    log("Found {} of {} training timestamps.".format(idx.size, timestamps_train.size))

    # x_T/P_T carry the prior at column 0 (dropped by the kernels), so
    # training step k is column k+1; keep column 0 as the prior slot.
    cols = np.r_[0, idx + 1]
    x_T = x_T[:, cols]
    P_T = P_T[:, :, cols]

    log("Computing the SSM model matrice H...")
    basis = latent_basis(state.cov_function, points, state.dtype)
    H = build_H(A, basis)
    log("Computing the H {} matrix... Done.".format(H.shape))

    # In the dtype of the fit (the design matrix is not cast, F45)
    dtype = np.dtype(state.dtype)
    Xbeta_predict = jnp.asarray(Xbeta_predict, dtype=dtype)
    log("Start Prediction the SSM...")
    t_start = time.time()
    y_pred = _compute_predict_mean_kernel_JAX(H, x_T, Xbeta_predict, beta)
    Sigma_y_pred = _compute_predict_sigma_kernel_JAX(H, P_T) if std else None
    tdelta = time.time() - t_start
    log("Prediction done. Time elapsed: {}.".format(tdelta))

    return {
        "points": points,
        "block_p": block_p,
        "y_pred": np.asarray(y_pred, dtype=dtype),
        "Sigma_y_pred": None if Sigma_y_pred is None else np.asarray(Sigma_y_pred, dtype=dtype),
        "tdelta": tdelta,
        # CRS is assumed identical across variables (build_predict already
        # enforces it matches the training CRS for each formula)
        "timestamps": [grid.timestamps for grid in grids],
        "crs": grids[0].crs,
    }
