"""
State Space Models Module
==========================

Implements the linear-Gaussian state-space model (SSM)

.. math::
    y_t &= H x_t + X_t \\beta + e_t, \\qquad e_t \\sim N(0, R) \\\\
    x_t &= F x_{t-1} + \\eta_t, \\qquad \\eta_t \\sim N(0, Q) \\\\
    x_0 &\\sim N(x_0, \\Sigma_0)

where ``y_t`` is the :math:`p`-dimensional observation vector at time
``t``, ``x_t`` is the :math:`q`-dimensional latent state, ``X_t`` is a
:math:`(p, b)` slice of exogenous regressors with coefficients ``beta``,
``H`` is the (time-constant) observation/design matrix, ``F`` is the
state transition matrix, and ``Q``/``R`` are the process/observation
noise covariances. In this implementation ``F`` and ``R`` are restricted
to diagonal matrices (stored internally as their 1D diagonal - see
`_prepare_diag_array`), while ``H`` and ``Q`` are general dense matrices.

The module exposes the JIT-compiled JAX kernels implementing the Kalman
filter (`_filter_kernelJAX`), the Rauch-Tung-Striebel smoother
(`_smoother_kernelJAX`), simulation (`_sim_kernelJAX`), the EM
sufficient statistics (`_compute_expected_values_kernelJAX`) and
predicted-observation covariance (`_compute_predict_kernel_JAX`), as
well as the public :class:`StateSpaceModel` class that wraps them.
"""

import functools
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit
import time
from geossm.utils import KeyStream
from datetime import date
from statsmodels.iolib.summary import Summary
from types import SimpleNamespace
from .statespace_results import StateSpaceResults
from geossm.utils import _select_device, _to_backend, _on_device


def _itype_for(dtype):
    """Integer dtype matching the precision of a given float dtype (32<->32, 64<->64)."""
    return jnp.int64 if jnp.dtype(dtype).itemsize == 8 else jnp.int32


def _ensure_x64_for_dtype(dtype):
    """Enable JAX's x64 mode if 64-bit precision was explicitly requested.

    JAX silently truncates float64/int64 arrays back to 32-bit unless x64 mode
    is enabled, so requesting a 64-bit dtype only takes effect if we flip this
    flag on. Left untouched (and off by default) for 32-bit dtypes so that the
    default, fast 32-bit path is unaffected.
    """
    if jnp.dtype(dtype).itemsize == 8 and not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)


def _highest_matmul_precision(f):
    """Trace `f` under full (non-tensor-core-reduced) matmul precision.

    On GPU, JAX's default (unset) matmul precision can route float32 `@`/dot
    ops through reduced-precision tensor-core kernels (e.g. TensorFloat32,
    ~10-bit mantissa vs float32's 23-bit). That's usually an acceptable
    tradeoff, but the Kalman covariance update here (P_upd = P_pred - K @ H @
    P_pred) subtracts two similar-magnitude matrices, and reduced-precision
    matmul error gets amplified by that cancellation into an outright
    indefinite P_pred at the next step (eigenvalues off by ~1e-1, not
    ~1e-7) -- far too large for any reasonable Cholesky jitter to absorb.
    Forcing "highest" precision here matches CPU's (always full-precision)
    behavior and is what actually fixes it, not a bigger jitter.

    Applied as the innermost decorator (below @jit) so the precision is baked
    into the compiled kernel at trace time, regardless of whatever ambient
    `jax.default_matmul_precision` the caller has set.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        with jax.default_matmul_precision("highest"):
            return f(*args, **kwargs)
    return wrapper


def _chol_jitter(mat):
    """Scale-aware jitter added before a stabilizing Cholesky decomposition.

    A fixed 1e-6 (the historical constant here) is only meaningful relative
    to an O(1)-magnitude matrix; scaling it by the matrix's own mean diagonal
    keeps it proportionate for differently-scaled covariances. This is a
    defense-in-depth measure for genuine float32 rounding on borderline
    (near-singular) matrices -- it is NOT a fix for reduced-precision-matmul
    errors, which can be orders of magnitude larger than any jitter should
    reasonably be; see `_highest_matmul_precision` for that.
    """
    scale = jnp.maximum(jnp.mean(jnp.abs(jnp.diag(mat))), 1.0)
    return 1e-6 * scale


# %% JAX kernel functions for SSM

@_highest_matmul_precision
def _sim_kernelJAX(keys, H, R, F, Q, x0, Sigma0, Xbeta, beta):
    """Simulate a trajectory forward from the state-space model, using a
    Python ``for`` loop over time (not ``jax.lax.scan``, despite the
    docstring of the historical/commented-out ``@jit`` variant below).

    Draws :math:`x_0 \\sim N(x_0, \\Sigma_0)` and then, for
    :math:`t = 0, \\dots, T-1`, generates

    .. math::
        y_t &= X_t \\beta + H x_t + e_t, & e_t &\\sim N(0, R) \\\\
        x_{t+1} &= F x_t + \\eta_t, & \\eta_t &\\sim N(0, Q)

    ``R`` and ``F`` are consumed as diagonals (see `_prepare_diag_array`),
    so their Cholesky factor / matrix product reduce to an elementwise
    square root / scaling rather than a dense ``(p, p)``/``(q, q)``
    operation.

    Parameters
    ----------
    keys : geossm.utils.KeyStream
        Stream of JAX PRNG keys; one key is drawn per random vector
        (initial state, then one observation-noise and one process-noise
        draw per time step).
    H, R, F, Q, x0, Sigma0, Xbeta, beta
        Model system matrices, as stored on :class:`StateSpaceModel`
        (``R`` and ``F`` as their 1D diagonals).

    Returns
    -------
    y_t : jax.numpy.ndarray, shape (p, T)
        Simulated observations :math:`y_0, \\dots, y_{T-1}`.
    x_t : jax.numpy.ndarray, shape (q, T+1)
        Simulated states :math:`x_0, \\dots, x_T` (includes the initial
        draw at index 0).
    """

    p = R.shape[0]
    q = F.shape[0]
    T = Xbeta.shape[2]

    # R and F are diagonal (stored as their 1D diagonal -- see
    # `_prepare_diag_array`), so their Cholesky factor / matmul reduce to
    # elementwise sqrt / scaling instead of dense (p, p)/(q, q) operations.
    chol_R_diag = jnp.sqrt(R)
    chol_Q = jnp.linalg.cholesky(Q)
    chol_Sigma0 = jnp.linalg.cholesky(Sigma0)

    # --- Initial State (t=0) ---
    initial_noise = jax.random.normal(keys.next(), shape=(q,))
    x_current = x0 + chol_Sigma0 @ initial_noise  # This is state x_0

    # --- Simulation using a Python for-loop ---
    # Since JAX arrays are immutable, we build Python lists of the results
    # and stack them into a single array at the end.
    x_history = [x_current]
    y_history = []

    # Loop T times to generate T observations (y_0, ..., y_{T-1})
    for t in range(T):
        # 1. Generate the observation y_t based on the current state x_t
        obs_noise = chol_R_diag * jax.random.normal(keys.next(), shape=(p,))
        mean_reg = Xbeta[:, :, t] @ beta
        y_t = mean_reg + H @ x_current + obs_noise
        y_history.append(y_t)

        # 2. Evolve the state to the next step: x_{t+1} from x_t
        process_noise = chol_Q @ jax.random.normal(keys.next(), shape=(q,))
        x_next = F * x_current + process_noise

        # 3. Store the new state and update the current state for the next loop iteration
        x_history.append(x_next)
        x_current = x_next

    # --- Final Assembly ---
    # Convert the lists of arrays into single JAX arrays with the correct shape.
    # jnp.stack(..., axis=1) is equivalent to np.array(...).T
    final_y_t = jnp.stack(y_history, axis=1)
    final_x_t = jnp.stack(x_history, axis=1)

    return final_y_t, final_x_t

""" 
@jit
def _sim_kernelJAX(keys, H, R, F, Q, x0, Sigma0, Xbeta, beta):
    ```Simulate a linear Gaussian SSM with JAX primitives (GPU-friendly).```
    T = Xbeta.shape[2]
    p = R.shape[0]
    q = F.shape[0]

    chol_R = jnp.linalg.cholesky(R)
    chol_Q = jnp.linalg.cholesky(Q)
    chol_Sigma0 = jnp.linalg.cholesky(Sigma0)

    key0, key_obs, key_state = jax.random.split(keys, 3)
    eps0 = jax.random.normal(key0, shape=(q,), dtype=jnp.float32)
    obs_eps = jax.random.normal(key_obs, shape=(T, p), dtype=jnp.float32)
    state_eps = jax.random.normal(key_state, shape=(T, q), dtype=jnp.float32)

    x_init = x0 + chol_Sigma0 @ eps0
    fixed_effect = jnp.einsum("pkt,k->pt", Xbeta, beta)

    def step(x_curr, inputs):
        obs_eps_t, state_eps_t, fe_t = inputs
        y_t = fe_t + H @ x_curr + chol_R @ obs_eps_t
        x_next = F @ x_curr + chol_Q @ state_eps_t
        return x_next, (y_t, x_next)

    _, (y_hist, x_next_hist) = jax.lax.scan(
        step,
        x_init,
        (obs_eps, state_eps, fixed_effect.T),
    )

    y_t = y_hist.T
    x_t = jnp.concatenate([x_init[:, None], x_next_hist.T], axis=1)

    return y_t, x_t
"""

@jit
@_highest_matmul_precision
def _filter_kernelJAX(y_t, H, R, F, Q, x0, Sigma0, Xbeta, beta):
    """Run the Kalman filter forward recursion via ``jax.lax.scan``.

    For each time step, computes the standard predict/update Kalman
    recursion

    .. math::
        x_{t|t-1} &= F x_{t-1|t-1}, &
        P_{t|t-1} &= F P_{t-1|t-1} F^T + Q \\\\
        e_t &= y_t - X_t \\beta - H x_{t|t-1}, &
        \\Sigma_{e,t} &= H P_{t|t-1} H^T + R \\\\
        K_t &= P_{t|t-1} H^T \\Sigma_{e,t}^{-1}, &
        x_{t|t} &= x_{t|t-1} + K_t e_t \\\\
        P_{t|t} &= P_{t|t-1} - K_t H P_{t|t-1}

    but never forms the dense ``(p, p)`` matrices :math:`\\Sigma_{e,t}`
    or :math:`\\Sigma_{e,t}^{-1}` explicitly. Instead it applies the
    Woodbury identity / matrix determinant lemma to express the gain,
    the quadratic form and the log-determinant needed for the
    log-likelihood purely in terms of ``(q, q)``-sized quantities

    .. math::
        M &= P_{t|t-1}^{-1} + H^T R^{-1} H \\\\
        \\Sigma_{e,t}^{-1} &= R^{-1} - R^{-1} H\\, M^{-1} H^T R^{-1} \\\\
        |\\Sigma_{e,t}| &= |M|\\, |P_{t|t-1}|\\, |R|

    This is what lets ``p`` (the number of observed locations, the
    dimension a low-rank spatial model is meant to scale in) stay out of
    any matrix inversion, while ``q`` (the latent rank) stays small.
    ``R`` and ``F`` are used only through their diagonal (see
    `_prepare_diag_array`), so the ``F @ P @ F.T`` and ``H.T @ R^{-1}``
    products above reduce to elementwise scalings rather than dense
    matmuls. Matrix inverses are computed via a jittered Cholesky
    factorization (`_chol_jitter`) for numerical stability.

    Missing observations (``NaN`` entries in ``y_t``) are handled by
    zeroing both the corresponding residual entries and the
    corresponding rows of ``H`` for that time step, which removes their
    contribution from the update and log-likelihood terms while keeping
    all arrays at their static, missingness-independent shape (required
    by ``jax.lax.scan``/``jit``).

    The log-likelihood accumulated here is the Gaussian log-likelihood
    up to the additive normalizing constant :math:`-\\tfrac{1}{2}\\sum_t
    p \\log(2\\pi)`, which this kernel omits (it is constant with respect
    to the model parameters).

    Only the final Kalman gain ``K`` (from the last time step) is
    returned - it is the only one the smoother's boundary condition
    needs - rather than the full ``(T, q, p)`` history.

    Parameters
    ----------
    y_t : jax.numpy.ndarray, shape (p, T)
        Observed data, may contain ``NaN`` for missing entries.
    H, R, F, Q, x0, Sigma0, Xbeta, beta
        Model system matrices, as stored on :class:`StateSpaceModel`
        (``R`` and ``F`` as their 1D diagonals).

    Returns
    -------
    x_t : jax.numpy.ndarray, shape (q, T+1)
        Filtered state means :math:`x_{0|0}, \\dots, x_{T|T}` (index 0
        is ``x0``).
    P_t : jax.numpy.ndarray, shape (q, q, T+1)
        Filtered state covariances :math:`P_{0|0}, \\dots, P_{T|T}`
        (index 0 is ``Sigma0``).
    K : jax.numpy.ndarray, shape (q, p)
        Kalman gain at the last time step ``T``.
    x_t_1 : jax.numpy.ndarray, shape (q, T+1)
        One-step-ahead predicted state means :math:`x_{t|t-1}` (index 0
        is a zero placeholder, since there is no prediction before
        ``t=0``).
    P_t_1 : jax.numpy.ndarray, shape (q, q, T+1)
        One-step-ahead predicted state covariances :math:`P_{t|t-1}`
        (index 0 is a zero placeholder).
    logL : float
        Gaussian log-likelihood of ``y_t`` under the model, up to the
        additive constant described above.
    """

    dtype = y_t.dtype.type()
    p = H.shape[0]
    q = F.shape[0]

    # Pre-compute constants
    Iq = jnp.eye(q, dtype=dtype)
    # R and F are stored as their diagonal (1D) directly -- see
    # `_prepare_diag_array` -- since both are only ever used through their
    # diagonal here and in the smoother/simulation kernels.
    R_diag = R
    invR_diag = jnp.reciprocal(R_diag)
    H_dense = H.astype(dtype)
    f_diag = F
    FF = jnp.outer(f_diag, f_diag)

    # This is the function for a single loop iteration
    def kalman_step(carry, step_data):
        # 1. Unpack carry and step_data
        # K is only carried (not stacked into the scan history below): only
        # the very last Kalman gain is ever used (by the smoother's boundary
        # term), so keeping a full (T, p, q) history of it was pure waste.
        x_prev, P_prev, logL_accum, _K_prev = carry
        yt_slice, Xbeta_slice = step_data

        # PREDICTION
        # x_pred = F @ x_prev
        # P_pred = F @ P_prev @ F.T + Q
        x_pred = f_diag * x_prev
        P_pred = FF * P_prev + Q


        # RESIDUAL
        nan_mask = jnp.isnan(yt_slice)
        Xb = Xbeta_slice @ beta
        e = yt_slice - Xb - (H @ x_pred)
        e = jnp.where(nan_mask, 0.0, e)

        # MODIFIED H
        Hna_dense = H_dense * (~nan_mask)[:, None]

        # WOODBURY
        # invP_pred is only needed within this step (Woodbury identity below);
        # it used to also be stacked into the scan history for the smoother to
        # reuse, but that duplicated a full (q, q, T) array for something the
        # smoother can recompute inline from the (already stored) P_t_1.
        #invP_pred = solve(P_pred, Iq)
        L_P = jnp.linalg.cholesky(P_pred + _chol_jitter(P_pred) * Iq)  # jitter for numerical stability
        invL_P = jax.scipy.linalg.solve_triangular(L_P, Iq, lower=True)
        invP_pred = invL_P.T @ invL_P

        # HtinvR = H.T @ invR, computed via elementwise scaling (invR is
        # diagonal) instead of a dense (p, p) invR matrix. This keeps every
        # intermediate in (q, p) or (q, q) space -- important because p
        # (observed locations) is exactly the dimension this low-rank model
        # is meant to scale in, while q (latent rank) stays small.
        HtinvR = Hna_dense.T * invR_diag[None, :]  # (q, p)
        HtinvRH = HtinvR @ Hna_dense  # (q, q) == H.T @ invR @ H

        M = invP_pred + HtinvRH

        L_M = jnp.linalg.cholesky(M + _chol_jitter(M) * Iq)  # jitter for numerical stability
        invL_M = jax.scipy.linalg.solve_triangular(L_M, Iq, lower=True)
        invM = invL_M.T @ invL_M

        # KALMAN GAIN
        # K = P_pred @ H.T @ invSigmaE, where (Woodbury identity)
        #   invSigmaE = invR - invR @ H @ invM @ H.T @ invR
        # Expanded and regrouped in terms of HtinvR/HtinvRH so the dense
        # (p, p) invSigmaE is never formed:
        #   K = P_pred @ (HtinvR - HtinvRH @ invM @ HtinvR)
        K = P_pred @ (HtinvR - HtinvRH @ (invM @ HtinvR))

        # UPDATE STATE
        x_upd = x_pred + K @ e
        P_upd = P_pred - K @ Hna_dense @ P_pred
        # Joseph form for P_upd keep the update symmetric and PD
        # IKH = (Iq - K @ Hna_dense)
        # P_upd =  IKH @ P_pred @ IKH.T + K @ R @ K.T


        # LOG-LIKELIHOOD
        # logdetSigmaE = (
        #     jnp.linalg.slogdet(M)[1]
        #     + jnp.linalg.slogdet(P_pred)[1]
        #     + jnp.sum(jnp.log(R_diag))
        # )

        logdet_M    = 2.0 * jnp.sum(jnp.log(jnp.diag(L_M)))
        logdet_Ppred = 2.0 * jnp.sum(jnp.log(jnp.diag(L_P)))
        logdetSigmaE = logdet_M + logdet_Ppred + jnp.sum(jnp.log(R_diag))

        # e.T @ invSigmaE @ e, expanded the same way as K above so this is
        # O(p*q) instead of O(p^2) via an explicit invSigmaE.
        v = HtinvR @ e
        quad_e = jnp.sum(invR_diag * e * e) - v @ (invM @ v)

        logL_accum += logdetSigmaE + quad_e

        # 2. Pack carry for next step and outputs for this step
        next_carry = (x_upd, P_upd, logL_accum, K)
        outputs = {
            "x_t": x_upd,
            "P_t": P_upd,
            "x_t_1": x_pred,
            "P_t_1": P_pred,
        }
        return next_carry, outputs

    # Prepare initial state and inputs for scan
    # The scan will iterate over the time dimension
    K_init = jnp.zeros((q, p), dtype=dtype)
    initial_carry = (x0, Sigma0, 0.0, K_init)
    # We need to transpose inputs so that T is the leading dimension
    # y_t: [p, T] -> [T, p]
    # Xbeta: [p, b, T] -> [T, p, b]
    scan_inputs = (y_t.T, jnp.moveaxis(Xbeta, -1, 0))

    (final_x, final_P, final_logL, final_K), history = jax.lax.scan(
        kalman_step, initial_carry, scan_inputs
    )

    # Post-process the results from the history dictionary
    # The outputs will have T as the leading dimension, so we move it back
    x_t = jnp.moveaxis(history["x_t"], 0, -1)
    P_t = jnp.moveaxis(history["P_t"], 0, -1)
    K = final_K  # only the final gain is needed (by the smoother's boundary term)
    x_t_1 = jnp.moveaxis(history["x_t_1"], 0, -1)
    P_t_1 = jnp.moveaxis(history["P_t_1"], 0, -1)

    # Add the initial state to the beginning of the time series arrays
    x_t = jnp.concatenate([x0[:, None], x_t], axis=1)
    x_t_1 = jnp.concatenate([jnp.zeros((q, 1), dtype=dtype), x_t_1], axis=1)

    P_t = jnp.concatenate([Sigma0[:, :, None], P_t], axis=2)
    P_t_1 = jnp.concatenate(
        [jnp.zeros(Sigma0.shape, dtype=dtype)[:, :, None], P_t_1], axis=2
    )
    logL = -0.5 * final_logL

    # jax.block_until_ready(x_t)

    return x_t, P_t, K, x_t_1, P_t_1, logL


@jit
@_highest_matmul_precision
def _smoother_kernelJAX(H, F, x_t, P_t, Klast, x_t_1, P_t_1):
    """Run the fixed-interval Rauch-Tung-Striebel (RTS) backward smoother.

    Starting from the last filtered state (:math:`x_{T|T}`,
    :math:`P_{T|T}`), scans backward over :math:`t = T-1, \\dots, 0`
    computing the standard RTS recursion

    .. math::
        J_t &= P_{t|t} F^T P_{t+1|t}^{-1} \\\\
        x_{t|T} &= x_{t|t} + J_t \\left(x_{t+1|T} - x_{t+1|t}\\right) \\\\
        P_{t|T} &= P_{t|t} + J_t \\left(P_{t+1|T} - P_{t+1|t}\\right) J_t^T

    together with the lag-one smoothed covariance
    :math:`P_{t,t-1|T} = \\mathrm{Cov}(x_t, x_{t-1} \\mid y_{0:T})`
    (needed by the EM sufficient statistics, see
    `_compute_expected_values_kernelJAX`), via the standard fixed-interval
    smoother recursion

    .. math::
        P_{t,t-1|T} = P_{t-1|t-1} J_{t-1}^T
            + J_t \\left(P_{t+1,t|T} - F P_{t|t}\\right) J_{t-1}^T

    initialized at :math:`t=T` with the boundary condition
    :math:`P_{T,T-1|T} = (I - K_T H) F P_{T-1|T-1}`, where ``K_T`` is the
    last Kalman gain from the filter pass (``Klast``).

    As in `_filter_kernelJAX`, ``F`` is used only through its diagonal,
    so ``F @ P`` products reduce to elementwise column scaling, and all
    matrix inverses use a jittered Cholesky factorization
    (`_chol_jitter`). ``P_t_1`` (the filter's one-step-ahead predicted
    covariances) is reused directly rather than a separately stored
    inverse, since it is already available from the filter pass.

    Parameters
    ----------
    H, F : jax.numpy.ndarray
        Observation matrix and state-transition diagonal, as stored on
        :class:`StateSpaceModel`.
    x_t, P_t : jax.numpy.ndarray
        Filtered state means/covariances from `_filter_kernelJAX`
        (shapes ``(q, T+1)`` / ``(q, q, T+1)``).
    Klast : jax.numpy.ndarray, shape (q, p)
        Kalman gain at the final time step, from `_filter_kernelJAX`.
    x_t_1, P_t_1 : jax.numpy.ndarray
        One-step-ahead predicted state means/covariances from
        `_filter_kernelJAX` (shapes ``(q, T+1)`` / ``(q, q, T+1)``).

    Returns
    -------
    x_T : jax.numpy.ndarray, shape (q, T+1)
        Smoothed state means :math:`x_{t|T}` for :math:`t = 0, \\dots, T`.
    P_T : jax.numpy.ndarray, shape (q, q, T+1)
        Smoothed state covariances :math:`P_{t|T}`.
    P_T_1 : jax.numpy.ndarray, shape (q, q, T+1)
        Smoothed lag-one covariances :math:`P_{t,t-1|T}` (index 0 has no
        preceding time step and is not meaningful/used downstream in the
        same way as indices :math:`t \\ge 1`).
    """

    dtype = x_t.dtype.type()
    q = F.shape[0]
    f_diag = F  # F is stored as its diagonal (1D) -- see `_prepare_diag_array`
    Iq = jnp.eye(q, dtype=dtype)

    def _chol_inv(P):
        # Same Cholesky-based inverse the filter uses for the Woodbury step
        # (with the same jitter for numerical stability). Recomputed here
        # instead of being read from a stored (q, q, T) invP_t_1 array,
        # since P_t_1 (its input) is already available from the filter pass.
        L = jnp.linalg.cholesky(P + _chol_jitter(P) * Iq)
        invL = jax.scipy.linalg.solve_triangular(L, Iq, lower=True)
        return invL.T @ invL

    def smoother_step(carry, inputs):
        """
        The body of the loop for a single time step.
        This function will be compiled once by `scan`.
        """
        # 1. Unpack the carry (state from the previous iteration, i.e., time t+1)
        x_T_next, P_T_next, P_T_1_next = carry

        # 2. Unpack the inputs for the current iteration (i.e., time t)
        # Note: The lag-one covariance calc needs P_t[t-2] and P_t_1[t-1],
        # so we pass them in as well.
        (
            x_t_curr,
            P_t_curr,
            x_t_1_next,
            P_t_1_next,
            P_t_prev,
            P_t_1_curr,
        ) = inputs

        invP_t_1_next = _chol_inv(P_t_1_next)
        invP_t_1_curr = _chol_inv(P_t_1_curr)

        # --- Core Smoother Logic (from your original loop) ---
        #J_t_1 = P_t_curr @ F.T @ invP_t_1_next
        PF = P_t_curr * f_diag[None, :]          # (q,q): scale columns of P_t_curr
        J_t_1 = PF @ invP_t_1_next               # (q,q)


        x_T_curr = x_t_curr + J_t_1 @ (x_T_next - x_t_1_next)
        P_T_curr = P_t_curr + J_t_1 @ (P_T_next - P_t_1_next) @ J_t_1.T

        # Lag-one covariance
        #J_t_2 = P_t_prev @ F.T @ invP_t_1_curr
        PF_prev = P_t_prev * f_diag[None, :]          # (q,q): scale columns of P_t_prev
        J_t_2 = PF_prev @ invP_t_1_curr               # (q,q)
        
        # term = P_T_1_next - F @ P_t_curr
        term = P_T_1_next - (f_diag[:, None] * P_t_curr)
        P_T_1_curr = P_t_curr @ J_t_2.T + J_t_1 @ term @ J_t_2.T

        # 3. Prepare the new carry for the next iteration (time t-1)
        new_carry = (x_T_curr, P_T_curr, P_T_1_curr)

        # 4. Define what to store/stack at each iteration
        stacked_output = (x_T_curr, P_T_curr, P_T_1_curr)

        return new_carry, stacked_output

    # --- Prepare inputs for jax.lax.scan ---

    # 1. Initial Carry: The state at time T (the end of the data)
    # This is the starting point for the backward pass.
    x_T_last = x_t[:, -1]
    P_T_last = P_t[:, :, -1]
    # Lag-one cov at T is special. F is diagonal, so F @ P_t[:, :, -2] is an
    # elementwise row-scale by f_diag (same pattern as `term` in the
    # per-step body below) rather than a dense (q, q) matmul; computing that
    # product first lets (Iq - Klast @ H) left-multiply it exactly as in the
    # original (Iq - Klast @ H) @ F @ P_t[:, :, -2].
    P_T_1_last = (jnp.eye(q, dtype=dtype) - Klast @ H) @ (f_diag[:, None] * P_t[:, :, -2])
    init_carry = (x_T_last, P_T_last, P_T_1_last)

    # 2. Prepare the arrays to be scanned over (`xs`)
    # The loop runs from t=T-1 down to 0. We need to feed the scan function
    # the inputs in that order. We do this by stacking and reversing.

    # We need inputs from t=T-1, T-2, ..., 0
    # x_t_curr, P_t_curr, P_t_prev
    xs_x_t = x_t[:, :-1]
    xs_P_t = P_t[:, :, :-1]
    xs_P_t_prev = P_t[:, :, :-2]  # For lag-one cov

    # We need inputs from t=T, T-1, ..., 1
    # x_t_1_next, P_t_1_next (invP_t_1_next is recomputed from P_t_1_next inside the step)
    xs_x_t_1 = x_t_1[:, 1:]
    xs_P_t_1 = P_t_1[:, :, 1:]

    # We need inputs from t=T-1, T-2, ..., 0 for the lag-one cov's J_t_2
    # (invP_t_1_curr is likewise recomputed inside the step, from P_t_1_curr)
    xs_P_t_1_curr = P_t_1[:, :, 1:-1]

    # Pad the arrays that are too short to align them for stacking
    # We need T elements for the scan (from T-1 down to 0)
    # P_t_prev needs one pad, P_t_1_curr needs one pad
    q_q_pad = jnp.zeros((q, q, 1), dtype=dtype)
    P_t_prev_padded = jnp.concatenate([q_q_pad, xs_P_t_prev], axis=2)
    P_t_1_curr_padded = jnp.concatenate([q_q_pad, xs_P_t_1_curr], axis=2)

    # Now, put them into a tuple and reverse for the backward pass
    # Transposing to (T, ...) shape for scan
    xs = (
        xs_x_t.T,
        jnp.moveaxis(xs_P_t, 2, 0),
        xs_x_t_1.T,
        jnp.moveaxis(xs_P_t_1, 2, 0),
        jnp.moveaxis(P_t_prev_padded, 2, 0),
        jnp.moveaxis(P_t_1_curr_padded, 2, 0),
    )
    # Reverse time axis for backward pass
    xs_reversed = jax.tree.map(lambda x: jnp.flip(x, axis=0), xs)

    # --- Run the scan ---
    # The final carry is not needed, we use the stacked outputs
    _, (x_T_scanned, P_T_scanned, P_T_1_scanned) = jax.lax.scan(
        smoother_step, init_carry, xs_reversed
    )

    # --- Post-process the results ---
    # The outputs are also in reverse time order, so flip them back
    x_T_scanned = jnp.flip(x_T_scanned.T, axis=1)
    P_T_scanned = jnp.flip(jnp.moveaxis(P_T_scanned, 0, 2), axis=2)
    P_T_1_scanned = jnp.flip(jnp.moveaxis(P_T_1_scanned, 0, 2), axis=2)

    # Combine the scanned results (for t=0 to T-1) with the initial values (at t=T)
    x_T = jnp.concatenate([x_T_scanned, x_T_last[:, None]], axis=1)
    P_T = jnp.concatenate([P_T_scanned, P_T_last[:, :, None]], axis=2)

    # For P_T_1, the last element is special
    P_T_1 = jnp.concatenate([P_T_1_scanned, P_T_1_last[:, :, None]], axis=2)

    return x_T, P_T, P_T_1


@jit
@_highest_matmul_precision
def _compute_expected_values_kernelJAX(H, x_T, P_T, P_T_1, Xbeta, beta):
    """Compute predicted observations and EM sufficient statistics from
    smoothed states.

    Computes the fitted/predicted mean :math:`\\hat y_t = X_t \\beta + H
    x_{t|T}` (using the smoothed states :math:`t=1,\\dots,T`, i.e.
    excluding the initial prior ``x_T[:, 0]``), together with the three
    sufficient statistics used by an EM M-step for this model's ``F``
    and ``Q``:

    .. math::
        S_{11} &= \\sum_{t=1}^{T} E[x_t x_t^T]
            = \\sum_{t=1}^{T} \\left(x_{t|T} x_{t|T}^T + P_{t|T}\\right) \\\\
        S_{00} &= \\sum_{t=1}^{T} E[x_{t-1} x_{t-1}^T]
            = \\sum_{t=1}^{T} \\left(x_{t-1|T} x_{t-1|T}^T + P_{t-1|T}\\right) \\\\
        S_{10} &= \\sum_{t=1}^{T} E[x_t x_{t-1}^T]
            = \\sum_{t=1}^{T} \\left(x_{t|T} x_{t-1|T}^T + P_{t,t-1|T}\\right)

    where the outer products of the smoothed means are computed as a
    single matrix product (e.g. ``x_t_slice @ x_t_slice.T``) rather than
    a sum of per-timestep outer products, and ``P_{t,t-1|T}`` is the
    lag-one smoothed covariance returned by `_smoother_kernelJAX`.

    Parameters
    ----------
    H : jax.numpy.ndarray, shape (p, q)
        Observation matrix.
    x_T, P_T, P_T_1 : jax.numpy.ndarray
        Smoothed state means/covariances and lag-one covariances from
        `_smoother_kernelJAX`.
    Xbeta : jax.numpy.ndarray, shape (p, b, T)
        Exogenous regressors.
    beta : jax.numpy.ndarray, shape (b,)
        Regression coefficients.

    Returns
    -------
    y_hat : jax.numpy.ndarray, shape (p, T)
        Fitted/predicted observation means.
    S11, S10, S00 : jax.numpy.ndarray, shape (q, q)
        EM sufficient statistics, as defined above.
    """

    # Slices of the smoothed states
    # x_t terms range from t=1 to T
    # x_{t-1} terms range from t=0 to T-1
    x_t_slice = x_T[:, 1:]  # Shape: [q, T]
    x_tm1_slice = x_T[:, :-1]  # Shape: [q, T]

    # --- 1. Compute predicted observations (y_hat) ---
    # The term Xbeta @ beta can be computed efficiently using einsum.
    # y_hat_t = Xbeta_t @ beta + H @ x_t
    # Computed inline (mean only) rather than via _compute_predict_kernel_JAX,
    # which also builds the (p, p, T) predictive covariance Sigma_y_hat -
    # unused here and, for a low-rank model (p >> q), far larger than any
    # of the (q, q, T) state arrays. That covariance is only needed by the
    # public `predict()` path, which calls _compute_predict_kernel_JAX directly.
    y_hat = jnp.einsum("pkt,k->pt", Xbeta, beta) + H @ x_t_slice
    # --- 2. Compute sufficient statistics (S11, S10, S00) ---
    # E[sum(x x')] = sum(E[x]E[x]' + Cov(x)) = sum(x_T x_T') + sum(P_T)
    # The sum of outer products (x @ x') can be vectorized as X @ X.T

    # S11 = E[sum_{t=1..T} x_t x_t']
    # We need sums over t=1 to T
    S11 = (x_t_slice @ x_t_slice.T) + jnp.sum(P_T[:, :, 1:], axis=2)

    # S00 = E[sum_{t=1..T} x_{t-1} x_{t-1}']
    # We need sums over t-1=0 to T-1
    S00 = (x_tm1_slice @ x_tm1_slice.T) + jnp.sum(P_T[:, :, :-1], axis=2)

    # S10 = E[sum_{t=1..T} x_t x_{t-1}']
    # P_T_1 is Cov(x_t, x_{t-1}), so the sum starts from t=1
    S10 = (x_t_slice @ x_tm1_slice.T) + jnp.sum(P_T_1[:, :, 1:], axis=2)

    return y_hat, S11, S10, S00

@jit
@_highest_matmul_precision
def _compute_predict_kernel_JAX(H, x_T, P_T, Xbeta, beta):
    """Compute predicted observations and their plug-in state-driven
    predictive covariance from smoothed (or filtered) states.

    .. math::
        \\hat y_t &= X_t \\beta + H x_{t|T} \\\\
        \\Sigma_{\\hat y, t} &= H P_{t|T} H^T

    ``Sigma_y_hat`` only propagates the state's own posterior
    uncertainty through ``H``; it does not add the observation noise
    covariance ``R``. Callers that want a full prediction interval
    (rather than a confidence interval for the mean) add ``R``
    themselves - see `StateSpaceResults.conf_int_y`.

    Parameters
    ----------
    H : jax.numpy.ndarray, shape (p, q)
        Observation matrix.
    x_T, P_T : jax.numpy.ndarray
        State means/covariances (typically the smoothed states, shapes
        ``(q, T+1)`` / ``(q, q, T+1)``); index 0 (the prior) is dropped
        before computing the outputs.
    Xbeta : jax.numpy.ndarray, shape (p, b, T)
        Exogenous regressors.
    beta : jax.numpy.ndarray, shape (b,)
        Regression coefficients.

    Returns
    -------
    y_hat : jax.numpy.ndarray, shape (p, T)
        Predicted observation means.
    Sigma_y_hat : jax.numpy.ndarray, shape (p, p, T)
        State-driven predictive covariance :math:`H P_{t|T} H^T` (excludes
        observation noise ``R``).
    """
    
    # Compute the expected valure
    y_hat_covariate_term = jnp.einsum("pkt,k->pt", Xbeta, beta)
    y_hat_state_term = H @ x_T[:, 1:]  # Use the smoothed states for prediction
    y_hat = y_hat_covariate_term + y_hat_state_term

    # compute the plugin uncertainty
    Sigma_y_hat = jnp.einsum("ip,pqt,jq->ijt", H, P_T[:, :, 1:], H)

    return y_hat, Sigma_y_hat

# %% State Space Model Class
class StateSpaceModel:
    """A linear-Gaussian state-space model with Kalman filtering,
    smoothing and simulation capabilities.

    Implements

    .. math::
        y_t &= H x_t + X_t \\beta + e_t, \\qquad e_t \\sim N(0, R) \\\\
        x_t &= F x_{t-1} + \\eta_t, \\qquad \\eta_t \\sim N(0, Q) \\\\
        x_0 &\\sim N(x_0, \\Sigma_0)

    where ``y_t`` is the :math:`p`-dimensional observation at time
    ``t``, ``x_t`` is the :math:`q`-dimensional latent state, and
    ``X_t \\beta`` is a regression/fixed-effect term with :math:`b`
    coefficients. The class stores the system matrices and dimensions,
    and exposes `filter`, `smoother`, `estimate` (filter + smoother),
    `sim` (simulation) and `predict` on top of the JIT-compiled JAX
    kernels defined at module level (`_filter_kernelJAX`,
    `_smoother_kernelJAX`, `_sim_kernelJAX`, ...).

    This class is meant to be reusable/subclassable: for example
    ``LRStateSpaceModel`` (in ``geossm.stmodel.lrssm``) subclasses it to
    specialize the model for large-scale spatial data via a low-rank
    state covariance, while reusing the filter/smoother machinery
    defined here.

    Notes
    -----
    ``F`` and ``R`` are restricted to diagonal matrices in this
    implementation (stored internally as their 1D diagonal - see
    `_prepare_diag_array`); ``H`` and ``Q`` are general dense matrices.
    """

    def __init__(
        self,
        H=None,
        R=None,
        F=None,
        Q=None,
        x0=None,
        Sigma0=None,
        Xbeta=None,
        beta=None,
        xbeta_names=None,
        backend: str = "auto",
        dtype=jnp.float32,
    ):
        """Initialize the state-space model with system matrices and
        initial state.

        Any subset of the matrices may be omitted (``None``) to allow
        partial/staged initialization (e.g. by a subclass that fills
        them in later via `set`); the full set of dimension checks in
        `_check_parameters` only runs once ``H``, ``F``, ``Q`` and ``R``
        are all set.

        Parameters
        ----------
        H : array_like, shape (p, q), optional
            Observation/design matrix mapping the latent state to the
            mean of ``y_t``.
        R : array_like, shape (p, p) or (p,), optional
            Observation noise covariance (diagonal). A dense ``(p, p)``
            matrix is accepted for backward compatibility, but only its
            diagonal is used/stored.
        F : array_like, shape (q, q) or (q,), optional
            State transition matrix (diagonal). A dense ``(q, q)`` matrix
            is accepted for backward compatibility, but only its
            diagonal is used/stored.
        Q : array_like, shape (q, q), optional
            Process noise covariance.
        x0 : array_like, shape (q,), optional
            Mean of the initial state :math:`x_0`. Defaults to a zero
            vector when ``F``/``Q`` are given but ``x0`` is not.
        Sigma0 : array_like, shape (q, q), optional
            Covariance of the initial state :math:`x_0`. Defaults to the
            identity matrix when ``F``/``Q`` are given but ``Sigma0`` is
            not.
        Xbeta : array_like, shape (p, b, T), optional
            Exogenous regressors entering the observation equation as
            ``X_t @ beta``. Defaults to a ``(p, 1, 1)`` zero array when
            ``H`` is given but ``Xbeta`` is not.
        beta : array_like, shape (b,), optional
            Regression coefficients for ``Xbeta``. Defaults to a zero
            vector matching ``Xbeta``'s second dimension.
        xbeta_names : list of str, optional
            Human-readable names for the ``b`` regression terms (used in
            summaries); defaults to ``["X_0", "X_1", ...]`` when omitted.
        backend : {'auto', 'cpu', 'gpu'} or jax.Device, default 'auto'
            JAX compute device the model's arrays and kernels run on.
            ``'auto'`` lets JAX pick its default device.
        dtype : numpy/jax dtype, default jax.numpy.float32
            Floating-point precision used for all internal computations.
            Requesting a 64-bit dtype (e.g. ``jax.numpy.float64``)
            transparently enables JAX's x64 mode (see
            `_ensure_x64_for_dtype`).

        Raises
        ------
        ValueError
            If ``H``, ``F``, ``Q`` and ``R`` are all provided but their
            shapes are inconsistent (see `_check_parameters`).
        """
        self.dtype = jnp.dtype(dtype)  # Data type for computations
        self.itype = _itype_for(self.dtype)  # Matching integer dtype for index/block arrays
        _ensure_x64_for_dtype(self.dtype)
        self._backend = _select_device(backend)  # Computational backend (e.g., 'cpu', 'gpu', 'tpu')

        self._F = None  # State transition matrix
        self._H = None  # Observation matrix
        self._Q = None  # Process noise covariance
        self._q = None  # State dimension
        self._R = None  # Observation noise covariance
        self._x0 = None  # Initial state estimate
        self._Sigma0 = None  # Initial covariance estimate
        self._y_t = None  # observed data
        self._Xbeta = None  # Exogenous variables
        self._beta = None  # Coefficients for exogenous variables
        self._xbeta_names = None
        self._T = None  # Time length
        self._p = None  # number of measurement equation
        self._q = None  # number of state equation
        self._b = None  # number of regression coefficent

        self._type = "Linear (Gaussian)"
        self._order = "(1, 0)"  # Placeholder for ARMA order if needed
        self._today = date.today()
        self._params = None  # only the beta par.
        self._params_names = None
        self._params_dim = None

        # Set the initial state starting values if not provided
        if x0 is None and F is not None and Q is not None:
            x0 = jnp.zeros(F.shape[0])
        if Sigma0 is None and F is not None and Q is not None:
            Sigma0 = jnp.eye(F.shape[0])

        # Set default Xbeta and beta if not provided
        if Xbeta is None and H is not None:
            Xbeta = jnp.zeros((H.shape[0], 1, 1))
        if beta is None and Xbeta is not None:
            beta = jnp.zeros(Xbeta.shape[1])

        # Update parameters without checking (we will check after setting all parameters)
        self.set(
            H=H,
            R=R,
            F=F,
            Q=Q,
            x0=x0,
            Sigma0=Sigma0,
            Xbeta=Xbeta,
            beta=beta,
            xbeta_names=xbeta_names,
        )

        # Check parameters only if the key parameters are set (allow for partial initialization)
        if (
            self.H is not None
            and self.F is not None
            and self.Q is not None
            and self.R is not None
        ):
            flag, msg = self._check_parameters()
            if not flag:
                raise ValueError(msg)


    def __call__(self, y_t):
        """Shorthand for `smoother`, i.e. run the Kalman filter followed
        by the RTS smoother ("estimation" of the state).

        Parameters
        ----------
        y_t : array_like, shape (p, T)
            Observed data (may contain ``NaN`` for missing entries).

        Returns
        -------
        StateSpaceResults
            Smoothed results; see `smoother`.
        """
        return self.smoother(y_t)

    def set(
        self,
        H=None,
        R=None,
        F=None,
        Q=None,
        x0=None,
        Sigma0=None,
        y_t=None,
        Xbeta=None,
        beta=None,
        xbeta_names=None,
        yname=None,
    ):
        """Update any subset of the model's system matrices/metadata in place.

        Only arguments that are not ``None`` are updated; omitted ones
        keep their current value. This is the method the constructor and
        every public entry point (`filter`, `smoother`, `sim`, ...) use
        internally to merge caller-supplied overrides into the model
        before running a kernel. No shape/consistency validation is
        performed here - see `_check_parameters` for that.

        Parameters
        ----------
        H : array_like, shape (p, q), optional
            Observation/design matrix.
        R : array_like, shape (p, p) or (p,), optional
            Observation noise covariance (diagonal).
        F : array_like, shape (q, q) or (q,), optional
            State transition matrix (diagonal).
        Q : array_like, shape (q, q), optional
            Process noise covariance.
        x0 : array_like, shape (q,), optional
            Initial state mean.
        Sigma0 : array_like, shape (q, q), optional
            Initial state covariance.
        y_t : array_like, shape (p, T), optional
            Observed data.
        Xbeta : array_like, shape (p, b, T), optional
            Exogenous regressors.
        beta : array_like, shape (b,), optional
            Regression coefficients for ``Xbeta``.
        xbeta_names : list of str, optional
            Names for the exogenous regression terms.
        yname : str, optional
            Name for the dependent variable (used in summaries).

        Returns
        -------
        None
        """
        # Check parameters

        self._update_parameters(
            F=F,
            H=H,
            Q=Q,
            R=R,
            x0=x0,
            Sigma0=Sigma0,
            y_t=y_t,
            Xbeta=Xbeta,
            beta=beta,
            xbeta_names=xbeta_names,
            yname=yname,
        )

    def _update_parameters(
        self,
        H=None,
        R=None,
        F=None,
        Q=None,
        x0=None,
        Sigma0=None,
        y_t=None,
        Xbeta=None,
        beta=None,
        xbeta_names = None,
        yname = None,
    ):
        """
        Helper function to update model parameters if provided.
        """

        if H is not None:
            self._H = self._prepare_array(H)
            self._p = H.shape[0]

        if R is not None:
            self._R = self._prepare_diag_array(R)

        if F is not None:
            self._F = self._prepare_diag_array(F)
            self._q = F.shape[0]

        if Q is not None:
            self._Q = self._prepare_array(Q)

        if x0 is not None:
            self._x0 = self._prepare_array(x0)
        else:
            if self.q is not None:
                self._x0 = self._prepare_array(jnp.zeros(self.q))

        if Sigma0 is not None:
            self._Sigma0 = self._prepare_array(Sigma0)
        else:
            if self.q is not None:
                self._Sigma0 = self._prepare_array(jnp.eye(self.q))

        if Xbeta is not None:
            Xbeta_arr = self._prepare_array(Xbeta)
            self._Xbeta = Xbeta_arr

            # infer (p, b, T) when possible
            if Xbeta_arr.ndim >= 3:
                # common shape: (p, b, T)
                self._b = int(Xbeta_arr.shape[1])
                self._T = int(Xbeta_arr.shape[2])
            elif Xbeta_arr.ndim == 2:
                # ambiguous: treat second dim as b
                self._b = int(Xbeta_arr.shape[1])
                self._T = None
            else:
                self._b = None
                self._T = None

        # --- handle beta if provided ---
        if beta is not None:
            beta_arr = self._prepare_array(beta)
            self._beta = beta_arr

            try:
                b_from_beta = int(beta_arr.shape[0])
            except Exception:
                b_from_beta = None

            if b_from_beta is not None:
                # if we've already inferred _b (e.g. from Xbeta), ensure consistency
                if getattr(self, "_b", None) is not None and self._b != b_from_beta:
                    raise ValueError(
                        f"Shape mismatch: beta has length {b_from_beta} but existing Xbeta/b implies {self._b}."
                    )
                self._b = b_from_beta

        # --- handle xbeta_names ---
        if xbeta_names is not None:
            # support nested lists per-variable: count total names across variables
            len_xbeta_names = sum(len(xb_names) for xb_names in xbeta_names)

            # if _b not known yet, set it from names
            if getattr(self, "_b", None) is None:
                self._b = int(len_xbeta_names)

            if self._b is None:
                raise ValueError("Cannot infer number of xbeta terms (b) from inputs.")
            if len_xbeta_names != int(self._b):
                raise ValueError(f"Expected {int(self._b)} xbeta names, got {len_xbeta_names}.")

            self._xbeta_names = xbeta_names
        
        if Xbeta is not None and xbeta_names is None:
            # no names provided: if b known create defaults, else leave None
            if getattr(self, "_b", None) is not None:
                self._xbeta_names = [f"X_{i}" for i in range(int(self._b))]
            else:
                self._xbeta_names = None

        if y_t is not None:
            self._y_t = self._prepare_array(y_t)
            self._p = self._y_t.shape[0]
            self._T = self._y_t.shape[1]

        if yname is not None:
            self._yname = yname
        elif self._p is not None:
            self._yname = "y"
        else:
            self._yname = None

        # update the params attributes
        self._params = self._beta
        self._params_names = self._xbeta_names
        self._params_dim = self._b

        return True

    @property
    def backend(self):
        """jax.Device: The JAX compute device this model's arrays and
        kernels are pinned to (resolved from the ``backend`` constructor
        argument via `geossm.utils._select_device`)."""
        return self._backend

    def _prepare_array(self, x):
        """Cast to the model dtype and commit the array to the configured backend device."""
        return _to_backend(self._backend, jnp.asarray(x, dtype=self.dtype))[0]

    def _prepare_diag_array(self, x):
        """Prepare R/F, which are diagonal everywhere they're used (the Kalman
        kernels only ever read their diagonal). Accepts either a dense square
        matrix (its diagonal is extracted) or a 1D vector, and always stores
        the 1D diagonal - this keeps the constructor/`.set()` call signature
        backward compatible with callers passing e.g. `R = sigma2 * np.eye(p)`,
        while avoiding a dense (p, p)/(q, q) matrix ever being built or carried
        internally.
        """
        arr = self._prepare_array(x)
        if arr.ndim == 2:
            arr = arr.diagonal()
        return arr

    def _check_parameters(self):
        """
        Checks the dimensions of the parameters.
        """
        # Convert key dims to Python ints and gather shapes safely
        p = int(self.p) if self.p is not None else None
        q = int(self.q) if self.q is not None else None
        T = int(self.T) if self.T is not None else None
        b = int(self.b) if self.b is not None else None

        flag = True
        messages = []

        def shape_str(x):
            try:
                return tuple(jnp.asarray(x).shape)
            except Exception:
                return None

        # Basic presence checks
        if p is None:
            messages.append("Number of measurement equations `p` is not set.")
            flag = False
        if q is None:
            messages.append("Number of state variables `q` is not set.")
            flag = False
        if T is None:
            messages.append(
                "Number of time steps `T` is not set (inferred from Xbeta)."
            )
            flag = False
        if b is None:
            messages.append("Number of regression coefficients `b` is not set.")
            flag = False

        if not flag:
            return flag, "\n".join(messages)

        # Helper: check positive semidefinite (symmetric) with tolerance
        def is_pos_semidef(mat):
            A = jnp.asarray(mat)
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                return False
            # symmetry check
            if not jnp.allclose(A, A.T, atol=1e-8):
                return False
            # eigenvalues >= -tol
            eigs = jnp.linalg.eigvalsh(A)
            return jnp.all(eigs >= -1e-8)
        # Sigma0: should be (q, q)
        sigma0_shape = shape_str(self.Sigma0)
        if sigma0_shape not in [(q, q), (q,), (q,)]:
            messages.append(f"Sigma0 must be shape ({q},{q}), got {sigma0_shape}.")
            flag = False

        # H: (p, q)
        H_shape = shape_str(self.H)
        if H_shape != (p, q):
            messages.append(f"H must be shape ({p},{q}), got {H_shape}.")
            flag = False

        # R: (p,) -- diagonal of the measurement noise covariance. Stored as
        # a vector (see `_prepare_diag_array`) since R is only ever used
        # through its diagonal, so its "positive semidefinite" check reduces
        # to each entry being non-negative (the eigenvalues of a diagonal
        # matrix are its entries).
        R_shape = shape_str(self.R)
        if R_shape != (p,):
            messages.append(f"R must be shape ({p},) (its diagonal), got {R_shape}.")
            flag = False
        else:
            if jnp.any(jnp.asarray(self.R) < -1e-8):
                messages.append("R (variances) must be non-negative.")
                flag = False

        # F: (q,) -- diagonal of the state transition matrix, stored as a
        # vector for the same reason as R.
        F_shape = shape_str(self.F)
        if F_shape != (q,):
            messages.append(f"F must be shape ({q},) (its diagonal), got {F_shape}.")
            flag = False

        # Q: (q, q)
        Q_shape = shape_str(self.Q)
        if Q_shape != (q, q):
            messages.append(f"Q must be shape ({q},{q}), got {Q_shape}.")
            flag = False
        # else:
        #    if not is_pos_semidef(self.Q):
        #        messages.append("Q must be symmetric positive semidefinite.")
        #        flag = False

        # x0: (q,)
        x0_shape = shape_str(self.x0)
        if x0_shape != (q,):
            messages.append(f"x0 must be shape ({q},), got {x0_shape}.")
            flag = False

        # Sigma0: (q, q)
        Sigma0_shape = shape_str(self.Sigma0)
        if Sigma0_shape != (q, q):
            messages.append(f"Sigma0 must be shape ({q},{q}), got {Sigma0_shape}.")
            flag = False
        # Check positive semidefinite
        # else:
        #     if not is_pos_semidef(self.Sigma0):
        #         messages.append(
        #             "Sigma0 must be symmetric positive semidefinite.")
        #         flag = False

        # Xbeta: (p, b, T)
        if self.Xbeta is None:
            messages.append("Set Xbeta to a default zero array of shape (p, b, T) before checking.")
            self._Xbeta = self._prepare_array(jnp.zeros((p, b, T)))

        Xbeta_shape = shape_str(self.Xbeta)
        if Xbeta_shape != (p, b, T):
            messages.append(f"Xbeta must be shape ({p},{b},{T}), got {Xbeta_shape}.")
            flag = False

        # beta: (b,)
        if self.beta is None:
            messages.append("Set beta to a default zero array of shape (b,) before checking.")
            self._beta = self._prepare_array(jnp.zeros((b,)))

        beta_shape = shape_str(self.beta)
        if beta_shape != (b,):
            messages.append(f"beta must be shape ({b},), got {beta_shape}.")
            flag = False

        return flag, "\n".join(messages)

    def _check_y_t(self, y_t):
        """
        Check the shape of the observed data y_t.
        """
        flag = True
        y_t_shape = y_t.shape

        expected_shape = (self.p, self.T)
        msg = ""
        if y_t_shape != expected_shape:
            msg = f"y_t must be shape {expected_shape}, got {y_t_shape}."
        return flag, msg

    @_on_device
    def estimate(
        self,
        y_t,
        yname=None,
        H=None,
        R=None,
        F=None,
        Q=None,
        x0=None,
        Sigma0=None,
        Xbeta=None,
        beta=None,
        xbeta_names=None,
        light=False,
    ):
        """Run the Kalman filter and smoother (filter + backward pass)
        and return the smoothed results, including the sufficient
        statistics (``S11``, ``S10``, ``S00``) needed by an EM M-step.

        This is equivalent to `smoother` - kept as a distinct, named
        entry point for API compatibility - and does not recompute
        ``S11``/``S10``/``S00``/``y_hat`` a second time on top of what
        `smoother` already computes.

        Parameters
        ----------
        y_t : array_like, shape (p, T)
            Observed data (may contain ``NaN`` for missing entries).
        yname : str, optional
            Name for the dependent variable (used in summaries).
        H, R, F, Q, x0, Sigma0, Xbeta, beta, xbeta_names : optional
            Overrides for the corresponding model parameters; see `set`.
            If omitted, the model's currently-set values are used.
        light : bool, default False
            Forwarded to `smoother`; if True, the returned
            `StateSpaceResults` omits the filter-stage arrays
            (``x_filtered``, ``P_filtered``, ``K``, ``x_pred``,
            ``P_pred``).

        Returns
        -------
        StateSpaceResults
            Smoothed results, including filtered arrays (unless
            ``light=True``), smoothed arrays, the log-likelihood and the
            EM sufficient statistics.
        """
        return self.smoother(
            y_t,
            yname=yname,
            H=H,
            R=R,
            F=F,
            Q=Q,
            x0=x0,
            Sigma0=Sigma0,
            Xbeta=Xbeta,
            beta=beta,
            xbeta_names=xbeta_names,
            light=light,
        )

    @_on_device
    def predict(self, H, x_T, P_T, Xbeta=None, beta=None):
        """Compute predicted observations and their state-driven
        predictive covariance from (typically smoothed) states.

        Thin wrapper around `_compute_predict_kernel_JAX`; see its
        docstring for the exact equations. Unlike `filter`/`smoother`,
        this does not update the model's own state (``H``, ``Xbeta``,
        ``beta`` are used as passed, not read from ``self``).

        Parameters
        ----------
        H : array_like, shape (p, q)
            Observation matrix.
        x_T : array_like, shape (q, T+1)
            State means (index 0 is the prior/initial state and is
            excluded from the output).
        P_T : array_like, shape (q, q, T+1)
            State covariances (index 0 excluded from the output).
        Xbeta : array_like, shape (p, b, T), optional
            Exogenous regressors.
        beta : array_like, shape (b,), optional
            Regression coefficients for ``Xbeta``.

        Returns
        -------
        y_hat : jax.numpy.ndarray, shape (p, T)
            Predicted observation means.
        Sigma_y_hat : jax.numpy.ndarray, shape (p, p, T)
            State-driven predictive covariance :math:`H P_{t} H^T`
            (excludes the observation noise covariance ``R``).
        """
        y_hat, Sigma_y_hat = _compute_predict_kernel_JAX(H, x_T, P_T, Xbeta, beta)
        return y_hat, Sigma_y_hat

    def _run_filter_kernel(
        self,
        y_t,
        yname=None,
        H=None,
        R=None,
        F=None,
        Q=None,
        x0=None,
        Sigma0=None,
        Xbeta=None,
        beta=None,
        xbeta_names=None,
    ):
        """
        Update parameters, validate them, and run the Kalman filter kernel.

        Returns the raw JAX arrays produced by `_filter_kernelJAX` (no numpy
        conversion, no StateSpaceResults wrapping). `filter()` wraps this for
        its own (numpy) public result; `smoother()` calls this directly and
        feeds the arrays straight into the smoother kernel, so the (q, q, T)
        filtered-covariance arrays never make an unnecessary device -> host
        (numpy) -> device round trip between the two stages.
        """
        # Update parameters if provided
        self.set(
            H=H,
            R=R,
            F=F,
            Q=Q,
            x0=x0,
            Sigma0=Sigma0,
            y_t=y_t,
            Xbeta=Xbeta,
            beta=beta,
            xbeta_names=xbeta_names,
            yname=yname,
        )

        # Check parameters
        flag, msg = self._check_parameters()
        if not flag:
            raise ValueError(msg)

        # check y_t
        flag, msg = self._check_y_t(y_t)
        if not flag:
            raise ValueError(msg)

        # Run the scan
        tStart = time.time()

        x_t, P_t, K, x_t_1, P_t_1, logL = _filter_kernelJAX(
            y_t,
            self.H,
            self.R,
            self.F,
            self.Q,
            self.x0,
            self.Sigma0,
            self.Xbeta,
            self.beta,
        )
        jax.block_until_ready(x_t)
        tDelta = time.time() - tStart

        return x_t, P_t, K, x_t_1, P_t_1, logL, tDelta

    @_on_device
    def filter(
        self,
        y_t,
        yname=None,
        H=None,
        R=None,
        F=None,
        Q=None,
        x0=None,
        Sigma0=None,
        Xbeta=None,
        beta=None,
        xbeta_names=None,
    ) -> tuple:
        """Run the Kalman filter forward pass and return the filtered
        results (state estimates, log-likelihood, and one-step-ahead
        predicted observations/sufficient statistics computed from the
        filtered - not smoothed - states).

        See `_filter_kernelJAX` for the exact filter recursion
        (predict/update, Woodbury identity, missing-data handling).

        Parameters
        ----------
        y_t : array_like, shape (p, T)
            Observed data (may contain ``NaN`` for missing entries).
        yname : str, optional
            Name for the dependent variable (used in summaries).
        H, R, F, Q, x0, Sigma0, Xbeta, beta, xbeta_names : optional
            Overrides for the corresponding model parameters; see `set`.
            If omitted, the model's currently-set values are used.

        Returns
        -------
        StateSpaceResults
            Filtered results: filtered/predicted state means and
            covariances, the Kalman gain, the log-likelihood, and the
            EM sufficient statistics computed from the filtered states.

        References
        ----------
        Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by
        State Space Methods*. Oxford University Press.
        """
        x_t, P_t, K, x_t_1, P_t_1, logL, tDelta = self._run_filter_kernel(
            y_t,
            yname=yname,
            H=H,
            R=R,
            F=F,
            Q=Q,
            x0=x0,
            Sigma0=Sigma0,
            Xbeta=Xbeta,
            beta=beta,
            xbeta_names=xbeta_names,
        )

        # compute expected values (given the filterd values)
        y_hat, S11, S10, S00, tdelta_expectation = self.computeExpectedValues(
            x_t, P_t, P_t_1
        )

        results = StateSpaceResults(
            model=self,
            x_filtered=x_t,
            P_filtered=P_t,
            K=K,
            x_pred=x_t_1,
            P_pred=P_t_1,
            llf=logL,
            time_filter=tDelta,
            y_hat=y_hat,
            S11=S11,
            S10=S10,
            S00=S00,
            time_expectation=tdelta_expectation,
        )

        return results
        # return (x_t, P_t, K, x_t_1, P_t_1, logL, tDelta)

    @_on_device
    def smoother(
        self,
        y_t,
        yname=None,
        H=None,
        R=None,
        F=None,
        Q=None,
        x0=None,
        Sigma0=None,
        Xbeta=None,
        beta=None,
        xbeta_names=None,
        light=False,
    ) -> tuple:
        """Run the Kalman filter followed by the RTS backward smoother,
        and return the smoothed results together with the EM sufficient
        statistics computed from the smoothed states.

        This is the main "estimation" entry point (filtering and
        smoothing algorithm for linear-Gaussian state-space models); see
        `_filter_kernelJAX` and `_smoother_kernelJAX` for the exact
        recursions.

        Parameters
        ----------
        y_t : array_like, shape (p, T)
            Observed data (may contain ``NaN`` for missing entries).
        yname : str, optional
            Name for the dependent variable (used in summaries).
        H, R, F, Q, x0, Sigma0, Xbeta, beta, xbeta_names : optional
            Overrides for the corresponding model parameters; see `set`.
            If omitted, the model's currently-set values are used.
        light : bool, default False
            If True, the returned `StateSpaceResults` omits the
            filter-stage arrays (``x_filtered``, ``P_filtered``, ``K``,
            ``x_pred``, ``P_pred``) instead of keeping numpy copies of
            them alive for the whole lifetime of the results object.
            They are only used internally by this method (as
            smoother-kernel inputs); once the smoothed states are
            computed they serve no further purpose for callers - such as
            an EM loop - that never read them back. The default ``False``
            preserves the previous behaviour for callers (e.g.
            ``model.smoother(y)`` used directly) that do want to inspect
            filtered vs. smoothed state.

        Returns
        -------
        StateSpaceResults
            Smoothed results: smoothed state means/covariances and
            lag-one covariances, the log-likelihood, predicted
            observations and the EM sufficient statistics (``S11``,
            ``S10``, ``S00``) computed from the smoothed states, plus
            (unless ``light=True``) the filter-stage arrays.

        References
        ----------
        Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by
        State Space Methods*. Oxford University Press.
        """

        # Run the filter kernel directly (bypassing `filter()`'s numpy-
        # converting StateSpaceResults wrapper) so the big (q, q, T) arrays
        # stay on-device until the smoother kernel has consumed them.
        x_t, P_t, K, x_t_1, P_t_1, logL, tDelta_filter = self._run_filter_kernel(
            y_t,
            yname=yname,
            H=H,
            R=R,
            F=F,
            Q=Q,
            x0=x0,
            Sigma0=Sigma0,
            Xbeta=Xbeta,
            beta=beta,
            xbeta_names=xbeta_names,
        )

        # Now run the smoother
        tStart = time.time()
        x_T, P_T, P_T_1 = _smoother_kernelJAX(
            self.H,
            self.F,
            x_t,
            P_t,
            K,
            x_t_1,
            P_t_1,
        )

        jax.block_until_ready(x_T)
        td_smoother = time.time() - tStart

        # compute expected values (given the smoothed values)
        y_hat, S11, S10, S00, tdelta_expectation = self.computeExpectedValues(
            x_T, P_T, P_T_1
        )

        results = StateSpaceResults(
            model=self,
            x_filtered=None if light else x_t,
            P_filtered=None if light else P_t,
            K=None if light else K,
            x_pred=None if light else x_t_1,
            P_pred=None if light else P_t_1,
            x_smoothed=x_T,
            P_smoothed=P_T,
            P_pred_smoothed=P_T_1,
            llf=logL,
            time_filter=tDelta_filter,
            time_smoother=td_smoother,
            y_hat=y_hat,
            S11=S11,
            S10=S10,
            S00=S00,
            time_expectation=tdelta_expectation,
        )

        return results

    def computeExpectedValues(self, x_T, P_T, P_T_1) -> tuple:
        """Compute predicted observations and EM sufficient statistics
        from a set of state means/covariances (filtered or smoothed).

        Thin, timed wrapper around `_compute_expected_values_kernelJAX`;
        see its docstring for the exact equations for ``y_hat``, ``S11``,
        ``S10`` and ``S00``. Called internally by both `filter` (on the
        filtered states) and `smoother` (on the smoothed states).

        Parameters
        ----------
        x_T : array_like, shape (q, T+1)
            State means (index 0 is the initial prior).
        P_T : array_like, shape (q, q, T+1)
            State covariances.
        P_T_1 : array_like, shape (q, q, T+1)
            Lag-one covariances :math:`\\mathrm{Cov}(x_t, x_{t-1})`
            (as produced by `_smoother_kernelJAX`; when called on
            filtered-only states this argument is typically not
            meaningful/available in the same sense).

        Returns
        -------
        y_hat : jax.numpy.ndarray, shape (p, T)
            Predicted observation means.
        S11, S10, S00 : jax.numpy.ndarray, shape (q, q)
            EM sufficient statistics.
        tDelta : float
            Wall-clock time (seconds) spent computing these quantities.
        """

        # Note: Type conversions are omitted for clarity. It's often better
        # to ensure inputs have the correct dtype before calling a JIT-compiled function.
        tStart = time.time()

        y_hat, S11, S10, S00 = _compute_expected_values_kernelJAX(
            self.H, x_T, P_T, P_T_1, self.Xbeta, self.beta
        )

        jax.block_until_ready(y_hat)
        tDelta = time.time() - tStart

        return (y_hat, S11, S10, S00, tDelta)

    @_on_device
    def sim(
        self,
        seed=1234,
        R=None,
        F=None,
        H=None,
        Q=None,
        x0=None,
        Sigma0=None,
        Xbeta=None,
        beta=None,
        block_p=None, 
        block_q=None,
        stats=True,
        verbose=False,
    ) -> jnp.ndarray:
        """Simulate a time series from the state-space model.

        Draws :math:`x_0 \\sim N(x_0, \\Sigma_0)` and then propagates the
        observation/state equations forward in time via `_sim_kernelJAX`
        (see its docstring for the exact equations). Uses a Python
        ``for``-loop under the hood, so this is not JIT-compiled and is
        slower than the filter/smoother kernels.

        Parameters
        ----------
        seed : int or geossm.utils.KeyStream, default 1234
            Either an integer seed used to build a new `KeyStream`, or an
            existing `KeyStream` to continue drawing from.
        R, F, H, Q, x0, Sigma0, Xbeta, beta : optional
            Overrides for the corresponding model parameters; see `set`.
            If omitted, the model's currently-set values are used.
        block_p : list of int, optional
            Block boundaries along the observation dimension ``p``,
            forwarded to `summarize_ssm_variances` when ``stats=True``
            (e.g. to separate multiple co-simulated processes). Defaults
            to a single block ``[0, p]``.
        block_q : list of int, optional
            Block boundaries along the latent dimension ``q``, forwarded
            to `summarize_ssm_variances`. Defaults to a single block
            ``[0, q]``.
        stats : bool, default True
            If True, additionally compute and return a theoretical-vs-
            empirical variance summary via `summarize_ssm_variances`.
        verbose : bool, default False
            If True (and ``stats=True``), print the variance summary.

        Returns
        -------
        y_t : jax.numpy.ndarray, shape (p, T)
            Simulated observations.
        x_t : jax.numpy.ndarray, shape (q, T+1)
            Simulated state vectors :math:`[x_0, \\dots, x_T]`.
        stats : dict or None
            Output of `summarize_ssm_variances` if ``stats=True``,
            otherwise ``None``.
        tdelta : float
            Wall-clock time (seconds) spent in the simulation kernel.
        """
        # Update parameters if provided
        xbeta_names = None
        yname = None
        self.set(
            H=H,
            R=R,
            F=F,
            Q=Q,
            x0=x0,
            Sigma0=Sigma0,
            Xbeta=Xbeta,
            beta=beta,
            xbeta_names=xbeta_names,
            yname=yname,
        )

        # Check parameters
        flag, msg = self._check_parameters()
        if not flag:
            raise ValueError(msg)

        if isinstance(seed, KeyStream):
            key = seed
        else:
            # Initialize PRNGKey stream
            main_key = jax.random.PRNGKey(seed)
            seed, main_key = jax.random.split(main_key)

            key = KeyStream(seed)

        # Call the simulation kernel to generate the time series
        tStart = time.time()
        y_t_sim, x_t_sim = _sim_kernelJAX(
            key,
            self.H,
            self.R,
            self.F,
            self.Q,
            self.x0,
            self.Sigma0,
            self.Xbeta,
            self.beta,
        )
        jax.block_until_ready(y_t_sim)
        tdelta = time.time() - tStart

        if stats:
            fixed_effect = jnp.einsum("pkt,k->pt", self.Xbeta, self.beta)
            y_delta = y_t_sim - fixed_effect
            stats = self.summarize_ssm_variances(x_t_sim, y_delta, block_p=block_p, block_q=block_q, verbose=verbose)
        else:
            stats = None

        return y_t_sim, x_t_sim, stats, tdelta


    def summarize_ssm_variances(self, x_sim, y_sim, block_p, block_q, decimals=4, verbose=True) -> dict:
        """Compute (and optionally print) a compact summary comparing
        theoretical (model-implied) and empirical (simulation-based)
        variances of the latent effect, observation noise and response.

        The theoretical stationary state covariance is obtained by
        solving the discrete Lyapunov equation :math:`P = F P F^T + Q`
        for :math:`P`; since ``F`` is diagonal here, this has the closed
        form :math:`P_{ij} = Q_{ij} / (1 - f_i f_j)` elementwise, avoiding
        a dense-matrix Lyapunov solve. The theoretical variance of the
        latent effect ``H @ x_t`` is then ``diag(H @ P @ H.T)``, and the
        theoretical response variance is the latent variance plus the
        observation noise variance (``R``); these are compared against
        the corresponding sample variances of ``x_sim``/``y_sim``,
        averaged within each block of ``block_p``/``block_q`` (useful
        when several processes are stacked/simulated together).

        Parameters
        ----------
        x_sim : array_like, shape (q, T+1) or (q, T)
            Simulated latent states, as returned by `sim`.
        y_sim : array_like, shape (p, T)
            Simulated observations with the fixed-effect term
            ``Xbeta @ beta`` already removed (i.e. the ``H @ x_t + e_t``
            part).
        block_p : list of int
            Block boundaries along the observation dimension ``p``
            (e.g. ``[0, p1, p1+p2, ...]``) delimiting separate processes
            for the purpose of averaging.
        block_q : list of int
            Block boundaries along the latent dimension ``q``.
        decimals : int, default 4
            Currently unused by this implementation.
        verbose : bool, default True
            If True, print a formatted summary table.

        Returns
        -------
        dict
            Dictionary with keys ``var_latent_theoretical``,
            ``var_latent_empirical``, ``var_noise``,
            ``var_y_theoretical``, ``var_y_empirical``, ``ratios`` (a
            dict of empirical/theoretical ratios) and ``P`` (the
            theoretical stationary state covariance).

        Raises
        ------
        AttributeError
            If the model is missing any of ``F``, ``Q``, ``H``, ``R``.
        """
        # Basic checks
        for attr in ("F", "Q", "H", "R"):
            if not hasattr(self, attr):
                raise AttributeError(f"model is missing '{attr}' attribute")
        F, Q, H, R = self.F, self.Q, self.H, self.R

        if block_p is None:
            block_p = [0, self.p]
        if block_q is None:
            block_q = [0, self.q]

        # Solve Lyapunov: P = F P F^T + Q. F is diagonal (stored as its 1D
        # diagonal -- see `_prepare_diag_array`), so (F P F^T)_ij = f_i f_j P_ij
        # and the equation has the closed form P_ij = Q_ij / (1 - f_i f_j)
        # elementwise -- exact, and avoids a scipy.linalg.solve_discrete_lyapunov
        # call (which would otherwise need a dense (q, q) F reconstructed just
        # for this one-off simulation-stats call).
        P = Q / (1.0 - jnp.outer(F, F))

        # Theoretical latent variance
        # temp = jnp.diag(H @ P @ H.T)
        # temp = jnp.diag(P)
        # var_z_t = jnp.array([float(jnp.sum(temp[block_q[i]:block_q[i+1]]) / (block_q[i+1] - block_q[i])) for i in range(len(block_q)-1)])

        # Theoretical variance of the latent effect (H @ x_t) is H P H^T
        temp = jnp.diag(H @ P @ H.T)
        var_latents = jnp.array([float(jnp.sum(temp[block_p[i]:block_p[i+1]]) / (block_p[i+1] - block_p[i])) for i in range(len(block_p)-1)])

        # Empirical latent variance:
        # H @ x_sim -> (n_locations, n_time) ; compute variance across locations per time, then average
        latent_effect = H @ x_sim
        var_latent_empirical = jnp.array([float(jnp.var(latent_effect[block_p[i]:block_p[i+1],:], axis=0).mean()) for i in range(len(block_p)-1)])

        # Observation noise variance (average over observation dims)
        temp = R  # R is stored as its diagonal (1D) directly
        var_noise = jnp.array([float(jnp.sum(temp[block_p[i]:block_p[i+1]]) / (block_p[i+1] - block_p[i])) for i in range(len(block_p)-1)])

        # Response variance (theoretical) and empirical
        var_y_theoretical = var_latents + var_noise
        var_y_empirical = jnp.array([jnp.var(y_sim[block_p[i]:block_p[i+1],:], axis=0).mean() for i in range(len(block_p)-1)])
        var_noise_empirical = var_y_empirical - var_latent_empirical
        
        # Ratios and SNRs
        ratios = {
            "empirical_over_theoretical_latent": var_latent_empirical / var_latents ,
            "empirical_over_theoretical_y": var_y_empirical / var_y_theoretical,
        }
        
        # Nicely formatted printout
        if verbose:       
            fmt = f"{{:<40}}{{}}"
            print("\nState-space variance summary")
            print("-" * 60)
            print(f"{'Matrix shapes:':<40} F={F.shape}, Q={Q.shape}, H={H.shape}, R={R.shape}")
            print(f"{'Observation blocks:':<40} {block_p}")
            print(f"{'Latent blocks:':<40} {block_q}")
            print("-" * 60)
            print(fmt.format("Theoretical latent variance:", var_latents))
            print(fmt.format("Empirical latent variance:", var_latent_empirical))
            print(fmt.format("Empirical / Theoretical (latent):", ratios["empirical_over_theoretical_latent"]))
            print()
            print(fmt.format("Theoretical response variance :", var_y_theoretical))
            print(fmt.format("Empirical response variance :", var_y_empirical))
            print(fmt.format("Empirical / Theoretical (y):", ratios["empirical_over_theoretical_y"]))
            print()
            print(fmt.format("Theoretical noise variance:", var_noise))
            print(fmt.format("Empirical noise variance:", var_noise_empirical))
            print("-" * 60)
           
        # Return numeric results for downstream use
        stats = {
            "var_latent_theoretical": var_latents,
            "var_latent_empirical": var_latent_empirical,
            "var_noise": var_noise,
            "var_y_theoretical": var_y_theoretical,
            "var_y_empirical": var_y_empirical,
            "ratios": ratios,
            "P": P,
        }
        
        return stats

    # propoerty
    @property
    def T(self):
        """int: Number of observed time steps."""
        return self._T

    @property
    def p(self):
        """int: Dimension of the measurement/observation equation (number
        of observed locations/variables)."""
        return self._p

    @property
    def q(self):
        """int: Dimension of the latent state equation (state rank)."""
        return self._q

    @property
    def b(self):
        """int: Number of regression coefficients (length of ``beta``)."""
        return self._b

    @property
    def y_t(self):
        """jax.numpy.ndarray, shape (p, T): Currently-set observed data
        (may contain ``NaN`` for missing entries)."""
        return self._y_t

    @property
    def params(self):
        """array_like: Free/estimated model parameters; currently an
        alias for `beta` (kept as a distinct attribute so subclasses,
        e.g. an EM/optimization-fitted model, can extend it with
        additional parameters)."""
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    @property
    def params_names(self):
        """list of str: Names corresponding to `params`; currently an
        alias for `xbeta_names`."""
        return self._params_names

    @params_names.setter
    def params_names(self, value):
        self._params_names = value

    @property
    def params_dim(self):
        """int: Dimension of `params`; currently an alias for `b`."""
        return self._params_dim

    @property
    def xbeta_names(self):
        """list of str: Human-readable names for the ``b`` regression
        terms in `Xbeta`/`beta`."""
        return self._xbeta_names

    @xbeta_names.setter
    def xbeta_names(self, value):
        self._xbeta_names = value

    @property
    def yname(self):
        """str: Name of the dependent variable ``y``, used in summaries."""
        return self._yname

    @yname.setter
    def yname(self, value):
        self._yname = value

    @property
    def type(self):
        """str: Human-readable model type/family label (used in
        summaries), e.g. ``"Linear (Gaussian)"``."""
        return self._type

    @type.setter
    def type(self, value: str):
        self._type = value

    @property
    def order(self):
        """str: Human-readable model order label (used in summaries),
        e.g. an ARMA-style ``"(p, q)"`` tuple string."""
        return self._order

    @order.setter
    def order(self, value):
        self._order = value

    @property
    def shape(self):
        """tuple of int: ``(p, q, T)`` - observation dimension, state
        dimension and number of time steps."""
        return (self._p, self._q, self._T)

    @property
    def Xbeta(self):
        """jax.numpy.ndarray, shape (p, b, T): Exogenous regressors
        entering the observation equation as ``X_t @ beta``."""
        return self._Xbeta

    @Xbeta.setter
    def Xbeta(self, value):
        """Set the Xbeta array (equivalent to ``self.set(Xbeta=value)``)."""
        self.set(Xbeta=value)

    @property
    def beta(self):
        """jax.numpy.ndarray, shape (b,): Regression coefficients for
        `Xbeta`."""
        return self._beta

    @beta.setter
    def beta(self, value):
        """Set beta (equivalent to ``self.set(beta=value)``)."""
        self.set(beta=value)

    @property
    def H(self):
        """jax.numpy.ndarray, shape (p, q): Observation/design matrix
        mapping the latent state to the mean of ``y_t``."""
        return self._H

    @H.setter
    def H(self, value):
        """Set H (equivalent to ``self.set(H=value)``)."""
        self.set(H=value)

    @property
    def R(self):
        """jax.numpy.ndarray, shape (p,): Diagonal of the observation
        noise covariance matrix ``R`` (variance of ``e_t`` per observed
        location/variable)."""
        return self._R

    @R.setter
    def R(self, value):
        """Set R (equivalent to ``self.set(R=value)``)."""
        self.set(R=value)

    @property
    def F(self):
        """jax.numpy.ndarray, shape (q,): Diagonal of the state
        transition matrix ``F``."""
        return self._F

    @F.setter
    def F(self, value):
        """Set F (equivalent to ``self.set(F=value)``)."""
        self.set(F=value)

    @property
    def Q(self):
        """jax.numpy.ndarray, shape (q, q): Process noise covariance
        matrix (covariance of ``eta_t`` in the state equation)."""
        return self._Q

    @Q.setter
    def Q(self, value):
        """Set Q (equivalent to ``self.set(Q=value)``)."""
        self.set(Q=value)

    @property
    def x0(self):
        """jax.numpy.ndarray, shape (q,): Mean of the initial state
        :math:`x_0`."""
        return self._x0

    @property
    def Sigma0(self):
        """jax.numpy.ndarray, shape (q, q): Covariance of the initial
        state :math:`x_0`."""
        return self._Sigma0

    # ----------------- Pickle support -----------------
    def __getstate__(self):
        """Return a serializable state for pickling.

        Convert JAX arrays to NumPy arrays and store basic metadata.
        """
        import numpy as np

        def to_np(x):
            if x is None:
                return None
            try:
                return np.array(x)
            except Exception:
                return x

        state = {
            "F": to_np(self._F),
            "H": to_np(self._H),
            "Q": to_np(self._Q),
            "R": to_np(self._R),
            "x0": to_np(self._x0),
            "Sigma0": to_np(self._Sigma0),
            "Xbeta": to_np(self._Xbeta),
            "beta": to_np(self._beta),
            "T": self._T,
            "p": self._p,
            "q": self._q,
            "b": self._b,
            # store dtype name for robust restoration
            "dtype": getattr(self.dtype, "name", str(self.dtype)),
            # store the backend platform (e.g. 'cpu'/'gpu') so it can be
            # re-resolved to a device on the machine that unpickles this model
            "backend": getattr(self._backend, "platform", "auto"),
            "today": self._today,
            "type": self._type,
            "order": self._order,
            "yname": self._yname,
            "xbeta_names": self._xbeta_names,
        }
        return state

    def __setstate__(self, state):
        """Restore object state from pickled state.

        Arrays are converted back to JAX arrays with the original dtype.
        """
        # Restore dtype first
        dt_name = state.get("dtype", None)
        try:
            self.dtype = jnp.dtype(dt_name) if dt_name is not None else jnp.dtype(jnp.float32)
        except Exception:
            # fallback
            try:
                self.dtype = jnp.dtype(getattr(jnp, dt_name))
            except Exception:
                self.dtype = jnp.dtype(jnp.float32)

        self.itype = _itype_for(self.dtype)
        _ensure_x64_for_dtype(self.dtype)

        # Restore the backend device. Fall back to 'auto' if the platform
        # requested at pickle time (e.g. 'gpu') isn't available on this
        # machine, so a model saved on a GPU host can still be loaded on CPU.
        backend_platform = state.get("backend", "auto")
        try:
            self._backend = _select_device(backend_platform)
        except ValueError:
            self._backend = _select_device("auto")

        def to_jax(x):
            if x is None:
                return None
            try:
                return _to_backend(self._backend, jnp.asarray(x, dtype=self.dtype))[0]
            except Exception:
                return x

        # Restore arrays and metadata
        self._F = to_jax(state.get("F", None))
        self._H = to_jax(state.get("H", None))
        self._Q = to_jax(state.get("Q", None))
        self._R = to_jax(state.get("R", None))
        self._x0 = to_jax(state.get("x0", None))
        self._Sigma0 = to_jax(state.get("Sigma0", None))
        self._Xbeta = to_jax(state.get("Xbeta", None))
        self._beta = to_jax(state.get("beta", None))

        self._T = state.get("T", None)
        self._p = state.get("p", None)
        self._q = state.get("q", None)
        self._b = state.get("b", None)

        # Metadata used by summary()/__str__(); fall back to sensible
        # defaults for state pickled before these were tracked.
        self._today = state.get("today", date.today())
        self._type = state.get("type", "Linear (Gaussian)")
        self._order = state.get("order", "(1, 0)")
        self._yname = state.get("yname", None)
        self._xbeta_names = state.get("xbeta_names", None)
        self._y_t = None
        self._params = self._beta
        self._params_names = self._xbeta_names
        self._params_dim = self._b

        # Ensure other attributes exist with sensible defaults
        if not hasattr(self, "dtype"):
            self.dtype = jnp.dtype(jnp.float32)
        if not hasattr(self, "itype"):
            self.itype = _itype_for(self.dtype)
        if not hasattr(self, "_backend"):
            self._backend = _select_device("auto")
        if not hasattr(self, "_F"):
            self._F = None
        if not hasattr(self, "_H"):
            self._H = None
        if not hasattr(self, "_Q"):
            self._Q = None
        if not hasattr(self, "_R"):
            self._R = None
        if not hasattr(self, "_x0"):
            self._x0 = None
        if not hasattr(self, "_Sigma0"):
            self._Sigma0 = None
        if not hasattr(self, "_Xbeta"):
            self._Xbeta = None
        if not hasattr(self, "_beta"):
            self._beta = None



    def generate_summary(self):
        """Build the two small key/value tables (model identity/config
        on the left, shape and mean system-matrix diagonals on the
        right) used by `summary`.

        Returns
        -------
        tuple of list
            ``(gen_top_left, gen_top_right)``, each a list of
            ``(label, [value])`` pairs suitable for
            ``statsmodels.iolib.summary.Summary.add_table_2cols``.
        """

        # top-left / top-right small tables
        p, q, T = self.shape if hasattr(self, "shape") else ("N/A", "N/A", "N/A")


        top_left = dict(
            [
                ("Model name:", lambda: [self.__class__.__name__]),
                (
                    "Model type:",
                    lambda: [self.type if hasattr(self, "type") else "N/A"],
                ),
                (
                    "Model order:",
                    lambda: [self.order if hasattr(self, "order") else "N/A"],
                ),
                ("Dep. Variable:", lambda: [self.y_name if hasattr(self, "y_name") and self.y_name is not None else "N/A"]),
                ("Date:", lambda: [self._today]),
                (
                    "Backend (JAX):",
                    lambda: [
                        f"{self.backend} (dtype {self.dtype}); "
                        f"{jax.default_backend()}, devices: {jax.devices()}"
                    ],
                ),
            ]
        )

        top_right = dict(
            [
                ("Shape (p, q, T) :", lambda: [f"(p = {p}, q = {q}, T = {T})"]),
                (
                    "Diag. R",
                    lambda: [f"{jnp.mean(self.R):.2f}"
                        if self.R is not None
                        else "None"]
                ),
                (
                    "Diag. Q",
                    lambda: [f"{jnp.mean(jnp.diag(self.Q)):.2f}"
                        if self.Q is not None
                        else "None"],
                ),
                (
                    "Diag. F",
                    lambda: [
                        f"{jnp.mean(self.F):.2f}"
                        if self.F is not None
                        else "None"],
                ),
                (
                    "mean x0",
                    lambda: [
                        f"{jnp.mean(self.x0):.2f}" if self.x0 is not None else "None"],
               ),
                (
                    "mean Sigma0",
                    lambda: [
                        f"{jnp.mean(jnp.diag(self.Sigma0)):.2f}"
                        if self.Sigma0 is not None
                        else "None"
                    ],
                ),
            ]
        )

        # Generate the dictionaly
        gen_top_left = []
        for item in top_left.keys():
            gen_top_left.append((item, list(top_left[item]())))

        gen_top_right = []
        for item in top_right.keys():
            gen_top_right.append((item, list(top_right[item]())))


        return gen_top_left, gen_top_right

    def summary(self) -> Summary:
        """Build a ``statsmodels``-style structured summary of the model
        configuration (not of fitted results - see
        `StateSpaceResults.summary` for that).

        Returns
        -------
        statsmodels.iolib.summary.Summary
            Printable summary table with model identity/type/order,
            shape ``(p, q, T)`` and mean system-matrix diagonals.
        """
        self.model = SimpleNamespace()
        self.model.results = jnp.array([0])
        
        self.params = ""
        self.params_names = ""
        self.model.bse = jnp.zeros(len(self.beta)) if self.beta is not None else jnp.array([0])
        self.model.tvalues = jnp.zeros(len(self.beta)) if self.beta is not None else jnp.array([0])
        self.model.pvalues = jnp.zeros(len(self.beta)) if self.beta is not None else jnp.array([0])

        gen_top_left, gen_top_right = self.generate_summary()

        # Generate the summary
        smry = Summary()
        smry.add_table_2cols(
            self,
            title="State Space Model",
            gleft=gen_top_left,
            gright=gen_top_right,
            yname=None,
            xname=None,
        )
  
        return smry

    def __str__(self):
        """String representation of the model."""

        return str(self.summary())

    def __repr__(self):
        return self.__str__()
