"""EM kernels of the spatio-temporal model (internal).

JIT-compiled pure functions used by `LRStateSpaceModel.fit`: initial values,
the closed-form M-step updates of `beta`, `s2e` and `A`, and the `ks`
objective with its compiled value-and-gradient (Section B, P6).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax

from geossm.ssm.statespace import _highest_matmul_precision
from geossm.stmodel._fitted import _spde_marginal_precision


# %% [Utils] Updating formula, JAX (M-Step)

@partial(jax.jit, static_argnums=(2, 3))
@_highest_matmul_precision
def _compute_inital_values_jax_kernel(y_t, Xbeta, block_p, block_q):
    """
    Computes initial parameter values for a model using JAX.

    Args:
        key (jax.random.PRNGKey): The random key for any stochastic operations.
        y_t (jnp.ndarray): The target variable array of shape (ni, T).
        Xbeta (jnp.ndarray): The feature array of shape (ni, b, T).
        block_p (tuple or list): Static list defining blocks for variables.
        block_q (tuple or list): Static list defining blocks for latent factors.
        T (int): Static integer for the number of time steps.

    Returns:
        A tuple of estimated initial values:
        (est_beta, est_s2, est_f, est_x0, est_Sigma0, est_A)
    """
    nvar = len(block_p) - 1
    nlat = len(block_q) - 1  # consider also the alpha

    # --- beta (OLS) ---

    # In JAX, arrays are immutable. Use jnp.where to replace nan/inf values
    # instead of in-place assignment.
    Yt_clean = jnp.nan_to_num(y_t)

    # Reshape X to (ni, T, b)
    Xr = Xbeta.transpose(0, 2, 1)  # (ni, T, b)

    # Compute Xs and ys efficiently using Einstein summation convention
    # This part is identical in syntax to NumPy.
    Xs = jnp.einsum("bij,bik->jk", Xr, Xr)
    ys = jnp.einsum("bij,bi->j", Xr, Yt_clean)

    # Solve for b using jnp.linalg.solve for numerical stability
    est_beta = jnp.linalg.solve(Xs, ys)

    # --- Mean variance of the residuals ---
    # Vectorize the residual calculation instead of using a Python loop.
    # 'ibt,b->it' means: multiply (ni, b, T) with (b,) -> result (ni, T)
    predicted_y = jnp.einsum("ibt,b->it", Xbeta, est_beta)
    res = Yt_clean - predicted_y

    # --- s2 measurement error [1 * p] ---
    # A Python loop over static values (from block_p) is acceptable and will
    # be "unrolled" by the JIT compiler.
    var_res_list = []
    for p in range(nvar):
        # Slicing and computing variance
        var_res_list.append(jnp.var(res[block_p[p] : block_p[p + 1]]))

    var_res = jnp.stack(var_res_list)
    est_s2 = var_res * 0.2

    # --- est_A (Loading Matrix) ---
    # Create the matrix without loops using JAX's functional update syntax.
    diag_vals = jnp.sqrt(var_res * 0.8)
    diag_size = min(nvar, nlat)
    # Create a zero matrix and set the diagonal elements
    est_A = jnp.zeros((nvar, nlat), dtype=y_t.dtype)
    est_A = est_A.at[jnp.arange(diag_size), jnp.arange(diag_size)].set(
        diag_vals[:diag_size]
    )

    # --- Random Initial Values ---
    # JAX requires explicit handling of random number keys.
    # Split the main key for each separate random operation.
    # key, f_key = jax.random.split(key)

    # est_f: Sorted initial values
    # rand_vals = jax.random.uniform(f_key, shape=(nlat,), minval=0.8, maxval=0.9)
    est_f = jnp.repeat(0.8, nlat)  # jnp.flip(jnp.sort(rand_vals))

    # est_x0: Initial state
    est_x0 = jnp.zeros((block_q[-1],), dtype=y_t.dtype)

    # est_Sigma0: Initial state covariance
    est_Sigma0 = 10 * jnp.eye(block_q[-1], dtype=y_t.dtype)

    return est_beta, est_s2, est_f, est_x0, est_Sigma0, est_A


@partial(jit, static_argnames=["b"])
@_highest_matmul_precision
def _compute_beta_jax_kernel(b, y_t, x_T, H, Xbeta):
    """M-step update of the regression coefficients `beta`, accumulating the
    (masked, missing-data-aware) normal equations over time via `lax.scan`
    and solving them once at the end."""

    # 1. Define the function for a single loop iteration (the "scan body")
    # This function is defined inside so it can close over the non-iterating
    # variable `H`.
    def iteration(carry, x):
        # Unpack the carry state and the sliced inputs for this iteration
        Xs, ys = carry
        yt, xt, Xbeta_t = x

        # The logic is identical to the inside of your original for-loop
        na = jnp.isnan(yt)
        valid = ~na

        yt_valid = jnp.where(valid, yt, 0.0)
        Hx = H @ xt
        Hx_valid = jnp.where(valid, Hx, 0.0)

        r = yt_valid - Hx_valid
        # The `valid` mask needs to be broadcast to match Xbeta_t's shape
        Xbeta_t_masked = jnp.where(valid[:, None], Xbeta_t, 0.0)

        # Update the carry state
        ys_new = ys + Xbeta_t_masked.T @ r
        Xs_new = Xs + Xbeta_t_masked.T @ Xbeta_t_masked

        # We don't need to collect per-iteration results, so return None
        return (Xs_new, ys_new), None

    # 2. Prepare the initial state for the carry
    Xs_init = jnp.zeros((b, b), dtype=y_t.dtype)
    ys_init = jnp.zeros((b,), dtype=y_t.dtype)

    # 3. Prepare the data to be scanned over ('xs')
    # lax.scan iterates over the *first* axis. We need to rearrange our data
    # so that the time dimension is axis 0.

    # y_t has shape (N, T), transpose to (T, N)
    y_t_sliced = y_t.T

    # x_T has shape (N, T+1), we need slices from t+1, so we take [:, 1:]
    # and then transpose to (T, N)
    x_T_sliced = x_T[:, 1:].T

    # Xbeta has shape (N, b, T), move axis 2 to axis 0 -> (T, N, b)
    Xbeta_sliced = jnp.moveaxis(Xbeta, 2, 0)

    sliced_data = (y_t_sliced, x_T_sliced, Xbeta_sliced)

    # 4. Run the scan
    # The scan returns the final carry state and any collected outputs
    result, _ = jax.lax.scan(iteration, (Xs_init, ys_init), sliced_data)

    Xs_final, ys_final = result
    beta = jnp.linalg.solve(Xs_final, ys_final)

    return beta


@partial(jit, static_argnames=["block_p"])
@_highest_matmul_precision
def _compute_s2e_jax_kernel(err, H, P_T, block_p):
    """M-step update of the measurement-error variances `s2e`, one value per
    response-variable block, from the smoothed residuals and smoothed state
    covariance (missing observations excluded)."""

    # 1. Prepare data for the time-scan. This is shared across all blocks.
    # We move the time axis to the front for lax.scan.
    # P_T shape: (q, q, T+1) -> we need t+1 slices, so we take [:,:,1:]
    # Resulting shape: (T, q, q)
    P_T_sliced = jnp.moveaxis(P_T[:, :, 1:], 2, 0)

    # err shape: (n, T) -> transpose to (T, n)
    err_sliced = err.T

    # 2. Define the function that computes s2e for a SINGLE block.
    # This is the function we will vectorize with `jax.vmap`.
    # It takes the start and end indices for its block as arguments.
    def compute_block_s2e(i_start, i_end):

        # --- Define the inner loop (scan over time) for this block ---

        # Slice H just once for this block.
        # Since block_p is static, i_start and i_end are constants during
        # compilation, so standard indexing is fine.
        H_p = H[i_start:i_end, :]  # Shape: (m_p, q)

        def iter_time(carry_acc, x_t):
            # Unpack the sliced data for the current time step t
            err_p_t, P_t_plus_1 = x_t

            # The logic is the same as your original compute_single_time
            nna = ~jnp.isnan(err_p_t)
            ep_valid = jnp.where(nna, err_p_t, 0.0)
            Hp_valid = H_p * nna[:, None]

            err_term = ep_valid @ ep_valid
            tr_term = jnp.trace(Hp_valid @ P_t_plus_1 @ Hp_valid.T)

            new_acc = carry_acc + err_term + tr_term
            return new_acc, None

        # --- Execute the logic for the single block ---

        # Slice the time-scannable err for this specific block
        err_p_sliced = err_sliced[:, i_start:i_end]  # Shape: (T, m_p)

        # The data to iterate over in the scan is a tuple of time-sliced arrays
        data_sliced = (err_p_sliced, P_T_sliced)

        # Run the scan over the time dimension starting from 0.0
        temp_p, _ = lax.scan(iter_time, 0.0, data_sliced)

        # Compute the denominator (number of non-NaNs for this block)
        nnaobs = jnp.sum(~jnp.isnan(err[i_start:i_end, :]))

        # Final result for this block, with a safeguard for division by zero
        return jnp.where(nnaobs > 0, temp_p / nnaobs, 0.0)

    # 3. Prepare inputs for vmap
    # We want to call compute_for_one_block for each pair of (start, end) indices.
    starts = block_p[:-1]
    ends = block_p[1:]

    # 4. Vectorize the single-block function over the start and end indices.
    # vmap will effectively call:
    # compute_block_s2e(starts[0], ends[0])
    # compute_block_s2e(starts[1], ends[1])
    # ... and so on, but in a single, efficient, compiled operation.

    s2e = jnp.asarray([compute_block_s2e(s, e) for s, e in zip(starts, ends)])

    return s2e


@partial(jit, static_argnames=["block_p", "block_q", "nvar", "nlat", "T"])
@_highest_matmul_precision
def _compute_A2_jax_kernel(
    y_t, Xbeta, beta, x_T, P_T, block_p, block_q, nvar, nlat, T, ldim, Phi
):
    """
    Fast-compiling replacement for `_compute_A2_jax_kernel`.

    The original kernel builds, *inside a Python loop unrolled over T*, a
    padded block-diagonal Psi_it matrix and Kronecker products to select
    per-domain blocks. That unrolling is what makes compilation blow up as
    T grows: XLA gets T (and T*nlat*nlat) copies of an already sparse/padded
    computation.

    This version keeps only the two loops that are genuinely small and
    static (over variables `nvar` and over latent domains `nlat`), and
    replaces the T-unrolled loop with a `lax.scan`, which is compiled once
    regardless of T. It also drops jnp.kron/block_diag/pad in favor of
    directly slicing the (static) block boundaries block_p/block_q.

    Per row w_i of A (see the derivation in the module docstring):
        g_i[j]   = sum_t resid_i,t' (Phi_i,j @ z_t,j)
        R_i[j,k] = sum_t sum_{a,b} (Phi_i,j' diag(mask_i,t) Phi_i,k)[a,b]
                        * (z_t,j[a] z_t,k[b] + P_t[j,k][a,b])
        w_i = R_i^{-1} g_i
    g_i has no dependence on the observation mask through Phi (only
    through the residual, which is already zeroed at missing entries), so
    it is computed directly with matmuls over all T at once. Only R_i needs
    a per-time contribution (because of the time-varying missing-data mask)
    and uses the scan.
    """
    residual = y_t - jnp.einsum("mpn,p->mn", Xbeta, beta)

    z = x_T[:, 1:]  # (q_total, T)
    P = P_T[:, :, 1:]  # (q_total, q_total, T)

    W_rows = []
    for i in range(nvar):
        p0, p1 = block_p[i], block_p[i + 1]

        y_i = y_t[p0:p1, :]  # (m_i, T)
        mask_i = ~jnp.isnan(y_i)  # (m_i, T)
        resid_i = jnp.where(mask_i, residual[p0:p1, :], 0.0)  # (m_i, T)

        Phi_i = [Phi[p0:p1, block_q[j] : block_q[j + 1]] for j in range(nlat)]
        Z = [z[block_q[j] : block_q[j + 1], :] for j in range(nlat)]  # (q_j, T)

        # V[j][:, t] = Phi_i,j @ z_t,j -- domain j's contribution, all T at once
        V = [Phi_i[j] @ Z[j] for j in range(nlat)]  # (m_i, T)

        # --- g_i: no time loop needed, mask already applied to resid_i ---
        g_i = jnp.stack([jnp.sum(resid_i * V[j]) for j in range(nlat)])

        # --- R_i: mask is time-varying, so scan over T (compiled once) ---
        mask_scan = jnp.moveaxis(mask_i, 1, 0)  # (T, m_i)
        z_scan = jnp.moveaxis(z, 1, 0)  # (T, q_total)
        P_scan = jnp.moveaxis(P, 2, 0)  # (T, q_total, q_total)

        def step(R_acc, xs, Phi_i=Phi_i):
            mask_t, z_t, P_t = xs
            R_new = R_acc
            for j in range(nlat):
                Phi_ij_masked = Phi_i[j] * mask_t[:, None]
                zj = z_t[block_q[j] : block_q[j + 1]]
                for k in range(nlat):
                    D_jk = Phi_ij_masked.T @ Phi_i[k]  # (q_j, q_k)
                    zk = z_t[block_q[k] : block_q[k + 1]]
                    P_jk = P_t[block_q[j] : block_q[j + 1], block_q[k] : block_q[k + 1]]
                    contrib = jnp.sum(D_jk * (jnp.outer(zj, zk) + P_jk))
                    R_new = R_new.at[j, k].add(contrib)
            return R_new, None

        R0 = jnp.zeros((nlat, nlat), dtype=y_t.dtype)
        R_i, _ = lax.scan(step, R0, (mask_scan, z_scan, P_scan))

        w_i = jnp.linalg.solve(R_i, g_i)
        W_rows.append(w_i)

    return jnp.stack(W_rows, axis=0)


def _ks_objective(params, T, Omega, mass_list, stiff_list, perm_list, n_inner):
    """M-step objective for `ks = exp(params)`: the `ks`-dependent part of the
    expected complete-data negative log-likelihood, -T log|Q^-1| + tr(Q^-1 Omega),
    scaled by 1e-4 to keep its magnitude moderate for the optimiser."""
    ks = jnp.exp(params)  # unconstrained optimisation, ks > 0 by construction
    invQ = _spde_marginal_precision(ks, mass_list, stiff_list, perm_list, n_inner)
    logdet_invQ = jnp.linalg.slogdet(invQ)[1]
    fun = -T * logdet_invQ + jnp.trace(invQ @ Omega)
    return fun / 1e4


# Compiled once per process (per array shapes), not once per fit() (F20); full
# matmul precision for the forward and backward pass (F31).
_ks_value_and_grad = jax.jit(
    _highest_matmul_precision(jax.value_and_grad(_ks_objective)), static_argnums=(6,)
)
