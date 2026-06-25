"""
Adapter scaffolding making the project's StateSpaceModel usable with
statsmodels' MLEModel API.
"""
import numpy as np
import jax.numpy as jnp

# inmport the state space model
from geossm.ssm import StateSpaceModel
from geossm.covmodel import spdeAppoxCov
from statsmodels.iolib.summary import Summary

from jax import jit, lax
import scipy.sparse as sp
from functools import partial

from datetime import datetime, timezone
import time
from jax.scipy.linalg import block_diag
from scipy.linalg import block_diag as scyp_block_diag
import jax
from scipy.optimize import minimize

from geossm import DesignMatricesBuilder
from geossm import block_diag_3D

from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from dataclasses import replace
from geossm.stmodel import Param, FitOptions, ModelParams

from .stmodel_results import LRStateSpaceResults
from types import SimpleNamespace

# % [Utils] Updating formula, JAX (M-Step)


@partial(jax.jit, static_argnums=(2, 3))
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
    est_A = jnp.zeros((nvar, nlat))
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
    est_x0 = jnp.zeros((block_q[-1],))

    # est_Sigma0: Initial state covariance
    est_Sigma0 = 10 * jnp.eye(block_q[-1])

    return est_beta, est_s2, est_f, est_x0, est_Sigma0, est_A


@partial(jit, static_argnames=["b"])
def _compute_beta_jax_kernel(b, y_t, x_T, H, Xbeta):

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
    Xs_init = jnp.zeros((b, b))
    ys_init = jnp.zeros((b,))

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
def _compute_s2e_jax_kernel(err, H, P_T, block_p):

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


# @partial(jit, static_argnames=["block_p", "block_q", "nvar", "nlat", "T"])
# def _compute_A2_jax_kernel(
#     y_t, Xbeta, beta, x_T, P_T, block_p, block_q, nvar, nlat, T, ldim, Phi
# ):
#     """
#     Optimized JAX implementation using einsum and vmap.
#     Assumes blocks are uniform or padded to fixed sizes.
#     """
#     T = y_t.shape[1]
#     nvar = len(block_p) - 1
#     nlat = x_T.shape[0] - 1 # Assuming x_T includes intercept or similar
    
#     # 1. Compute Residuals: (Total_M, T)
#     # y_t is (Total_M, T), Xbeta is (Total_M, P, T), beta is (P,)
#     # If Xbeta is already (Total_M, T), use it directly.
#     residual = y_t - jnp.einsum("mpt,p->mt", Xbeta, beta)
#     # residual = y_t - jnp.dot(Xbeta, beta) # simplified based on your snippet
    
#     # Handle NaNs: create mask and zero out NaNs in residuals
#     mask = jnp.where(jnp.isnan(residual), 0.0, 1.0)
#     residual_clean = jnp.nan_to_num(residual)

#     # 2. Prepare Latent States
#     # z_t = x_T[:, 1:] (Shape: nlat, T)
#     # M_t = z_t z_t' + P_t (Shape: T, nlat, nlat)
#     z_t = x_T[:, 1:] 
#     P_t = P_T[:, :, 1:] # (nlat, nlat, T)
    
#     # Outer product zz' over all T: (nlat, nlat, T)
#     ZZ_T = jnp.einsum('it,jt->ijt', z_t, z_t) + P_t

#     # Initialize result matrix
#     W = jnp.zeros((nvar, nlat), dtype=y_t.dtype)

#     # 3. Loop over blocks (i)
#     # Since nvar is static, this loop is unrolled during JIT compilation.
#     # Because 'i' is a concrete integer here, slicing block_p[i] works.
#     for i in range(nvar):
#         # Slice indices for current block i
#         start_p, end_p = block_p[i], block_p[i+1]
        
#         # Phi_i shape: (mdim_i, nlat)
#         # Note: Your block_q logic suggests Phi is sliced by both i and j
#         # But if Phi maps factors to variables, it's usually (mdim_i, nlat)
#         Phi_i = Phi[start_p:end_p, :] 
#         Resid_i = residual_clean[start_p:end_p, :] * mask[start_p:end_p, :] # (mdim_i, T)

#         # --- Compute g_i ---
#         # Equation: sum_t (Resid_i_t' * Phi_i_j * z_t_j)
#         # jnp.einsum('mt, mj, jt -> j', Resid_i, Phi_i, z_t)
#         # m: mdim_i, t: T, j: nlat
#         g_i = jnp.einsum('mt, mj, jt -> j', Resid_i, Phi_i, z_t)

#         # --- Compute R_i ---
#         # Equation: sum_t (Phi_i_j' * Phi_i_k) * ZZ_T_jk_t
#         # m: mdim_i, j: nlat, k: nlat, t: T
        
#         # First, precompute dot products of Phi columns: (nlat, nlat)
#         # However, because of the mask, we need to account for 't' in the Phi dot product
#         # Mask_i is (mdim_i, T)
#         Mask_i = mask[start_p:end_p, :]
        
#         # Weighted Phi-Phi product: (nlat, nlat, T)
#         # For every t, compute (Phi * mask_t)' @ (Phi * mask_t)
#         Phi_dot_Phi_t = jnp.einsum('mj, mk, mt -> jkt', Phi_i, Phi_i, Mask_i)
        
#         # Now contract with ZZ_T
#         R_i = jnp.einsum('jkt, jkt -> jk', Phi_dot_Phi_t, ZZ_T)

#         # Solve for w_i: R_i w_i = g_i
#         # Add a small epsilon to diagonal for stability if needed
#         w_i = jnp.linalg.solve(R_i, g_i)
        
#         W = W.at[i, :].set(w_i)
    
#     return W

@partial(jit, static_argnames=["block_p", "block_q", "nvar", "nlat", "T"])
def _compute_A2_jax_kernel(
    y_t, Xbeta, beta, x_T, P_T, block_p, block_q, nvar, nlat, T, ldim, Phi
):
    """
    Rewritten version of compute_A2 to work with jax.numpy and jax.jit,
    considering 'block_p', 'block_q', 'nvar', 'nlat', 'T', 'max_mdim_i' as static arguments.
    Fixes NonConcreteBooleanIndexError by padding to max_mdim_i and
    using jax.scipy.linalg.block_diag for static shapes.
    """
    # Precompute fixed effects and residuals for all data points
    # Shape (total_mdim, T)
    residual = y_t - jnp.einsum("mpn,p->mn", Xbeta, beta)

    # Initialize W (W will hold the results)
    W = jnp.zeros((nvar, nlat), dtype=y_t.dtype)

    # Loop over nvar (i) - nvar is a static argument
    for i in range(nvar):
        R_it = jnp.zeros((nlat, nlat), dtype=y_t.dtype)  # nlat is static
        g_it = jnp.zeros((1, nlat), dtype=y_t.dtype)  # nlat is static

        # Get the actual number of observations for the current block 'i'
        # mdim_i is static for a given `i` but can vary between `i` blocks.
        # max_mdim_i is the global maximum used for padding.
        mdim_i = block_p[i + 1] - block_p[i]
        max_mdim_i = mdim_i

        # Loop over time steps T (t) - T is a static argument
        for t in range(T):
            # Extract the slice of y_t for the current block and time, then get NaN mask
            y_t_slice_full = y_t[block_p[i] : block_p[i + 1], t]  # Shape (mdim_i,)
            # Boolean mask, shape (mdim_i,)
            nna_mask = ~jnp.isnan(y_t_slice_full)

            # Get the full residual for the current block and time
            residual_full_slice = residual[
                block_p[i] : block_p[i + 1], t
            ]  # Shape (mdim_i,)

            # Mask and pad residual_full_slice
            # Multiply by mask to zero out invalid entries, then pad to max_mdim_i
            # This ensures ep_valid_padded always has shape (max_mdim_i,)
            ep_valid_padded = jnp.pad(
                residual_full_slice * nna_mask,  # (mdim_i,) * (mdim_i,)
                # Pad only the first dimension
                (0, max_mdim_i - mdim_i),
                constant_values=0.0,
            )  # (max_mdim_i,)

            # --- Compute Psi_it (fixed shape) ---
            temp_padded_blocks = []
            # Assuming q_j_dim is constant, e.g., 1, as per common use with block_q
            # q_j_dim = block_q[j+1] - block_q[j]
            # This needs to be consistent, so let's use the first block_q diff for column padding.
            # Or ensure block_q always yields same q_j_dim for consistent Phi slicing.
            # Assuming block_q[j+1]-block_q[j] is always the same for all j (e.g. 1)
            # as in your example: block_q = jnp.array([0, 1, 2, nlat]) implies q_j_dim=1.
            # Get the dimension of a single q block
            # q_j_dim = block_q[1] - block_q[0]

            for j in range(nlat):  # nlat is static
                # Current block of Phi, shape (mdim_i, q_j_dim)
                phi_block_current = Phi[
                    block_p[i] : block_p[i + 1], block_q[j] : block_q[j + 1]
                ]

                # Apply mask by multiplying (zeros out rows corresponding to NaNs)
                # nna_mask[:, jnp.newaxis] broadcasts the mask to all columns of phi_block_current
                phi_block_masked = phi_block_current * nna_mask[:, jnp.newaxis]

                # Pad this masked block to (max_mdim_i, q_j_dim) for static shapes
                padded_phi_block = jnp.pad(
                    phi_block_masked,
                    ((0, max_mdim_i - mdim_i), (0, 0)),
                    constant_values=0.0,
                )
                temp_padded_blocks.append(padded_phi_block)

            # Construct the block diagonal matrix using jax.scipy.linalg.block_diag
            # If each q_j_dim is 1, then the total columns is nlat.
            # The total rows will be sum of rows from each block, i.e., nlat * max_mdim_i.
            # So, Psi_it now has shape (nlat * max_mdim_i, nlat) (if q_j_dim=1 for all j)
            Psi_it = jax.scipy.linalg.block_diag(*temp_padded_blocks)

            # --- Update g_it ---
            kron_terms = []
            for j in range(nlat):
                # Use max_mdim_i directly for jnp.eye to ensure static shape
                # This identity matrix now has a fixed size.
                # Shape (max_mdim_i, max_mdim_i)
                tt = jnp.eye(max_mdim_i, dtype=y_t.dtype)

                # kron_term_val_j shape: (max_mdim_i, nlat * max_mdim_i)
                kron_term_val_j = jnp.kron(_ei_jax(j, nlat), tt)

                # Check for compatibility between ldim and nlat for Psi_it @ x_T
                # This is an implicit assumption from your original code's structure.
                # Shape: (nlat * max_mdim_i, 1)
                Psi_times_xT = Psi_it @ x_T[:, [t + 1]]

                # Product shape: (max_mdim_i, nlat * max_mdim_i) @ (nlat * max_mdim_i, 1) = (max_mdim_i, 1)
                kron_terms.append(kron_term_val_j @ Psi_times_xT)

            # Concatenate these (max_mdim_i, 1) vectors horizontally to form (max_mdim_i, nlat) matrix
            combined_H_times_x = jnp.concatenate(
                kron_terms, axis=1
            )  # Shape: (max_mdim_i, nlat)

            # Add to g_it: ep_valid_padded.T is (1, max_mdim_i)
            # Result: (1, max_mdim_i) @ (max_mdim_i, nlat) = (1, nlat) -> Matches g_it's shape
            g_it += ep_valid_padded[jnp.newaxis, :] @ combined_H_times_x

            # --- Update R_it ---
            # Common term for R_it update, shape: (ldim, ldim)
            x_T_x_T_P = x_T[:, [t + 1]] @ x_T[:, [t + 1]].T + P_T[:, :, t + 1]

            # Nested loops for j and k
            for j_idx in range(nlat):
                for k_idx in range(nlat):
                    # Use max_mdim_i directly for jnp.eye for static shape
                    tt_jk = jnp.eye(max_mdim_i, dtype=y_t.dtype)
                    # Kron term: shape (nlat * max_mdim_i, nlat * max_mdim_i)
                    kron_op_jk = jnp.kron(
                        _ei_jax(j_idx, nlat).T @ _ei_jax(k_idx, nlat), tt_jk
                    )

                    # Calculate the term for trace: Psi_it.T @ kron_op_jk @ Psi_it @ (x_T @ x_T.T + P_T)
                    # Psi_it.T is (nlat, nlat * max_mdim_i)
                    # kron_op_jk is (nlat * max_mdim_i, nlat * max_mdim_i)
                    # Psi_it is (nlat * max_mdim_i, nlat)
                    # x_T_x_T_P is (ldim, ldim) -> (nlat, nlat) based on assertion

                    # The full product chain:
                    # (nlat, nlat * max_mdim_i) @ (nlat * max_mdim_i, nlat * max_mdim_i)
                    # -> (nlat, nlat * max_mdim_i) @ (nlat * max_mdim_i, nlat)
                    # -> (nlat, nlat) @ (nlat, nlat)
                    # -> (nlat, nlat)
                    term_to_trace = Psi_it.T @ kron_op_jk @ Psi_it @ x_T_x_T_P

                    R_it = R_it.at[j_idx, k_idx].add(jnp.trace(term_to_trace))

        # Solve linear system for W row
        w_i_row = jnp.linalg.solve(R_it, g_it.T).reshape(
            nlat,
        )

        # Update W (immutable operation using .at[].set())
        W = W.at[i, :].set(w_i_row)

    return W


# Helper function ei for JAX
def _ei_jax(i, dim):
    """
    Creates a one-hot row vector for JAX.
    """
    return jax.nn.one_hot(i, dim, dtype=jnp.float32).reshape(1, dim)


# % Low Rank State-Space Model adapter to statsmodels MLEModel API


class LRStateSpaceModel(StateSpaceModel):

    def __init__(self, df, formulas:list = None, domain:list = None, verbose=True):

        self.df = df.copy()
        self.formulas = formulas
        self.verbose = verbose

        # Overwrite the model attributes
        self.model_name = "Low-Rank State Space Model"
        self.type = "Linear (Gaussian)"
        self.order = (
            1,
            0,
        )  # ARMA order (p, d, q) - not used in this model but kept for compatibility

        self._log(f"Initializing {self.model_name}...")
        self._log(f"Model type: {self.type}")
        self._log(f"Model order: {self.order}")

        if formulas is not None:

            # Compute the design matrices
            self._log("Building observation grid...")


            self.nvar, self.points, self.gridList, self.ndim, self.pdim, self.block_p, T = (
                self._buildObservationGrid(df, formulas, verbose=verbose)
            )

            self._log("Building observation grid... Done.")

            self._log("Building the design matrix...")

            # self.train will be used later for estimation and for the results
            # Xbeta_train -> Xbeta in the parant class, y_train -> y_obs in the parent class
            self.y_train, Xbeta = self._buildDesignMatrix(self.gridList)

            # get response name
            self.y_name = [g.y_name for g in self.gridList]
            xbeta_names = [g.x_names for g in self.gridList]

            self._log("Building the design matrix... Done.")

        else:
            self._log("Formulas not provided. The model will be initialized without them")

            Xbeta = None
            xbeta_names = None


        # Check the domain
        self._log(f"Checking {len(domain)} domains (start)...")

        flag, msg = self._checkDomain(domain)
        if flag:
            raise ValueError(msg)
        else:
            self._domain = self._setDomain(domain)
            self._log(f"{len(self._domain)} valid domains found and set.")
        self._log(f"Checking {len(domain)} domains (done)")

        self._cov_matern = None

        # Inizialisate the StateSpaceModel as a null model (we will set the parameters later)
        # y_train will be used later for estimation and for the results
        super().__init__(Xbeta=Xbeta, beta=None, xbeta_names=xbeta_names)

    @property
    def domain(self):
        return self._domain

    def setup(self, mesh_obj: list = None, cov_fun: list = None, domain_latent: list = None):

        # this domain is the domain on which the mesh is defined, and where the
        # covariance function has a meaning
        if domain_latent is not None:
            self._log(f"Checking {len(domain_latent)} domains (start)...")
            flag, msg = self._checkDomain(domain_latent)
            if flag:
                raise ValueError(msg)
        else:
            domain_latent = self._domain
            self._log(f"{len(self._domain)} valid domains found and set.")
        self._log("Checking domains (done)")

        # mehs_obj = list of the latent domain
        self._cov_matern = []

        # Check the consistency of the inputs
        if mesh_obj is None and cov_fun is None:
            raise ValueError("Or mesh_obj or cov_fun must be provided")

        if mesh_obj is not None:
            # If mesh_obj is provided, we create the covariance model of the matern for each domain,
            # and we store it in the list _cov_matern
            self._log(f"Checking {len(mesh_obj)} mesh objects...")
            
            if len(mesh_obj) != len(domain_latent):
                raise ValueError(
                    f"Number of mesh objects ({len(mesh_obj)}) must match number of domains ({len(domain_latent)})"
                )

            for i, (meshi, domi) in enumerate(zip(mesh_obj, domain_latent)):

                line = len(meshi.cells_dict["line"])
                vertex = len(meshi.cells_dict["vertex"])
                triangle = len(meshi.cells_dict["triangle"])
                
                self._log(f"Create the GMRF {i} object, with (line: {line},triangle: {triangle},vertex: {vertex})")
                # create the covariance model of the matern
                temp = spdeAppoxCov([domi], latlon=False, nu=1.0, var=1.0, rescale=1.0)
                self._cov_matern.append(temp.setup(meshi))

        elif cov_fun is not None:
            # If cov_fun is provided, we check that it is a list of covariance functions of the same length
            # as the number of domains, and we store it in the list _cov_matern
            self._log(f"Checking {len(cov_fun)} covariance functions (start)...")

            if len(cov_fun) != len(domain_latent):
                raise ValueError(
                    f"Number of covariance functions ({len(cov_fun)}) must match number of domains ({len(domain_latent)})"
                )

            for i, (covi, domi) in enumerate(zip(cov_fun, domain_latent)):
                # self._log(f"Cov_fun-{i} covariance function...")

                # check if the covariance function is an instance of spdeAppoxCov
                if not isinstance(covi, spdeAppoxCov):
                    raise ValueError(
                        "Covariance function must be an instance of spdeAppoxCov"
                    )
                self._log(f"Cov.Fun.-{i}: rescale = {covi.rescale}, nu = {covi.nu}, var = {covi.var}")
                self._cov_matern.append(covi)
            
            
            self.qdim = jnp.array([cov.fem_solver.n_inner_points for cov in self._cov_matern],dtype=jnp.int32)
        
            self.block_q = jnp.hstack((0, jnp.cumsum(self.qdim)))
            self._log(f"Set the latent dimension (q) to {self.block_q[-1]}")
            self._log("Checking covariance functions (done)")




        else:
            raise ValueError("Invalid input: either mesh_obj or cov_fun must be provided")

        # set the number of latent factors (i.e. the number of covariance functions)
        self.nlat = len(self._cov_matern)

        return self

    @property
    def shape(self):
        p = None
        q = None
        T = None

        if self.y_train is not None:
            p, T = self.y_train.shape
        
        if self.Xbeta is not None:
            p, b, T = self.Xbeta.shape
        
        if self._cov_matern is not None and len(self._cov_matern) > 0:
            q = sum([cov.fem_solver.n_inner_points for cov in self._cov_matern])

        return p, q, T



    def sim(self, formulas:list = None , seed=1234, params: ModelParams = None, verbose=None, stats=False):
        
        if formulas is None and self.formulas is None:
            raise ValueError("Formulas must be provided for simulation")
        
        if formulas is None:
            formulas = self.formulas
            Xbeta = self.Xbeta
            xbeta_names = self.xbeta_names
            y_name = self.y_name
            nvar = self.nvar
            nlat = self.nlat
            pdim = self.pdim
            block_p = self.block_p
            points = self.points
            T = self.T
            self._log("Using the formulas provided at initialization, lenght = {}.".format(len(formulas)))
        else:
            self._log("Building observation grid...")
            
            nvar, points, gridList, ndim, pdim, block_p, T = (
                self._buildObservationGrid(self.df, formulas, verbose=verbose)
            )
            self._log("Building observation grid... Done.")


            # self.train will be used later for estimation and for the results
            # Xbeta_train -> Xbeta in the parant class, y_train -> y_obs in the parent class
            self._log("Building the design matrix...")
            y, Xbeta = self._buildDesignMatrix(gridList)
            
            y_name = [g.y_name for g in gridList]
            xbeta_names = [g.x_names for g in gridList]


            # get response name
            # self.y_name = [g.y_name for g in self.gridList]
            # self.xbeta_names = [g.x_names for g in self.gridList]

            self._log("Building the design matrix... Done.")
        
        # check if the covariance function is defined
        if self._cov_matern is None or len(self._cov_matern) == 0:
            raise ValueError("Covariance function is not defined. Please run the setup method first.")
        else:
            qdim = self.qdim
            block_q = self.block_q
            nlat = self.nlat

        
        # Get the model parameters (if not provided, they will be set to None and the model will use default initial values)
        self._log("Parsing the parameters (ModelParams|None)...")
        
        params = self._parseParams(params)
        A = params.A.value
        s2e = params.s2e.value
        f = params.f.value
        beta = params.beta.value
        ks = params.ks.value

        # TODO: check if the parameters are valid (e.g. positive variance, etc.)
        
        # Chech the length of beta
        if beta is not None and len(beta) != Xbeta.shape[1]:
            raise ValueError(f"Length of beta ({len(beta)}) must match number of columns in Xbeta ({Xbeta.shape[1]})")
        
        # Check the lenght of A
        if A is None or A.shape[0] != nvar or A.shape[1] != self.nlat:
            raise ValueError(f"Shape of A ({A.shape}) must match (nvar, nlat) = ({nvar}, {self.nlat})")
        
         # Check the lenght of ks
        if ks is None or len(ks) != len(self._cov_matern):
            raise ValueError(f"Length of ks ({len(ks)}) must match number of covariance functions ({len(self._cov_matern)})")
        
        # set the covarriance rescale paramiter to the value of ks (if provided, otherwise it will be set to 1)
        if ks is not None:
            for cov, ksi in zip(self.cov_function, ks):
                cov.rescale = ksi
        
        self._log("Parsing the parameters (ModelParams|None)... Done.")
        
        # Get the dimensions of the latent variable
        qdim = np.sum(self.block_q)

        self._log("Computing the SSM model matrices H, R, F and Q...")

        # Compute the basis matrix (just one) - no boundary
        basis = self._buildBasis_list(points, self.cov_function)
        
        # ---- build parametrised matrices
        H = self._buildH_dense(A, basis)  # dense
        self._log("Computing the H {} matrix... Done.".format(H.shape))

        # R, F = buildRF(est_s2e, est_f, pdim, qdim)
        R, F = self._buildRF_dense(s2e, f, pdim, qdim)
        self._log("Computing the R {} matrix... Done.".format(R.shape))
        self._log("Computing the F {} matrix... Done.".format(F.shape))


        # Compute the maginal precision matrix
        invQ = []
        for fcov in self.cov_function:
            invQi = fcov.precision()

            # index of the inner points (i.e. the points of the latent domain)
            inx = fcov.fem_solver.inner
            Q_11 = invQi[inx, :][:, inx]
            Q_12 = invQi[inx, :][:, ~inx]
            Q_22 = invQi[~inx, :][:, ~inx]

            # Marginal precision matrix of the inner points (i.e. the points of the latent domain)
            Q_mar = Q_11 - Q_12 @ np.linalg.inv(Q_22.toarray()) @ Q_12.T

            invQ.append(Q_mar)

        # Compute the block diagonal covariance matrix Q of the latent factors (i.e. the points of the latent domain)
        Q = block_diag(
            *[
                jnp.linalg.solve(mt, jnp.eye(mt.shape[0], dtype=jnp.float32))
                for mt in invQ
            ]
        )
        self._log("Computing the Q {} matrix... Done.".format(Q.shape))

        self._log("Start simulating the SSM...")

        # Simulate the SSM using the parent class method (we need to pass the parameters to it)
        # Create a new SSM with the same parameters as the current model, but with the matrices H, R, F and Q computed above

    
        sim_model = StateSpaceModel(H=H, R=R, F=F, Q=Q,Xbeta=Xbeta, beta=beta, xbeta_names=xbeta_names, x0=None, Sigma0=None)

        y_sim, x_sim, variance_stats, tdelta = sim_model.sim(
            seed, R=R, F=F, H=H, Q=Q, Xbeta=Xbeta, beta=beta, block_p=block_p, block_q=self.block_q, stats=stats, verbose=verbose
        )

        self._log("Simulation done. Time elapsed: {}.".format(tdelta))

        info = {}
        info['formulas'] = formulas
        info['y_name'] = y_name
        info['xbeta_names'] = xbeta_names
        info['Xbeta'] = Xbeta
        info['params'] = params
        info['points'] = points
        info['T'] = T
        info['stats'] = variance_stats
        info['qdim'] = qdim
        info['nvar'] = nvar
        info['nlat'] = nlat
        info['pdim'] = pdim
        info['block_p'] = block_p
        info['block_q'] = block_q
        info['sim_model'] = sim_model

        return y_sim, x_sim, info, tdelta    
    
    def predict(self, df, modelresults: LRStateSpaceResults, verbose = True):
        """
        Internal method to predict the response variable for the given points (or all points if None) using the fitted model parameters.
        """ 
        self._log("Predicting response variable...")

        # Cut the dataframe time to the time range of the model results
        self._log("Cutting the dataframe to the time range of the model results...")
        tmin = self.gridList[0].timestamps.min()
        tmax = self.gridList[0].timestamps.max()

         
        # Compute the design matrices
        self._log("Building observation grid...")

        nvar, points, gridList, ndim, pdim, block_p, T = (
            self._buildObservationGrid(df, self.formulas, predict = True, verbose=verbose, tmin=tmin, tmax=tmax)
        )
        
        if nvar != self.nvar:
            raise ValueError(f"Number of response variables in the input data ({nvar}) does not match the model's number of response variables ({self.nvar}).")
    
        self._log("Building Prediction grid... Done.")

        self._log("Building the design matrix...")

        # self.train will be used later for estimation and for the results
        # Xbeta_train -> Xbeta in the parant class, y_train -> y_obs in the parent class
        _, Xbeta_predict = self._buildDesignMatrix(gridList)
        

        self._log("Building the design matrix... Done.")

        if modelresults is None:
            raise ValueError("Model results must be provided for prediction")

        self._log("Parsing the model results (LRStateSpaceResults)...")
        params = modelresults.params
        beta = params.beta.value
        A = params.A.value
        ks = params.ks.value

        self._log("Get the filtered state")
        x_T = modelresults.x_smoothed
        P_T = modelresults.P_smoothed

        # update the cov_function rescale
        for cov, ksi in zip(self.cov_function, ks):
            cov.rescale = ksi
        
        self._log("Computing the SSM model matrice H...")

        # Compute the basis matrix (just one) - no boundary
        basis = self._buildBasis_list(points, self.cov_function)
        
        # ---- build parametrised matrices
        H = self._buildH_dense(A, basis)  # dense
        self._log("Computing the H {} matrix... Done.".format(H.shape))

    
        self._log("Start Prediction the SSM...")
        tStart = time.time()
        y_hat_full, Sigma_y_hat_full = super().predict(H, x_T, P_T, Xbeta_predict, beta)
        tdelta = time.time()- tStart

        self._log("Simulation done. Time elapsed: {}.".format(tdelta))

        # return the results as a list (same lengh of points and block_p)
        block_p = self.block_p
        y_hat = []
        Sigma_y_hat = []
        for i in range(len(block_p)-1):
            y_hat.append(y_hat_full[block_p[i]:block_p[i+1], :])
            Sigma_y_hat.append(Sigma_y_hat_full[block_p[i]:block_p[i+1], block_p[i]:block_p[i+1],:])

        return points, y_hat, Sigma_y_hat, tdelta
            

    

    def fit(
        self, params0: ModelParams | None = None, options: FitOptions | None = None
    ):

        # set the global options
        self.verbose = options.verbose if options is not None else True
        
        smr = self.summary(print_full = False)
        if self.verbose:
            print(smr)
        self._log("Starting the estimation of the model parameters using EM algorithm...")

        
        self._log("Parsing the fit options (FitOptions|None)...")

        max_iter = options.max_iter if options is not None else 100
        tol_relat = options.tol_relat if options is not None else 1e-3
        dtype = options.dtype if options is not None else jnp.float32

        # Get the initial parameters (if not provided, they will be set to None and the model will use default initial values)
        # Create the est_params object, filling in the provided values and leaving the rest as None (or default) for the model to handle
        self._log("Parsing the initial parameters (ModelParams|None)...")
        params0 = self._parseParams(params0)

        # Get global constants
        nvar = self.nvar  # len(self.pdim)
        nlat = self.nlat
        # cov_function = self.cov_function
        pdim = jnp.asarray(self.pdim, dtype=jnp.int32)
        block_p = jnp.asarray(self.block_p, dtype=jnp.int32)

        # Get the observed data
        y_obs = jnp.asarray(self.y_train, dtype=dtype)
        Xbeta = jnp.asarray(self.Xbeta, dtype=dtype)

        points = self.points
        p, T = y_obs.shape

        # Get latent dimension (i.e. the rank)
        qdim = jnp.array(
            [cov.fem_solver.n_inner_points for cov in self.cov_function],
            dtype=jnp.int32,
        )
        block_q = jnp.hstack((0, jnp.cumsum(qdim)))

        # Get the initial values
        self._log("Computing the initial parameter values...")
        est_params = self._getInitialValues(y_obs, Xbeta, block_p, block_q)

        # Set the initial values of the parameters (if not provided, they will be set to the estimated initial values)
        self._log("Updating the initial parameter values...")
        est_params = self._updateParams0(params0, est_params)

        # Compute the basis matrix (just one) - no boundary
        
        self._log("Computing the basis matrix...")
        basis = self._buildBasis_list(points, self.cov_function)
        Phi = self._buildH_dense(jnp.ones((nvar, nlat), dtype=jnp.float32), basis)

        self._log("Starting the EM iterations...")

        # Flag of the EM convergence
        flag = True
        niter = 0
        logL_prev = 0
        logL_cur = 0
        delta_par = jnp.nan
        delta_lik = jnp.nan
        relat_lik = jnp.nan
        tdelta_iter = 0
        tdelta_Edet = np.zeros(3)
        tdelta_Mdet = np.zeros(3)
        nstat = []  # list to store the results of each iteration

        # Log the initial state before starting the EM iterations
        nstat = self._log_iteration(
            nstat,
            niter,
            est_params,
            0,
            logL_cur,
            delta_par,
            delta_lik,
            relat_lik,
            tdelta_iter,
            tdelta_Edet,
            tdelta_Mdet,
        )

        if self.verbose:
            msg = self.logger(nstat[-1])
            print(msg)

        # Start EM iteration
        while flag:
            niter += 1

            # Start the timer for the iteration
            tStart_iter = time.time()

            # ---- build parametrised matrices
            H = self._buildH_dense(est_params.A.value, basis)  # dense

            # R, F = buildRF(est_s2e, est_f, pdim, qdim)
            R, F = self._buildRF_dense(
                est_params.s2e.value, est_params.f.value, pdim, qdim)

            # Compute the maginal precision matrix
            invQ = []
            for fcov in self.cov_function:
                invQi = fcov.precision()

                # index of the inner points (i.e. the points of the latent domain)
                inx = fcov.fem_solver.inner
                Q_11 = invQi[inx, :][:, inx]
                Q_12 = invQi[inx, :][:, ~inx]
                Q_22 = invQi[~inx, :][:, ~inx]

                # Marginal precision matrix of the inner points (i.e. the points of the latent domain)
                Q_mar = Q_11 - Q_12 @ np.linalg.inv(Q_22.toarray()) @ Q_12.T

                invQ.append(Q_mar)

            # Compute the block diagonal covariance matrix Q of the latent factors (i.e. the points of the latent domain)
            Q = block_diag(
                *[
                    jnp.linalg.solve(mt, jnp.eye(mt.shape[0], dtype=jnp.float32))
                    for mt in invQ
                ]
            )

            # Inizialisate the SSM with the current parameters (we need it for the E step)
            super().set(
                H=H,
                R=R,
                F=F,
                Q=Q,
                x0=est_params.x0.value,
                Sigma0=est_params.Sigma0.value,
                beta=est_params.beta.value,
            )

            # ---- E step
            # y_hat, x_T, P_T, S11, S10, S00, logL_cur, tdelta_Edet = self._E_step(
            #    y_obs, R, F, H, Q, est_params.x0.value, est_params.Sigma0.value, Xbeta, est_params.beta.value)

            y_hat, x_T, P_T, S11, S10, S00, logL_cur, tdelta_Edet = self._E_step(y_obs)

            # ---- M step, get the updated parameters
            update_params, opt_success, tdelta_Mdet = self._M_step(
                y_obs,
                y_hat,
                self.F,
                self.H,
                Xbeta,
                self.cov_function,
                block_p,
                block_q,
                x_T,
                P_T,
                S11,
                S10,
                S00,
                Phi,
            )

            # Compute the delta log likelihood ( 0 < current - previous < tol_lik )
            delta_lik = logL_cur - logL_prev
            relat_lik = delta_lik / abs(logL_prev) if logL_prev != 0 else jnp.inf

            # End the timer for the iteration
            tdelta_iter = time.time() - tStart_iter

            # Update the paramiters to be used in the next EM iteration
            est_params = self._updateParams(est_params, update_params)
            logL_prev = logL_cur

            # append the results
            nstat = self._log_iteration(
                nstat,
                niter,
                est_params,
                opt_success,
                logL_cur,
                delta_par,
                delta_lik,
                relat_lik,
                tdelta_iter,
                tdelta_Edet,
                tdelta_Mdet,
            )

            # print the results of the iteration
            if self.verbose:
                msg = self.logger(nstat[-1])
                print(msg)

            # Check the EM convergence (if the log-likelihood is not improving more than tol_lik or the max number of iterations is reached)
            if niter == max_iter or relat_lik <= tol_relat:
                flag = False


        self._log("EM algorithm converged after {} iterations.".format(niter))
        self._log("Final log-likelihood: {}.".format(logL_cur))
        self._log("Create the results object...")
        
        results = LRStateSpaceResults(
            model=self, 
            params=est_params, 
            nstats=nstat, 
            options=options,
            # main arrays
            y_hat=y_hat, 
            x_smoothed=x_T,
            P_smoothed=P_T,
            P_pred_smoothed=None,
            # sufficient statistics
            S11=S11,
            S10=S10,
            S00=S00,
            )

        return results

    @property
    def cov_function(self):
        return self._cov_matern

    def _E_step(self, y_t):

        # E step: compute the expected values of the latent factors and the log-likelihood
        # 1) Create the SSM object with the current parameters
        # 2) Run the Kalman filter and smoother to get the expected values of the latent factors and the log-likelihood

        # Run the Kalman filter and smoother to get the expected values of the latent factors and the log-likelihood
        # Call the parent class's estimate method to perform the Kalman filter and smoother
        results = super().estimate(y_t)

        y_hat = results.y_hat
        x_T = results.x_smoothed
        P_T = results.P_smoothed
        # P_T_1 = results.P_pred_smoothed
        S11 = results.S11
        S10 = results.S10
        S00 = results.S00
        logL = results.llf
        tdelta_filter = results.time_filter
        tdelta_smoother = results.time_smoother
        tdelta_expectation = results.time_expectation

        tdelta = np.array(
            [tdelta_filter, tdelta_smoother, tdelta_expectation], dtype=jnp.float32
        )

        return y_hat, x_T, P_T, S11, S10, S00, logL, tdelta

    def _M_step(
        self,
        y_t,
        y_hat,
        F,
        H,
        Xbeta,
        est_covList,
        block_p,
        block_q,
        x_T,
        P_T,
        S11,
        S10,
        S00,
        Phi,
    ):

        # convert all input to save memory
        p, T = y_t.shape
        q = block_q[-1]
        b = Xbeta.shape[1]
        nvar = len(block_p) - 1
        nlat = len(block_q) - 1
        # ndim = block_p[1:] - block_p[:-1]  # observed dimensions
        ldim = block_q[1:] - block_q[:-1]  # latent dimensions

        # Update f (Eq 3a)
        est_f = jnp.zeros((nlat))
        tStart = time.time()
        for q in range(nlat):
            s = slice(block_q[q], block_q[q + 1])

            num = jnp.trace(S10[s, s])
            den = jnp.trace(S00[s, s])

            # print(q, num, den)
            est_f = est_f.at[q].set(num / den)

        tdelta_f = time.time() - tStart

        # beta (the same as Eq.5 Calculli)
        err = y_t - y_hat  # compute the prediction error

        tStart = time.time()
        beta = _compute_beta_jax_kernel(b, y_t, x_T, H, Xbeta)
        jax.block_until_ready(beta)
        tdelta_beta = time.time() - tStart

        # s2 (see Eq, 14c)
        # compute the s2e
        # tStart = time.time()
        # s2e = compute_s2e(
        #     err, H, P_T, block_p, nvar, T, ndim)
        # tdelta_s2e = time.time() - tStart

        tStart = time.time()
        s2e = _compute_s2e_jax_kernel(err, H, P_T, tuple(block_p.tolist()))
        jax.block_until_ready(s2e)
        tdelta_s2e = time.time() - tStart

        # update A
        # est_A, tdelta_A = compute_A2(
        #     y_t, Xbeta, est_beta, x_T, P_T, block_p, block_q, nvar, nlat, T, ldim, Phi)

        tStart = time.time()

        est_A = _compute_A2_jax_kernel(
            y_t,
            Xbeta,
            beta,
            x_T,
            P_T,
            tuple(block_p.tolist()),
            tuple(block_q.tolist()),
            nvar,
            nlat,
            T,
            ldim,
            Phi,
        )

        jax.block_until_ready(est_A)
        tdelta_A = time.time() - tStart

        # Set the parameter of the minimise object

        tStart = time.time()
        Omega = S11 - S10 @ F.T - F @ S10.T + F @ S00 @ F.T
        par0 = jnp.log(
            jnp.array([fcov.rescale for fcov in est_covList], dtype=jnp.float32)
        )

        # 'L-BFGS-B': add options={'maxiter': 100, 'eps': 1e-8}, eps = gradiend step
        # opt = minimize(minf, par0, args=(est_covList, T, Omega), method='L-BFGS-B',
        #                tol=1e-3, jac=False, options={'maxiter': 100, 'eps': 1e-8})

        # 'Nelder-Mead': doesn't use gradients at all. It is much more robust for functions
        # that are "jumpy" or have extreme slopes.
        opt = minimize(
            self._minf,
            par0,
            args=(est_covList, T, Omega),
            method="Nelder-Mead",
            tol=1e-3,
            jac=False,
            options={"maxiter": 50},
        )

        tdelta_ks = time.time() - tStart

        # update the initial state mu and variance
        x0 = x_T[:, 0]
        Sigma0 = P_T[:, :, 0]

        tdelta = jnp.array(
            [tdelta_beta, tdelta_s2e, tdelta_f, tdelta_ks, tdelta_A], dtype=jnp.float32
        )

        est_ks = jnp.exp(opt.x)
        # print("est_ks", est_ks)

        # Note that the Cov. function is updated in place with the new rescale parameter (i.e. the range parameter of the Matern covariance function)
        update_params = self._createParams(beta, s2e, est_f, x0, Sigma0, est_ks, est_A)

        return update_params, opt.success, tdelta

    # %[Utils] Argmin problem, JAX  (M-step, rescale)
    def _minf(self, params, est_covList, T, Omega):
        ks = np.exp(params)  # Stability, add small eps to avoid zeros

        # Compute the precision and the logdetQ (sparse matrix)
        # invQ = [fcov.precision(rescale=ki) ]

        # Compute the maginal precision matrix
        invQ = []
        for fcov, ki in zip(est_covList, ks):
            invQi = fcov.precision(rescale=ki)

            # index of the inner points (i.e. the points of the latent domain)
            inx = fcov.fem_solver.inner
            Q_11 = invQi[inx, :][:, inx]
            Q_12 = invQi[inx, :][:, ~inx]
            Q_22 = invQi[~inx, :][:, ~inx]

            # Marginal precision matrix of the inner points (i.e. the points of the latent domain)
            Q_mar = Q_11 - Q_12 @ np.linalg.inv(Q_22.toarray()) @ Q_12.T

            invQ.append(Q_mar)

        # invQ = sp.block_diag(invQ) # dense matrix, not sparse
        invQ = scyp_block_diag(*invQ)

        # Compute the log determinant of the precision matrix (dense matrix, not sparse)
        logdet_invQ = np.linalg.slogdet(invQ)[1]

        # Compute the optimization function (negative log-likelihood)
        fun = -T * logdet_invQ + np.trace(invQ @ Omega)

        # Rescale the optimisation function to avoid numerical issues (e.g., overflow) during optimization
        return fun / 1e4

    def _getInitialValues(self, y_obs, Xbeta, block_p, block_q):

        # Compute the initial values of the parameters
        est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_A = (
            _compute_inital_values_jax_kernel(
                y_obs, Xbeta, tuple(block_p.tolist()), tuple(block_q.tolist())
            )
        )

        # Compute the initial values of the range parameters (i.e. the rescale parameter of the Matern covariance function) using the distance between the points of the latent domain (i.e. the inner points of the mesh)
        box = [cv.fem_solver.box for cv in self.cov_function]
        est_ks = [
            jnp.sqrt(8.0) / (jnp.minimum(jnp.abs(bx[2] - bx[0]), jnp.abs(bx[3] - bx[1])) / 3.0)
            for bx in box
        ]

        # Create the est_params object with the estimated initial values (if not provided, they will be set to None and the model will use default initial values)
        est_params = self._createParams(
            est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_ks, est_A
        )

        return est_params

    def _createParams(
        self, est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_ks, est_A
    ):
        est_params = ModelParams(
            beta=Param("beta", est_beta),
            s2e=Param("s2e", est_s2e),
            f=Param("f", est_f),
            A=Param("A", est_A),
            ks=Param("ks", est_ks),
            x0=Param("x0", est_x0),
            Sigma0=Param("Sigma0", est_Sigma0),
        )

        return est_params

    def _parseParams(self, params0: ModelParams | None):

        return (
            params0
            if params0 is not None
            else ModelParams(
                beta=Param("beta", None),
                s2e=Param("s2e", None),
                f=Param("f", None),
                A=Param("A", None),
                ks=Param("ks", None),
                x0=Param("x0", None),
                Sigma0=Param("Sigma0", None),
            )
        )

    def _updateParams(self, params: ModelParams, updates: ModelParams) -> ModelParams:
        """
        Update parameters using values from `updates`,
        respecting the `fixed` flags and handling None initial values.
        """

        updated_fields = {}

        for name in params.__dataclass_fields__:
            current_param = getattr(params, name)
            new_param = getattr(updates, name)

            # ---- CASE 1: parameter is fixed, never update
            if current_param.fixed:
                updated_fields[name] = current_param
                continue

            # ---- CASE 3: normal update (free parameter with initial value)
            updated_fields[name] = replace(
                current_param,
                value=jnp.asarray(new_param.value),
            )

        new_params = ModelParams(**updated_fields)

        # Special handling for ks update covariance scaling
        if not new_params.ks.fixed and new_params.ks.value is not None:
            for fcov, ksi in zip(self.cov_function, new_params.ks.value):
                fcov.rescale = ksi

        return new_params

    def _updateParams0(self, params0: ModelParams, updates: ModelParams) -> ModelParams:
        """
        Update parameters using values from `updates`,
        respecting the `fixed` flags and handling None initial values.
        """

        updated_fields = {}

        for name in params0.__dataclass_fields__:
            current_param0 = getattr(params0, name)
            new_param = getattr(updates, name)

            if current_param0.value is not None:
                updated_fields[name] = current_param0
                continue

            # ---- CASE 3: normal update (free parameter with initial value)
            updated_fields[name] = replace(
                current_param0,
                value=jnp.asarray(new_param.value),
            )

        new_params = ModelParams(**updated_fields)

        # Special handling for ks update covariance scaling
        if not new_params.ks.fixed and new_params.ks.value is not None:
            for fcov, ksi in zip(self.cov_function, new_params.ks.value):
                fcov.rescale = ksi

        return new_params

    # dense matrix
    def _buildBasis_list(self, points, hmesh):
        nvar = len(points)
        nlat = len(hmesh)

        basis = []  # list of the rows x columns matrices
        notfindInx = []
        for p in range(nvar):
            hrow = []
            notfindInxRow = []
            for q in range(nlat):
                # This function works iif the cov class is already defined
                # Compute basis between vertex and gird_obs point [m x q]
                countij, notfindInxij, hij = hmesh[q].fem_solver._compute_basis(
                    points[p]
                )

                hij = self._normalize_rows_sparse(hij[:, hmesh[q].fem_solver.inner])
                notfindInxRow.append(notfindInxij)

                # conver into coo format
                hij = jnp.asarray(hij.toarray(), dtype=jnp.float32)

                # Append the sub matrices
                hrow.append(hij)

            notfindInx.append(notfindInxij)  # append not find index
            basis.append(hrow)  # [n_i x qsize ]

        return basis

    def _buildRF_dense(self, s2error, flatent, pdim, qdim):
        """
        Builds dense diagonal matrices in JAX.

        This is the recommended approach for a direct, easy-to-use replacement.

        Args:
            s2error: 1D array of error variances.
            flatent: 1D array of latent factor values.
            pdim: Tuple of block dimensions for s2error (static for JIT).
            qdim: Tuple of block dimensions for flatent (static for JIT).

        Returns:
            A tuple of two dense JAX diagonal matrices.
        """
        # 1. Create the full diagonal vector by repeating elements
        rdiag_vec = jnp.repeat(s2error, repeats=jnp.array(pdim)).astype(
            dtype=jnp.float32
        )
        fdiag_vec = jnp.repeat(flatent, repeats=jnp.array(qdim)).astype(
            dtype=jnp.float32
        )

        # 2. Create the dense diagonal matrices from the vectors
        rdiag_matrix = jnp.diag(rdiag_vec)
        fdiag_matrix = jnp.diag(fdiag_vec)

        return rdiag_matrix, fdiag_matrix

    def _buildH_dense(self, A, basis):

        nvar, nlat = A.shape

        Phi = []  # list of the rows x columns matrices
        for p in range(nvar):
            hrow = []
            for q in range(nlat):
                hrow.append(A[p, q] * basis[p][q])
            Phi.append(hrow)

        return jnp.block(Phi)

    def _normalize_rows_sparse(self, sparse_mat):
        """
        Given a SciPy sparse matrix sparse_mat (CSR, CSC, etc.), return a new sparse matrix
        where each row sums to 1. Rows that originally sum to 0 will remain all zeros.
        """
        if not sp.isspmatrix(sparse_mat):
            raise ValueError("Input must be a SciPy sparse matrix.")

        # 1. Compute the sum of each row. The result is a (n_rows, 1) matrix, so we flatten to 1D.
        row_sums = np.array(sparse_mat.sum(axis=1)).ravel()  # shape = (n_rows,)

        # 2. Build an array of inverse sums, with zeros where row_sums == 0
        inv_sums = np.zeros_like(row_sums, dtype=np.float64)
        nonzero_mask = row_sums != 0
        inv_sums[nonzero_mask] = 1.0 / row_sums[nonzero_mask]

        # 3. Create a sparse diagonal matrix D so that D[i, i] = 1 / row_sums[i]
        #    Rows with row_sums == 0 get D[i,i] = 0, so that D @ row_i = 0 row.
        D = sp.diags(inv_sums)

        # 4. Multiply D @ sparse_mat; each row_i of the result is row_i of sparse_mat divided by row_sums[i].
        normalized = D.dot(sparse_mat)

        return normalized

    # fix H row functions
    def _setDomain(self, polygon):
        if polygon is None:
            polygon = [ConvexHull(pts) for pts in self.points]

        return polygon

    def _checkDomain(self, domain):

        flag = False
        msg = ""

        if domain is not None:
            if not isinstance(domain, (list, tuple)):
                raise TypeError("domain must be a list of Polygon objects")
            for i, poly in enumerate(domain):
                if not isinstance(poly, Polygon):
                    flag = True
                    msg = f"Each domain element must be a shapely Polygon, got {type(poly).__name__}"
                else:
                    bounds_str = ", ".join(f"{b:.2f}" for b in poly.bounds)
                    self._log("Domain-{}: area = {:2f}, box = ({})".format(i+1, poly.area, bounds_str))

        return flag, msg

    def _buildObservationGrid(self, df, formulas, predict = False, verbose=True, tmin=None, tmax=None):

        nvar = len(formulas)  # numer of the response variable

        # todo - check if the formulas are valid (e.g. if the response variable is in the dataframe, if the covariates are in the dataframe, etc.)
        dfs = [DesignMatricesBuilder(df, f, verbose=verbose, tmin=tmin, tmax=tmax).build(predict=predict) for f in formulas]

        T = [gr.T for gr in dfs]
        points = [gr.points for gr in dfs]

        # get dimnesion of each grid
        pdim = [grid.N for grid in dfs]
        block_p = np.hstack((0, np.cumsum(pdim)))
        ndim = block_p[-1]

        return nvar, points, dfs, ndim, pdim, block_p, T
    
    def _buildDesignMatrix(self, gridList):

        Ylist = [grid.y for grid in gridList if grid.y is not None]

        # X - Fixed effect design matrix -> 3D block diag - [N x beta x T]
        Xbeta_list = [grid.X for grid in gridList if grid.X is not None]

        # points_train = [pt[index, :] for pt, index in zip(points, itrain)]
        # points_test = [pt[index, :] for pt, index in zip(points, itest)]

        # Y_train_list = [yi[index, :] for yi, index in zip(Ylist, itrain)]
        # Xbeta_train_list = [xi[index, :, :] for xi, index in zip(Xlist, itrain)]

        # Y_test_list = [yi[index, :] for yi, index in zip(Ylist, itest)]
        # Xbeta_test_list = [xi[index, :, :] for xi, index in zip(Xlist, itest)]

        if len(Ylist) > 0:
            y_train = jnp.vstack(Ylist) 
        else:
            y_train = None

        if len(Xbeta_list) > 0:
            Xbeta_train = block_diag_3D(*Xbeta_list)
        else:          
            Xbeta_train = None

        # Y_test = np.vstack(Y_test_list)
        # Xbeta_test = block_diag_3D(Xbeta_test_list)

        return y_train, Xbeta_train

    def logger(self, stats, beta_decimals=2, scalar_decimals=2, relat_decimals=5):
        """
        Nicely formatted iteration logger for optimization/Kalman filter loops.
        """

        # --- Identify and format scalars vs arrays ---
        def format_value(x, decimals=2):
            x_np = np.asarray(x)

            # Scalar case
            if x_np.ndim == 0:
                return f"{float(x_np):.{decimals}f}"

            # Array case → keep full array print
            return np.array2string(
                x_np,
                precision=decimals,
                separator=", ",
                floatmode="fixed",
                suppress_small=False,
            )

        msg = f"""
------------------------------------------------------------------
Iteration : {stats['niter']}
logL      : {format_value(stats['logL'], scalar_decimals)}
delta L   : {format_value(stats['deltaL'], scalar_decimals)}
relat L   : {format_value(stats['relatL'], relat_decimals)}
beta      : {format_value(stats['beta'], beta_decimals)}
s2e       : {format_value(stats['s2e'], scalar_decimals)}
f param   : {format_value(stats['f'], scalar_decimals)}
rescale   : {format_value(stats['ks'], scalar_decimals)} (Status: {stats['opt_success']})
A (flat)  : {format_value(np.asarray(stats['A']).flatten(), scalar_decimals)}
x0        : {format_value(stats['x0'], scalar_decimals)}
S0 diag   : {format_value(stats['Sigma0'], scalar_decimals)}
Run time  : Tot: {format_value(stats['time_tot'], scalar_decimals)}, Estep: {format_value(stats['tdelta_E'], scalar_decimals)}, Mstep: {format_value(stats['tdelta_M'], scalar_decimals)}
------------------------------------------------------------------
"""
        return msg

    def _log_iteration(
        self,
        history: list,
        niter: int,
        params: ModelParams,
        opt_success: bool,
        logL_cur,
        delta_par,
        delta_lik,
        relat_lik,
        tdelta_iter,
        tdelta_Edet,
        tdelta_Mdet,
    ):
        """
        Create iteration dictionary and append to history list.
        """

        it = {
            "niter": niter,
            "beta": params.beta.value,
            "s2e": params.s2e.value,
            "f": params.f.value,
            "ks": params.ks.value,
            "opt_success": opt_success,
            "A": params.A.value.flatten(),
            "x0": params.x0.value.mean(),
            "Sigma0": jnp.diag(params.Sigma0.value).mean(),
            "logL": logL_cur,
            "deltaP": delta_par,
            "deltaL": delta_lik,
            "relatL": relat_lik,
            "time_tot": tdelta_iter,
            "tdelta_E": tdelta_Edet.sum(),
            "tdelta_E_detail": tdelta_Edet,
            "tdelta_M": tdelta_Mdet.sum(),
            "tdelta_M_detail": tdelta_Mdet,
        }

        history.append(it)

        return history


    def generate_summary(self, print_full=True):

        # top-left / top-right small tables
        p = self.shape[0] if hasattr(self, "shape") else "N/A"
        q = self.shape[1] if hasattr(self, "shape") else "N/A"
        T = self.shape[2] if hasattr(self, "shape") else "N/A"

        if p is None:
            p = "N/A"
        if q is None:
            q = "N/A"
        if T is None:
            T = "N/A"

        # Header information for the summary table
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
                (
                    "Dep. Variables:",
                    lambda: [self.y_name if hasattr(self, "y_name") else "N/A"],
                ),
                ("Date:", lambda: [self._today]),
                ("JAX backend:", lambda: [f"{jax.default_backend()}"]),
                ("JAX devices:", lambda: [f"{jax.devices()}"]),
            ]
        )

        top_right = dict(
            [
                ("Shape:", lambda: [f"(p = {p}, q = {q}, T = {T})"]),
                (
                    "Diag. R",
                    lambda: (
                        f"{jnp.mean(jnp.diag(self.R)):2f}"
                        if self.R is not None
                        else ["N/A"]
                    ),
                ),
                (
                    "Diag. Q",
                    lambda: (
                        f"{jnp.mean(jnp.diag(self.Q)):2f}"
                        if self.Q is not None
                        else ["N/A"]
                    ),
                ),
                (
                    "Diag. F",
                    lambda: (
                        f"{jnp.mean(jnp.diag(self.F)):2f}"
                        if self.F is not None
                        else ["N/A"]
                    ),
                ),
                (
                    "mean x0",
                    lambda: (
                        f"{jnp.mean(self.x0):2f}" if self.x0 is not None else ["N/A"]
                    ),
                ),
                (
                    "mean Sigma0",
                    lambda: (
                        f"{jnp.mean(jnp.diag(self.Sigma0)):2f}"
                        if self.Sigma0 is not None
                        else ["N/A"]
                    ),
                ),
                (
                    "Rank",
                    lambda: (
                        [f"{q/p :4f}"] if q != "N/A" and p != "N/A" and p > 0 else ["N/A"]
                        if q != "N/A" and p != "N/A"
                        else ["N/A"]
                    ),
                ),
            ]
        )

        # Generate the dictionaly
        gen_top_left = []
        for item in top_left.keys():
            gen_top_left.append((item, list(top_left[item]())))

        gen_top_right = []
        for item in top_right.keys():
            gen_top_right.append((item, top_right[item]()))

        len_empty = len(gen_top_left)- len(gen_top_right) 
        if len_empty > 0:
            gen_top_right = gen_top_right + [("", [""])] * len_empty
        elif len_empty < 0:
            gen_top_left = gen_top_left + [("", [""])] * (-len_empty)

        if not print_full:
            return gen_top_left, gen_top_right

        else:  
            # Get the generate table from the gridlist
            if hasattr(self,"gridList") and self.gridList is not None:

                gen_top_left_grid = []
                gen_top_right_grid = []
                for i, grid in enumerate(self.gridList):

                    left, righ = grid.generate_summary()
                    
                    # check the length of the left and right tables and add empty rows if they are different
                    len_empty = len(left) - len(righ)
                    if len_empty > 0:
                        righ = righ + [("", [""])] * len_empty
                    elif len_empty < 0:
                        left = left + [("", [""])] * (-len_empty)
                    
                    left = [(f"Grid {i}", ["-" * 28])] + left
                    righ = [(f"Grid {i}", ["-" * 28])] + righ
                    

                    
                    gen_top_left_grid = gen_top_left_grid + left
                    gen_top_right_grid = gen_top_right_grid + righ

                gen_top_left = gen_top_left + gen_top_left_grid
                gen_top_right = gen_top_right + gen_top_right_grid

            
            # Get the generate table from the covariance
            if hasattr(self,"_cov_matern") and self._cov_matern is not None:

                gen_top_left_cov = []
                gen_top_right_cov = []
                for i, cov in enumerate(self._cov_matern):
                    left, righ = cov.generate_summary()

                    # check the length of the left and right tables and add empty rows if they are different
                    len_empty = len(left) - len(righ)
                    if len_empty > 0:
                        righ = righ + [("", [""])] * len_empty
                    elif len_empty < 0:
                        left = left + [("", [""])] * (-len_empty)

                    
                    left = [(f"Latent. {i}", ["-" * 28])] + left
                    righ = [(f"Latent {i}", ["-" * 28])] + righ

                    
                    gen_top_left_cov = gen_top_left_cov + left
                    gen_top_right_cov = gen_top_right_cov + righ

                gen_top_left = gen_top_left + gen_top_left_cov
                gen_top_right = gen_top_right + gen_top_right_cov
            
            return gen_top_left, gen_top_right 

    def summary(self, print_full=True) -> Summary:
        """Return or print a structured summary of the model."""
        self.model = SimpleNamespace()
        # self.params = np.zeros(1)  # Placeholder for model parameters if needed in the future

        # Generate the summary tables
        gen_top_left, gen_top_right = self.generate_summary(print_full=print_full)
        
        # Add the header to the summary
        smry = Summary()
        smry.add_table_2cols(
            self,
            title="State Space Model",
            gleft=gen_top_left,
            gright=gen_top_right,
            yname= self.yname if self.yname is not None else "None",
            xname= self.xbeta_names if self.xbeta_names is not None else "None",
        )

        return smry

    def format_info_table(self, items, indent=0):
        """
        Format a list of (key, value) tuples into a clean aligned string.

        Parameters
        ----------
        items : list of tuples
            [(key, value), ...]
        indent : int
            Number of spaces to indent each row.

        Returns
        -------
        str
            Nicely formatted multi-line string.
        """
        # Compute longest key for alignment
        max_key_len = max(len(str(k)) for k, _ in items)

        lines = []
        pad = " " * indent

        for key, value in items:
            key = str(key).rstrip(":") + ":"

            # Convert lists to readable string
            if isinstance(value, list):
                value = ", ".join(map(str, value))

            lines.append(f"{pad}{key:<{max_key_len+1}} {value}")

        return "\n".join(lines)
 
    def _is_verbose(self, verbose=None) -> bool:
        return self.verbose if verbose is None else verbose

    def _log(self, msg: str, verbose=None) -> None:
        if self._is_verbose(verbose):
            self.print_info(msg)

    def print_info(self, msg):

        dt = datetime.fromtimestamp(time.time(), tz=timezone.utc)
        print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")
