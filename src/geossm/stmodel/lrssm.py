"""
Adapter scaffolding making the project's StateSpaceModel usable with
statsmodels' MLEModel API.
"""

import numpy as np
from scipy.spatial import ConvexHull

try:
    from statsmodels.tsa.statespace.mlemodel import MLEModel
except Exception:
    # Minimal fallback base class to allow importing this module when
    # statsmodels is not installed. This fallback does not implement
    # any optimization or fit behavior.
    class MLEModel(object):
        def __init__(self, endog=None, exog=None, **kwargs):
            self.endog = endog
            self.exog = exog

# inmport the state space model
from geossm.ssm import StateSpaceModel
from geossm.covmodel import spdeAppoxCov

import jax.numpy as jnp
from jax import jit, lax, vmap
import scipy.sparse as sp
from functools import partial
import time
from jax.scipy.linalg import block_diag
from scipy.linalg import block_diag as scyp_block_diag
import jax
from scipy.optimize import minimize

from geossm import data_preparation, DesignMatrices
from geossm import block_diag_3D, getHardware

from shapely.geometry import LineString, Point, Polygon, MultiPoint


# %% Utilities for the EM algorithm (jax kernel functions)
# %% [Utils] Updating formula, JAX (M-Step)


@partial(jit, static_argnames=['b'])
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


@partial(jit, static_argnames=['block_p'])
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


@partial(jit, static_argnames=['block_p', 'block_q', 'nvar', 'nlat', 'T'])
def _compute_A2_jax_kernel(y_t, Xbeta, beta, x_T, P_T, block_p, block_q, nvar, nlat, T, ldim, Phi):
    """
    Rewritten version of compute_A2 to work with jax.numpy and jax.jit,
    considering 'block_p', 'block_q', 'nvar', 'nlat', 'T', 'max_mdim_i' as static arguments.
    Fixes NonConcreteBooleanIndexError by padding to max_mdim_i and
    using jax.scipy.linalg.block_diag for static shapes.
    """
    # Precompute fixed effects and residuals for all data points
    # Shape (total_mdim, T)
    residual = y_t - jnp.einsum('mpn,p->mn', Xbeta, beta)

    # Initialize W (W will hold the results)
    W = jnp.zeros((nvar, nlat), dtype=y_t.dtype)

    # Loop over nvar (i) - nvar is a static argument
    for i in range(nvar):
        R_it = jnp.zeros((nlat, nlat), dtype=y_t.dtype)  # nlat is static
        g_it = jnp.zeros((1, nlat), dtype=y_t.dtype)     # nlat is static

        # Get the actual number of observations for the current block 'i'
        # mdim_i is static for a given `i` but can vary between `i` blocks.
        # max_mdim_i is the global maximum used for padding.
        mdim_i = block_p[i+1] - block_p[i]
        max_mdim_i = mdim_i

        # Loop over time steps T (t) - T is a static argument
        for t in range(T):
            # Extract the slice of y_t for the current block and time, then get NaN mask
            y_t_slice_full = y_t[block_p[i]:block_p[i+1], t]  # Shape (mdim_i,)
            # Boolean mask, shape (mdim_i,)
            nna_mask = ~jnp.isnan(y_t_slice_full)

            # Get the full residual for the current block and time
            residual_full_slice = residual[block_p[i]:block_p[i+1], t]  # Shape (mdim_i,)

            # Mask and pad residual_full_slice
            # Multiply by mask to zero out invalid entries, then pad to max_mdim_i
            # This ensures ep_valid_padded always has shape (max_mdim_i,)
            ep_valid_padded = jnp.pad(residual_full_slice * nna_mask,  # (mdim_i,) * (mdim_i,)
                                      # Pad only the first dimension
                                      (0, max_mdim_i - mdim_i),
                                      constant_values=0.0)  # (max_mdim_i,)

            # --- Compute Psi_it (fixed shape) ---
            temp_padded_blocks = []
            # Assuming q_j_dim is constant, e.g., 1, as per common use with block_q
            # q_j_dim = block_q[j+1] - block_q[j]
            # This needs to be consistent, so let's use the first block_q diff for column padding.
            # Or ensure block_q always yields same q_j_dim for consistent Phi slicing.
            # Assuming block_q[j+1]-block_q[j] is always the same for all j (e.g. 1)
            # as in your example: block_q = jnp.array([0, 1, 2, nlat]) implies q_j_dim=1.
            # Get the dimension of a single q block
            q_j_dim = block_q[1] - block_q[0]

            for j in range(nlat):  # nlat is static
                # Current block of Phi, shape (mdim_i, q_j_dim)
                phi_block_current = Phi[block_p[i]                                        :block_p[i+1], block_q[j]:block_q[j+1]]

                # Apply mask by multiplying (zeros out rows corresponding to NaNs)
                # nna_mask[:, jnp.newaxis] broadcasts the mask to all columns of phi_block_current
                phi_block_masked = phi_block_current * nna_mask[:, jnp.newaxis]

                # Pad this masked block to (max_mdim_i, q_j_dim) for static shapes
                padded_phi_block = jnp.pad(phi_block_masked,
                                           ((0, max_mdim_i - mdim_i), (0, 0)),
                                           constant_values=0.0)
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
                Psi_times_xT = Psi_it @ x_T[:, [t+1]]

                # Product shape: (max_mdim_i, nlat * max_mdim_i) @ (nlat * max_mdim_i, 1) = (max_mdim_i, 1)
                kron_terms.append(kron_term_val_j @ Psi_times_xT)

            # Concatenate these (max_mdim_i, 1) vectors horizontally to form (max_mdim_i, nlat) matrix
            combined_H_times_x = jnp.concatenate(
                kron_terms, axis=1)  # Shape: (max_mdim_i, nlat)

            # Add to g_it: ep_valid_padded.T is (1, max_mdim_i)
            # Result: (1, max_mdim_i) @ (max_mdim_i, nlat) = (1, nlat) -> Matches g_it's shape
            g_it += ep_valid_padded[jnp.newaxis, :] @ combined_H_times_x

            # --- Update R_it ---
            # Common term for R_it update, shape: (ldim, ldim)
            x_T_x_T_P = x_T[:, [t+1]] @ x_T[:, [t+1]].T + P_T[:, :, t+1]

            # Nested loops for j and k
            for j_idx in range(nlat):
                for k_idx in range(nlat):
                    # Use max_mdim_i directly for jnp.eye for static shape
                    tt_jk = jnp.eye(max_mdim_i, dtype=y_t.dtype)
                    # Kron term: shape (nlat * max_mdim_i, nlat * max_mdim_i)
                    kron_op_jk = jnp.kron(
                        _ei_jax(j_idx, nlat).T @ _ei_jax(k_idx, nlat), tt_jk)

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
        w_i_row = jnp.linalg.solve(R_it, g_it.T).reshape(nlat,)

        # Update W (immutable operation using .at[].set())
        W = W.at[i, :].set(w_i_row)

    return W

# Helper function ei for JAX
def _ei_jax(i, dim):
    """
    Creates a one-hot row vector for JAX.
    """
    return jax.nn.one_hot(i, dim, dtype=jnp.float32).reshape(1, dim)

# %% Low Rank State-Space Model adapter to statsmodels MLEModel API


class LRStateSpaceModel:

    def __init__(self, df, formulas, domain=None, verbose=True):

        self.df = df
        self.formulas = formulas

        # Compute the design matrices
        if verbose:
            self.print_info("Building observation grid...")
        
        self.nvar, self.points, self.gridList, self.ndim, self.pdim, self.block_p, self.T = self._buildObservationGrid(
            df, formulas)
        
        if verbose:
            self.print_info("Building design matrix...")    
        self.y_train, self.Xbeta_train = self._buildDesignMatrix()

        # Check the domain
        if verbose:
            self.print_info("Checking the domain...")

        flag, msg = self._checkDomain(domain)
        if flag:
            raise ValueError(msg)
        else:
            self.domain = self._setDomain(domain)

    def setup(self, mesh_obj: list, domain: list = None):
        
        # this domain is the domain on which the mesh is defined, and where the 
        # covariance function has a meaning
        if domain is not None:
            flag, msg = self._checkDomain(domain)
            if flag:
                raise ValueError(msg)
        else:
            domain  = self._domain
        
        if len(mesh_obj) != len(domain):
            raise ValueError(f"Number of mesh objects ({len(mesh_obj)}) must match number of domains ({len(domain)})")  
    
        # mehs_obj = list of the latent domain
        self._cov_matern = []
        for meshi, domi in zip(mesh_obj, domain):

            # create the covariance model of the matern
            temp = spdeAppoxCov(
                [domi], latlon=False, nu=1, var=1, rescale=1)
            self._cov_matern.append(temp.setup(meshi))

        self.nlat = len(self._cov_matern)

        return self

    def fit(self, beta0=None, s2e0=None,
            f0=None, A0=None, ks0=None, x0=None, Sigma0=None,
            max_iter=100, tol_relat=1e-3, nstat=[], verbose=True, dtype=jnp.float32,
            fix_b=False, fix_s2=False, fix_f=False, fix_A=False, fix_ks=False, fix_x0=False, fix_Sigma0=False):

        # make sure all think are jax.numpy
        # y_t = jnp.asarray(y_t, dtype=jnp.float32)
        # Xbeta = jnp.asarray(Xbeta, dtype=jnp.float32)
        # pdim = jnp.asarray(pdim)

        # Get global constants
        nvar = self.nvar  # len(self.pdim)
        nlat = self.nlat
        cov_function = self.cov_function
        pdim = jnp.asarray(self.pdim, dtype=jnp.int32)
        block_p = jnp.asarray(self.block_p, dtype=jnp.int32)

        # Get the observed data
        y_obs = jnp.asarray(self.yTrain, dtype=dtype)
        Xbeta = jnp.asarray(self.Xbeta_train, dtype=dtype)
        points = self.points
        p, T = y_obs.shape

        # Get latent dimension (i.e. the rank)
        qdim = jnp.array(
            [cov.n_inner_points for cov in cov_function], dtype=jnp.int32)
        block_q = jnp.hstack((0, jnp.cumsum(qdim)))

        q = jnp.sum(qdim)

        # TODO: add verbose option here to print the initial values of the parameters and the log-likelihood

        # set the initial values
        est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_A = self._getInitialValues(
            y_obs, Xbeta, tuple(block_p.tolist()), tuple(block_q.tolist()), T)

        est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_ks, est_A = self.updateParams(
            beta0, s2e0, f0, A0, ks0, x0, Sigma0, fix_b, fix_s2, fix_f, fix_A, fix_ks, fix_x0, fix_Sigma0)

        # ---- print messages
        if verbose:
            msg = f"beta:{jnp.round(est_beta,2)} - s2e:{jnp.round(est_s2e,2)} - f:{jnp.round(est_f,2)} - rescale:{jnp.round(est_ks,2)} - A: {jnp.round(est_A.flatten(),2)}"
            print(msg)

        est_vet_par = jnp.hstack(
            (est_beta.flatten(), est_s2e.flatten(), est_f.flatten(), est_ks.flatten(), est_A.flatten()))

        # Flag of the EM convergence
        flag = True
        niter = 0
        logL_prev = -jnp.inf
        logL_cur = 0
        delta_par = jnp.nan
        delta_lik = jnp.nan
        relat_lik = jnp.nan
        tdelta_iter = 0
        tdelta_Edet = np.zeros(3)
        tdelta_Mdet = np.zeros(3)

        it = {'niter': niter, 'beta': est_beta, 's2e': est_s2e, 'f': est_f,
              'ks': est_ks, 'est_A': est_A.flatten(), 'x0': est_x0.mean(),
              'S0': jnp.diag(est_Sigma0).mean(), 'logL': logL_cur,
              'deltaP': delta_par, 'deltaL': delta_lik, 'relatL': relat_lik,
              'time_tot': tdelta_iter, 'tdelta_E': tdelta_Edet.sum(),
              'tdelta_E_detail': tdelta_Edet, 'tdelta_M': tdelta_Mdet.sum(), 'tdelta_M_detail': tdelta_Mdet}

        nstat.append(it)

        # Compute the basis matrix (just one) - no boundary
        basis = self._buildBasis_list(points, cov_function)
        Phi = self._buildH_dense(
            jnp.ones((nvar, nlat), dtype=jnp.float32), basis)

        # Start EM iteration
        while flag:
            niter += 1

            # Start the timer for the iteration
            tStart_iter = time.time()

            # ---- build parametrised matrices
            H = self.buildH_dense(est_A, basis)  # dense

            # R, F = buildRF(est_s2e, est_f, pdim, qdim)
            R, F = self.buildRF_dense(est_s2e, est_f, pdim, qdim)

            # Compute the maginal precision matrix
            invQ = []
            for fcov in cov_function:
                invQi = fcov.precision()

                # index of the inner points (i.e. the points of the latent domain)
                inx = fcov.inner
                Q_11 = invQi[inx, :][:, inx]
                Q_12 = invQi[inx, :][:, ~inx]
                Q_22 = invQi[~inx, :][:, ~inx]

                # Marginal precision matrix of the inner points (i.e. the points of the latent domain)
                Q_mar = Q_11 - Q_12 @ np.linalg.inv(Q_22.toarray()) @ Q_12.T

                invQ.append(Q_mar)

            # Compute the block diagonal covariance matrix Q of the latent factors (i.e. the points of the latent domain)
            Q = block_diag(
                *[jnp.linalg.solve(mt, jnp.eye(mt.shape[0], dtype=jnp.float32)) for mt in invQ])

            # ---- E step
            y_hat, x_t, x_T, P_T, P_T_1, S11, S10, S00, logL_cur, tdelta_Edet = self._E_step(
                y_obs, R, F, H, Q, est_x0, est_Sigma0, Xbeta, est_beta)

            # ---- M step, get the updated parameters
            update_beta, update_s2e, update_f, update_x0, update_Sigma0, cov_function, update_A, tdelta_Mdet, opt_success = self._M_step(
                y_obs, y_hat, F, H, Xbeta, points, cov_function, block_p, block_q, x_T, P_T, S11, S10, S00, Phi, est_beta)

            # Update the paramiters after the M step
            update_ks = jnp.array([cov.rescale for cov in cov_function])
            update_beta, update_s2e, update_f, update_x0, update_Sigma0, update_ks, update_A = self._updateParams(beta=update_beta, s2e=update_s2e, f=update_f, A=update_A, ks=update_ks, x0=update_x0, Sigma0=update_Sigma0,
                                                                                                                  fix_b=fix_b, fix_s2=fix_s2, fix_f=fix_f, fix_A=fix_A, fix_ks=fix_ks, fix_x0=fix_x0, fix_Sigma0=fix_Sigma0)

            # Stack the vector parameter
            # update_vet_par = jnp.hstack(
            #     (update_beta.flatten(), update_s2e.flatten(), update_f.flatten(), update_ks.flatten(), update_A.flatten()))

            # Compute the delta log likelihood ( 0 < current - previous < tol_lik )
            delta_lik = logL_cur - logL_prev
            relat_lik = jnp.abs(delta_lik / logL_prev)

            # End the timer for the iteration
            tdelta_iter = time.time() - tStart_iter

            # Print iteration messages
            if verbose:
                msg = self.logger(niter, logL_cur, delta_lik, relat_lik, update_beta,
                                  update_s2e, update_f, update_ks, est_A, est_x0, est_Sigma0, opt_success,  tdelta_iter, tdelta_Edet.sum(), tdelta_Mdet.sum())

                print(msg)

            # Update the paramiters to be used in the next EM iteration
            est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_ks, est_A = update_beta, update_s2e, update_f, update_x0, update_Sigma0, update_ks, update_A
            logL_prev = logL_cur

            # append the results
            it = {'niter': niter, 'beta': est_beta, 's2e': est_s2e, 'f': est_f,
                  'ks': est_ks, 'est_A': est_A.flatten(), 'x0': est_x0.mean(),
                  'S0': jnp.diag(est_Sigma0).mean(), 'logL': logL_cur,
                  'deltaP': delta_par, 'deltaL': delta_lik, 'relatL': relat_lik,
                  'time_tot': tdelta_iter, 'tdelta_E': tdelta_Edet.sum(),
                  'tdelta_E_detail': tdelta_Edet, 'tdelta_M': tdelta_Mdet.sum(), 'tdelta_M_detail': tdelta_Mdet}

            nstat.append(it)

            # Check the EM convergence (if the log-likelihood is not improving more than tol_lik or the max number of iterations is reached)
            if niter == max_iter or relat_lik <= tol_relat:
                flag = False

        return est_beta, est_s2e, est_f, est_x0, est_Sigma0, cov_function, est_A, nstat, y_hat, x_T, P_T, P_T_1, S11, S10, S00

    @property
    def cov_function(self):
        return self._cov_matern

    def _E_step(self, y_t, R, F, H, Q, est_x0, est_Sigma0, Xbeta, est_beta):

        # E step: compute the expected values of the latent factors and the log-likelihood
        # 1) Create the SSM object with the current parameters
        # 2) Run the Kalman filter and smoother to get the expected values of the latent factors and the log-likelihood

        # Create the SSM object with the current parameters
        ssmodel = StateSpaceModel(
            F=F, H=H, Q=Q, R=R, Xbeta=Xbeta, beta=est_beta, x0=est_x0, Sigma0=est_Sigma0)

        # Run the Kalman filter and smoother to get the expected values of the latent factors and the log-likelihood
        y_hat, x_t, x_T, P_T, P_T_1, S11, S10, S00, logL, tdelta_filter, tdelta_smoother, tdelta_expectation = ssmodel.estiamte(
            y_t)

        tdelta = np.array([tdelta_filter, tdelta_smoother,
                          tdelta_expectation], dtype=jnp.float32)

        return y_hat, x_t, x_T, P_T, P_T_1, S11, S10, S00, logL, tdelta

    def _M_step(self, y_t, y_hat, F, H, Xbeta, points, est_covList, block_p, block_q, x_T,
                P_T, S11, S10, S00, Phi, est_beta):

        # convert all input to save memory
        p, T = y_t.shape
        q = block_q[-1]
        b = Xbeta.shape[1]
        nvar = len(block_p)-1
        nlat = len(block_q)-1
        ndim = block_p[1:] - block_p[:-1]  # observed dimensions
        ldim = block_q[1:] - block_q[:-1]  # latent dimensions

        # Update f (Eq 3a)
        est_f = jnp.zeros((nlat))
        tStart = time.time()
        for q in range(nlat):
            s = slice(block_q[q], block_q[q+1])

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

        est_A = _compute_A2_jax_kernel(y_t, Xbeta, beta, x_T, P_T,
                                       tuple(block_p.tolist()), tuple(block_q.tolist()),  nvar, nlat, T, ldim, Phi)

        jax.block_until_ready(est_A)
        tdelta_A = time.time() - tStart

        # Set the parameter of the minimise object

        tStart = time.time()
        Omega = S11 - S10 @ F.T - F @ S10.T + F @ S00 @ F.T
        par0 = jnp.log(
            jnp.array([fcov.rescale for fcov in est_covList], dtype=jnp.float32))

        # 'L-BFGS-B': add options={'maxiter': 100, 'eps': 1e-8}, eps = gradiend step
        # opt = minimize(minf, par0, args=(est_covList, T, Omega), method='L-BFGS-B',
        #                tol=1e-3, jac=False, options={'maxiter': 100, 'eps': 1e-8})

        # 'Nelder-Mead': doesn't use gradients at all. It is much more robust for functions
        # that are "jumpy" or have extreme slopes.
        opt = minimize(self._minf, par0, args=(est_covList, T, Omega), method='Nelder-Mead',
                       tol=1e-3, jac=False, options={'maxiter': 50})

        tdelta_ks = time.time() - tStart

        # update the initial state mu and variance
        x0 = x_T[:, 0]
        Sigma0 = P_T[:, :, 0]

        tdelta = jnp.array([tdelta_beta, tdelta_s2e, tdelta_f,
                            tdelta_ks, tdelta_A], dtype=jnp.float32)
        return beta, s2e, est_f, x0, Sigma0, est_covList, est_A, tdelta, opt.success

    # %[Utils] Argmin problem, JAX  (M-step, rescale)

    def _minf(self, params, est_covList, T, Omega):
        ks = np.exp(params)  # Stability, add small eps to avoid zeros

        # Compute the precision and the logdetQ (sparse matrix)
        # invQ = [fcov.precision(rescale=ki) ]

        # Compute the maginal precision matrix
        invQ = []
        for fcov in est_covList:
            invQi = fcov.precision()

            # index of the inner points (i.e. the points of the latent domain)
            inx = fcov.inner
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

    def logdetSparse(Q):
        # Perform LU decomposition of the sparse matrix
        lu = sp.linalg.splu(Q)

        # Extract diagonal elements from L and U
        diagL = lu.L.diagonal().astype(np.complex128)
        diagU = lu.U.diagonal().astype(np.complex128)

        # Compute the log-determinant
        logdet = np.log(np.abs(diagL)).sum() + np.log(np.abs(diagU)).sum()

        return logdet

    def _updateParams(self, beta=None, s2e=None, f=None, A=None, ks=None, x0=None, Sigma0=None, fix_b=False, fix_s2=False, fix_f=False, fix_A=False, fix_ks=False, fix_x0=False, fix_Sigma0=False):

        if beta is not None and fix_b == False:
            est_beta = jnp.asarray(beta)

        if s2e is not None and fix_s2 == False:
            est_s2e = jnp.asarray(s2e)

        if f is not None and fix_f == False:
            est_f = jnp.asarray(f)
        if A is not None and fix_A == False:
            est_A = jnp.asarray(A)

        if ks is not None and fix_ks == False:
            est_ks = jnp.asarray(ks)
            for fcov, ksi in zip(self.cov_function, est_ks):
                fcov.rescale = ksi

        if x0 is not None and fix_x0 == False:
            est_x0 = jnp.asarray(x0)
        if Sigma0 is not None and fix_Sigma0 == False:
            est_Sigma0 = jnp.asarray(Sigma0)

        # return the updated parameters
        return est_beta, est_s2e, est_f, est_x0, est_Sigma0, est_ks, est_A

    @partial(jax.jit, static_argnums=(2, 3, 4))
    def _getInitialValues(self, y_t, Xbeta, block_p, block_q, T):
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
        Xs = jnp.einsum('bij,bik->jk', Xr, Xr)
        ys = jnp.einsum('bij,bi->j', Xr, Yt_clean)

        # Solve for b using jnp.linalg.solve for numerical stability
        est_beta = jnp.linalg.solve(Xs, ys)

        # --- Mean variance of the residuals ---
        # Vectorize the residual calculation instead of using a Python loop.
        # 'ibt,b->it' means: multiply (ni, b, T) with (b,) -> result (ni, T)
        predicted_y = jnp.einsum('ibt,b->it', Xbeta, est_beta)
        res = Yt_clean - predicted_y

        # --- s2 measurement error [1 * p] ---
        # A Python loop over static values (from block_p) is acceptable and will
        # be "unrolled" by the JIT compiler.
        var_res_list = []
        for p in range(nvar):
            # Slicing and computing variance
            var_res_list.append(jnp.var(res[block_p[p]:block_p[p+1]]))

        var_res = jnp.stack(var_res_list)
        est_s2 = var_res * 0.2

        # --- est_A (Loading Matrix) ---
        # Create the matrix without loops using JAX's functional update syntax.
        diag_vals = jnp.sqrt(var_res * 0.8)
        diag_size = min(nvar, nlat)
        # Create a zero matrix and set the diagonal elements
        est_A = jnp.zeros((nvar, nlat))
        est_A = est_A.at[jnp.arange(diag_size), jnp.arange(
            diag_size)].set(diag_vals[:diag_size])

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
                countij, notfindInxij, hij = hmesh[q]._compute_basis(
                    points[p])

                hij = self.normalize_rows_sparse(hij[:, hmesh[q].inner])
                notfindInxRow.append(notfindInxij)

                # conver into coo format
                hij = jnp.asarray(hij.toarray(), dtype=jnp.float32)

                # Append the sub matrices
                hrow.append(hij)

            notfindInx.append(notfindInxij)  # append not find index
            basis.append(hrow)  # [n_i x qsize ]

        return basis

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
        nonzero_mask = (row_sums != 0)
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
            for poly in domain:
                if not isinstance(poly, Polygon):
                    flag = True
                    msg = f"Each domain element must be a shapely Polygon, got {type(poly).__name__}"

        return flag, msg

    def _buildObservationGrid(self, df, formulas):

        nvar = len(formulas)  # numer of the response variable
        gridList = [data_preparation(df, f) for f in formulas]

        T = [gr.T for gr in gridList]
        points = [gr.points for gr in gridList]

        # get dimnesion of each grid
        pdim = [grid.N for grid in gridList]
        block_p = np.hstack((0, np.cumsum(pdim)))

        return nvar, points, gridList, pdim, block_p[-1], block_p, T

    def _buildDesignMatrix(self):

        Ylist_original = [grid.y for grid in self.gridList]

        # applay the log transofrmation (natural log) [positive prediction]
        Ylist = Ylist_original
        # Ylist = [] # Ylist_original
        # for yi in Ylist_original:
        #     yi[yi <= 0.5] = np.nan
        #     Ylist.append(np.log(yi))

        # X - Fixed effect design matrix -> 3D block diag - [N x beta x T]
        Xbeta_list = [grid.X for grid in self.gridList]

        # points_train = [pt[index, :] for pt, index in zip(points, itrain)]
        # points_test = [pt[index, :] for pt, index in zip(points, itest)]

        # Y_train_list = [yi[index, :] for yi, index in zip(Ylist, itrain)]
        # Xbeta_train_list = [xi[index, :, :] for xi, index in zip(Xlist, itrain)]

        # Y_test_list = [yi[index, :] for yi, index in zip(Ylist, itest)]
        # Xbeta_test_list = [xi[index, :, :] for xi, index in zip(Xlist, itest)]

        y_train = jnp.vstack(Ylist)
        Xbeta_train = block_diag_3D(*Xbeta_list)

        # Y_test = np.vstack(Y_test_list)
        # Xbeta_test = block_diag_3D(Xbeta_test_list)

        return y_train, Xbeta_train

    def logger(self,
               niter,
               logL_cur,
               delta_lik,
               relat_lik,
               update_beta,
               update_s2e,
               update_f,
               update_ks,
               est_A,
               est_x0,
               est_Sigma0,
               opt_success,
               time_iter,
               time_Estep,
               time_Mstep,
               beta_decimals=2,
               scalar_decimals=2,
               relat_decimals=5):
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
                separator=', ',
                floatmode='fixed',
                suppress_small=False
            )

        msg = f"""
------------------------------------------------------------------
Iteration : {niter}
logL      : {format_value(logL_cur, scalar_decimals)}
delta L   : {format_value(delta_lik, scalar_decimals)}
relat L   : {format_value(relat_lik, relat_decimals)}
beta      : {format_value(update_beta, beta_decimals)}
s2e       : {format_value(update_s2e, scalar_decimals)}
f param   : {format_value(update_f, scalar_decimals)}
rescale   : {format_value(update_ks, scalar_decimals)} (Status: {opt_success})
A (flat)  : {format_value(np.asarray(est_A).flatten(), scalar_decimals)}
x0        : {format_value(est_x0.mean(), scalar_decimals)}
S0 diag   : {format_value(jnp.diag(est_Sigma0).mean(), scalar_decimals)}
Run time  : Tot: {format_value(time_iter, scalar_decimals)}, Estep: {format_value(time_Estep, scalar_decimals)}, Mstep: {format_value(time_Mstep, scalar_decimals)}
------------------------------------------------------------------
"""
        return msg
    
    def __str__(self):

        flag = False
        if hasattr(self, '_cov_matern'):
            qdim = [len(cov.n_inner_points) for cov in self._cov_matern]
            covfs = [str(cov) for cov in self._cov_matern]
            nlat = self.nlat
            flag = True
        else:
            qdim = 'None'
            covfs = 'None'
            nlat = 'None'
        
        lines = []

        lines.append("LRStateSpaceModel")
        lines.append("-" * 60)

        lines.append(f"Observed variables : {self.nvar} - {self.ndim}")
        lines.append(f"Latent factors     : {nlat} - {qdim}")
        lines.append(f"Domain             : {[d.geom_type for d in self.domain]}")

        lines.append("-" * 60)
        formulas = [gr.formula for gr in self.gridList]
        lines.append("Observation eqs    :")
        lines.extend(f"  - {f}" for f in formulas)


        lines.append("-" * 60)
        if flag:
            lines.append(f"Covariance         :")
            lines.extend(f"  - {cf}" for cf in covfs)
        else:
            lines.append("Covariance         : None (not set up)")

        lines.append("-" * 60)
        lines.append("Model structure:")
        lines.append("-" * 60)
        lines.append(self.model_structure(qdim))

        lines.append("-" * 60)
        lines.append(f"JAX backend        : {jax.default_backend()}")
        lines.append(f"JAX devices        : {jax.devices()}")

        return "\n".join(lines)


    def model_structure(self,  qdim):
        """Return a formatted summary of the state-space structure."""

        return (
            "Observation equation:\n"
            + f"  y(t) = X{getattr(self.Xbeta_train, 'shape', 'None')} beta + H({self.pdim},{qdim}) x(t)"
            + f"+ e(t),  e(t) ~ N(0, R({self.pdim}, {self.pdim}))\n\n"
            + "State equation:\n"
            + f"  x(t) = F({qdim},{qdim}) x(t-1) "
            + f"+ u(t),  u(t) ~ N(0, Q({qdim},{qdim}))\n"
        )
        
    def print_info(self, msg):
        print(f"{time.time()} - {msg}")
    
    

