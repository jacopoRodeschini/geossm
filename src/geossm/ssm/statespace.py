"""
State Space Models Module
"""

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
from geossm.utils import _select_device, _to_backend


# %% JAX kernel functions for SSM

def _sim_kernelJAX(keys, H, R, F, Q, x0, Sigma0, Xbeta, beta):
    """
    JIT-compiled kernel for simulating a time series from the state-space model using JAX.
    This version uses jax.lax.scan for efficient looping.

    Args:
        keys: JAX PRNGKey stream object.
        ... other model parameters.
    Returns:
        y_t : (p, T) JAX array of simulated observations
        x_t : (q, T+1) JAX array of simulated state vectors
    """

    p = R.shape[0]
    q = F.shape[0]
    T = Xbeta.shape[2]

    # Pre-compute Cholesky decompositions
    chol_R = jnp.linalg.cholesky(R)
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
        obs_noise = chol_R @ jax.random.normal(keys.next(), shape=(p,))
        mean_reg = Xbeta[:, :, t] @ beta
        y_t = mean_reg + H @ x_current + obs_noise
        y_history.append(y_t)

        # 2. Evolve the state to the next step: x_{t+1} from x_t
        process_noise = chol_Q @ jax.random.normal(keys.next(), shape=(q,))
        x_next = F @ x_current + process_noise

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
def _filter_kernelJAX(y_t, H, R, F, Q, x0, Sigma0, Xbeta, beta):

    dtype = y_t.dtype.type()
    q = F.shape[0]

    # Pre-compute constants
    Iq = jnp.eye(q, dtype=dtype)
    R_diag = R.diagonal()
    invR_diag = jnp.reciprocal(R_diag)
    invR = jnp.diag(invR_diag)
    H_dense = H.astype(dtype)
    f_diag = jnp.diag(F)
    FF = jnp.outer(f_diag, f_diag)

    # This is the function for a single loop iteration
    def kalman_step(carry, step_data):
        # 1. Unpack carry and step_data
        x_prev, P_prev, logL_accum = carry
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
        #invP_pred = solve(P_pred, Iq)
        L_P = jnp.linalg.cholesky(P_pred + 1e-6 * Iq)  # Add small jitter for numerical stability
        invL_P = jax.scipy.linalg.solve_triangular(L_P, Iq, lower=True)
        invP_pred = invL_P.T @ invL_P

        M = invP_pred + Hna_dense.T @ (invR @ Hna_dense)
        
        L_M = jnp.linalg.cholesky(M + 1e-6 * Iq)  # Add small jitter for numerical stability
        invL_M = jax.scipy.linalg.solve_triangular(L_M, Iq, lower=True)
        invM = invL_M.T @ invL_M
        
        invSigmaE = invR - invR @ Hna_dense @ invM @ Hna_dense.T @ invR

        # KALMAN GAIN
        K = P_pred @ Hna_dense.T @ invSigmaE

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

        logL_accum += logdetSigmaE + e.T @ (invSigmaE @ e)

        # 2. Pack carry for next step and outputs for this step
        next_carry = (x_upd, P_upd, logL_accum)
        outputs = {
            "x_t": x_upd,
            "P_t": P_upd,
            "K": K,
            "x_t_1": x_pred,
            "P_t_1": P_pred,
            "invP_t_1": invP_pred,
        }
        return next_carry, outputs

    # Prepare initial state and inputs for scan
    # The scan will iterate over the time dimension
    initial_carry = (x0, Sigma0, 0.0)
    # We need to transpose inputs so that T is the leading dimension
    # y_t: [p, T] -> [T, p]
    # Xbeta: [p, b, T] -> [T, p, b]
    scan_inputs = (y_t.T, jnp.moveaxis(Xbeta, -1, 0))

    (final_x, final_P, final_logL), history = jax.lax.scan(
        kalman_step, initial_carry, scan_inputs
    )

    # Post-process the results from the history dictionary
    # The outputs will have T as the leading dimension, so we move it back
    x_t = jnp.moveaxis(history["x_t"], 0, -1)
    P_t = jnp.moveaxis(history["P_t"], 0, -1)
    K = history["K"][-1]  # Often only the final gain is needed
    x_t_1 = jnp.moveaxis(history["x_t_1"], 0, -1)
    P_t_1 = jnp.moveaxis(history["P_t_1"], 0, -1)
    invP_t_1 = jnp.moveaxis(history["invP_t_1"], 0, -1)

    # Add the initial state to the beginning of the time series arrays
    x_t = jnp.concatenate([x0[:, None], x_t], axis=1)
    x_t_1 = jnp.concatenate([jnp.zeros((q, 1), dtype=dtype), x_t_1], axis=1)

    P_t = jnp.concatenate([Sigma0[:, :, None], P_t], axis=2)
    P_t_1 = jnp.concatenate(
        [jnp.zeros(Sigma0.shape, dtype=dtype)[:, :, None], P_t_1], axis=2
    )
    invP_t_1 = jnp.concatenate(
        [jnp.diag(1 / Sigma0.diagonal())[:, :, None], invP_t_1], axis=2
    )
    logL = -0.5 * final_logL

    # jax.block_until_ready(x_t)

    return x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL


@jit
def _smoother_kernelJAX(H, F, x_t, P_t, Klast, x_t_1, P_t_1, invP_t_1):

    dtype = x_t.dtype.type()
    q = F.shape[0]
    f_diag = jnp.diag(F)

    def smoother_step(carry, inputs):
        """
        The body of the loop for a single time step.
        This function will be compiled once by `scan`.
        """
        # 1. Unpack the carry (state from the previous iteration, i.e., time t+1)
        x_T_next, P_T_next, P_T_1_next = carry

        # 2. Unpack the inputs for the current iteration (i.e., time t)
        # Note: The lag-one covariance calc needs P_t[t-2] and invP_t_1[t-1],
        # so we pass them in as well.
        (
            x_t_curr,
            P_t_curr,
            x_t_1_next,
            P_t_1_next,
            invP_t_1_next,
            P_t_prev,
            invP_t_1_curr,
        ) = inputs

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
    # Lag-one cov at T is special
    P_T_1_last = (jnp.eye(q, dtype=dtype) - Klast @ H) @ F @ P_t[:, :, -2]
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
    # x_t_1_next, P_t_1_next, invP_t_1_next
    xs_x_t_1 = x_t_1[:, 1:]
    xs_P_t_1 = P_t_1[:, :, 1:]
    xs_invP_t_1 = invP_t_1[:, :, 1:]

    # We need inputs from t=T-1, T-2, ..., 0 for the lag-one cov's J_t_2
    xs_invP_t_1_curr = invP_t_1[:, :, 1:-1]

    # Pad the arrays that are too short to align them for stacking
    # We need T elements for the scan (from T-1 down to 0)
    # P_t_prev needs one pad, invP_t_1_curr needs one pad
    q_q_pad = jnp.zeros((q, q, 1), dtype=dtype)
    P_t_prev_padded = jnp.concatenate([q_q_pad, xs_P_t_prev], axis=2)
    invP_t_1_curr_padded = jnp.concatenate([q_q_pad, xs_invP_t_1_curr], axis=2)

    # Now, put them into a tuple and reverse for the backward pass
    # Transposing to (T, ...) shape for scan
    xs = (
        xs_x_t.T,
        jnp.moveaxis(xs_P_t, 2, 0),
        xs_x_t_1.T,
        jnp.moveaxis(xs_P_t_1, 2, 0),
        jnp.moveaxis(xs_invP_t_1, 2, 0),
        jnp.moveaxis(P_t_prev_padded, 2, 0),
        jnp.moveaxis(invP_t_1_curr_padded, 2, 0),
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
def _compute_expected_values_kernelJAX(H, x_T, P_T, P_T_1, Xbeta, beta):
    """JIT-compiled kernel for computing expected values needed for M-step in EM. This is a straightforward implementation that can be optimized further if needed."""

    # Slices of the smoothed states
    # x_t terms range from t=1 to T
    # x_{t-1} terms range from t=0 to T-1
    x_t_slice = x_T[:, 1:]  # Shape: [q, T]
    x_tm1_slice = x_T[:, :-1]  # Shape: [q, T]

    # --- 1. Compute predicted observations (y_hat) ---
    # The term Xbeta @ beta can be computed efficiently using einsum.
    # y_hat_t = Xbeta_t @ beta + H @ x_t
    y_hat, _ = _compute_predict_kernel_JAX(H, x_T, P_T, Xbeta, beta)
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
def _compute_predict_kernel_JAX(H, x_T, P_T, Xbeta, beta):
    """Compute the predicted observations (y_hat) based on the smoothed states and the model parameters."""
    
    # Compute the expected valure
    y_hat_covariate_term = jnp.einsum("pkt,k->pt", Xbeta, beta)
    y_hat_state_term = H @ x_T[:, 1:]  # Use the smoothed states for prediction
    y_hat = y_hat_covariate_term + y_hat_state_term

    # compute the plugin uncertainty
    Sigma_y_hat = jnp.einsum("ip,pqt,jq->ijt", H, P_T[:, :, 1:], H)

    return y_hat, Sigma_y_hat

# %% State Space Model Class
class StateSpaceModel:
    """
    A class representing a State Space Model with Kalman filtering capabilities.
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
        """
        Initialize the State Space Model with system matrices and initial state.
        """
        self.dtype = dtype  # Data type for computations
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
        """
        Docstring for __call__

        :param self: Run the estimation of the state == fitler + smoother
        :param y_t: Observed dataset
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
        """
        Set model parameters and matrices.
        @ return: None
        @ param F: State transition matrix
        @ param H: Observation matrix
        @ param Q: Process noise covariance
        @ param R: Observation noise covariance
        @ param x0: Initial state estimate
        @ param Sigma0: Initial covariance estimate
        @ param y_t: Observed dataset
        @ param Xbeta: Exogenous variables
        @ param beta: Coefficients for exogenous variables
        @ param xbeta_names: Names for the exogenous variables (optional)
        @ param yname: Name for the dependent variable (optional)
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
            self._R = self._prepare_array(R)

        if F is not None:
            self._F = self._prepare_array(F)
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
        return self._backend

    def _prepare_array(self, x):
        """Cast to the model dtype and commit the array to the configured backend device."""
        return _to_backend(self._backend, jnp.asarray(x, dtype=self.dtype))[0]

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

        # R: (p, p) or scalar for p==1
        R_shape = shape_str(self.R)
        if not (R_shape == (p, p) or (p == 1 and R_shape in [(1,), (1, 1)])):
            messages.append(
                f"R must be shape ({p},{p}) (or scalar for p=1), got {R_shape}."
            )
            flag = False
        else:
            if p > 1:
                if not is_pos_semidef(self.R):
                    messages.append("R must be symmetric positive semidefinite.")
                    flag = False
            else:
                # scalar case
                Rval = jnp.asarray(self.R).ravel()[0]
                if Rval < 0:
                    messages.append("R (variance) must be non-negative.")
                    flag = False

        # F: (q, q)
        F_shape = shape_str(self.F)
        if F_shape != (q, q):
            messages.append(f"F must be shape ({q},{q}), got {F_shape}.")
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
    ):

        # run the smoother ( = filter + backward pass)
        # x_T, P_T, P_T_1, logL, tdelta_filter, tdelta_smoother = self.smoother(y_t, yname=yname, H=H, R=R, F=F, Q=Q, x0=x0, Sigma0=Sigma0, Xbeta=Xbeta, beta=beta, xbeta_names=xbeta_names)
        smooth_results = self.smoother(
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

        x_T = smooth_results.x_smoothed
        P_T = smooth_results.P_smoothed
        P_T_1 = smooth_results.P_pred_smoothed

        # compute expected values
        y_hat, S11, S10, S00, tdelta_expectation = self.computeExpectedValues(
            x_T, P_T, P_T_1
        )

        # Create the results object with the expected values
        # update the results object with the smoothed values and expected values
        smooth_results = smooth_results.update(
            y_hat=y_hat, S11=S11, S10=S10, S00=S00, time_expectation=tdelta_expectation
        )

        return smooth_results
        # return y_hat, x_T, P_T, P_T_1, S11, S10, S00, logL, tdelta_filter, tdelta_smoother, tdelta_expectation

    def predict(self, H, x_T, P_T, Xbeta=None, beta=None):
        """
        Compute predicted observations (y_hat) based on smoothed states and model parameters.
        """
        y_hat, Sigma_y_hat = _compute_predict_kernel_JAX(H, x_T, P_T, Xbeta, beta)
        return y_hat, Sigma_y_hat
    
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
        """
        Kalman Filter using jax.lax.scan for variable-length inputs.

        ========= References ==========
        | 1. Durbin, J., & Koopman, S. J. (2012). Time Series Analysis by State Space Methods. Oxford University Press.
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

        x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL = _filter_kernelJAX(
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

        # compute expected values (given the filterd values)
        # y_hat, S11, S10, S00, tdelta_expectation = self.computeExpectedValues(
        #     x_t, P_t, P_t_1
        # )

        results = StateSpaceResults(
            model=self,
            x_filtered=x_t,
            P_filtered=P_t,
            K=K,
            x_pred=x_t_1,
            P_pred=P_t_1,
            invP_pred=invP_t_1,
            llf=logL,
            time_filter=tDelta,
            y_hat=None,
            S11=None,
            S10=None,
            S00=None,
            time_expectation=0.0,
        )

        return results
        # return (x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL, tDelta)

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
    ) -> tuple:
        """
        Kalman smoother using jax.lax.scan for efficient, T-independent compilation.

        description: filtering and smoothing algorithm for linear state space models

        ========= References ==========
        | 1. Durbin, J., & Koopman, S. J. (2012). Time Series Analysis by State Space Methods. Oxford University Press.
        """

        # First, run the filter to get necessary inputs for the smoother
        res_filter = self.filter(
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
        )

        # Now run the smoother
        tStart = time.time()
        x_T, P_T, P_T_1 = _smoother_kernelJAX(
            self.H,
            self.F,
            res_filter.x_filtered,
            res_filter.P_filtered,
            res_filter.K,
            res_filter.x_pred,
            res_filter.P_pred,
            res_filter.invP_pred,
        )

        jax.block_until_ready(x_T)
        td_smoother = time.time() - tStart

        # compute expected values (given the smoothed values)
        y_hat, S11, S10, S00, tdelta_expectation = self.computeExpectedValues(
            x_T, P_T, P_T_1
        )

        # update the results object with the smoothed values and expected values
        res_smooth = res_filter.update(
            x_smoothed=x_T,
            P_smoothed=P_T,
            P_pred_smoothed=P_T_1,
            y_hat=y_hat,
            S11=S11,
            S10=S10,
            S00=S00,
            time_smoother=td_smoother,
            time_expectation=tdelta_expectation,
        )

        # return (x_T, P_T, P_T_1, res_filter.llf, res_filter.time_filter, td_smoother)

        return res_smooth

    def computeExpectedValues(self, x_T, P_T, P_T_1) -> tuple:

        # Note: Type conversions are omitted for clarity. It's often better
        # to ensure inputs have the correct dtype before calling a JIT-compiled function.
        tStart = time.time()

        y_hat, S11, S10, S00 = _compute_expected_values_kernelJAX(
            self.H, x_T, P_T, P_T_1, self.Xbeta, self.beta
        )

        jax.block_until_ready(y_hat)
        tDelta = time.time() - tStart

        return (y_hat, S11, S10, S00, tDelta)

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
        """
        Simulates a time series from the state-space model using JAX and a Python for-loop.
        This version does NOT use JIT compilation and is therefore slower.

        Args:
            key: is a JAX PRNGKey strem object (next methods).
            ... other model parameters.
            stats: if True, compute and get also the simulation statistics.

        Returns:
            y_t : (p, T) JAX array of simulated observations
            x_t : (q, T+1) JAX array of simulated state vectors [x_0, ..., x_T]
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
        """
        Compute and print a compact summary of theoretical vs empirical variances
        for the LR-SSM. Returns a dict with numeric results.

        block_p is the dimension of the observation (number of locations) for each process in case of multivariate processes simulation
        block_q is the dimension of the latent state for each process in case of multivariate processes simulation
        
        """
        from scipy.linalg import solve_discrete_lyapunov

        # Basic checks
        for attr in ("F", "Q", "H", "R"):
            if not hasattr(self, attr):
                raise AttributeError(f"model is missing '{attr}' attribute")
        F, Q, H, R = self.F, self.Q, self.H, self.R

        # Solve Lyapunov: P = F P F^T + Q
        P = solve_discrete_lyapunov(F, Q)

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
        temp = jnp.diag(R)
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
        """Returns the time lenght."""
        return self._T

    @property
    def p(self):
        """Returns the dimension of the measurement equation."""
        return self._p

    @property
    def q(self):
        """Returns the dimension of the latent equation."""
        return self._q

    @property
    def b(self):
        """Returns the dimension of the regression term coefficent vector."""
        return self._b

    @property
    def y_t(self):
        """Returns the observation matrix."""
        return self._y_t
    
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    @property
    def params_names(self):
        return self._params_names

    @params_names.setter
    def params_names(self, value):
        self._params_names = value

    @property
    def params_dim(self):
        return self._params_dim

    @property
    def xbeta_names(self):
        return self._xbeta_names

    @xbeta_names.setter
    def xbeta_names(self, value):
        self._xbeta_names = value

    @property
    def yname(self):
        return self._yname

    @yname.setter
    def yname(self, value):
        self._yname = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value: str):
        self._type = value

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value

    @property
    def shape(self):
        return (self._p, self._q, self._T)

    @property
    def Xbeta(self):
        """Returns the observation matrix H."""
        return self._Xbeta

    @Xbeta.setter
    def Xbeta(self, value):
        """Sete the Xbeta matrix."""
        self.set(Xbeta=value)

    @property
    def beta(self):
        """Returns the observation matrix H."""
        return self._beta

    @beta.setter
    def beta(self, value):
        """Sete the beta matrix."""
        self.set(beta=value)

    @property
    def H(self):
        """Returns the observation matrix H."""
        return self._H

    @H.setter
    def H(self, value):
        """Sete the H matrix."""
        self.set(H=value)

    @property
    def R(self):
        """Returns the measurement noise covariance R."""
        return self._R

    @R.setter
    def R(self, value):
        """Sete the R matrix."""
        self.set(R=value)

    @property
    def F(self):
        """Returns the state transition matrix F."""
        return self._F

    @F.setter
    def F(self, value):
        """Sete the F matrix."""
        self.set(F=value)

    @property
    def Q(self):
        """Returns the process noise covariance Q."""
        return self._Q

    @Q.setter
    def Q(self, value):
        """Sete the Q matrix."""
        self.set(Q=value)

    @property
    def x0(self):
        """Returns the initial state estimate x0."""
        return self._x0

    @property
    def Sigma0(self):
        """Returns the initial state covariance Sigma0."""
        return self._Sigma0

    @property
    def yname(self):
        return self._yname

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
        }
        return state

    def __setstate__(self, state):
        """Restore object state from pickled state.

        Arrays are converted back to JAX arrays with the original dtype.
        """
        # Restore dtype first
        dt_name = state.get("dtype", None)
        try:
            self.dtype = jnp.dtype(dt_name) if dt_name is not None else jnp.float32
        except Exception:
            # fallback
            try:
                self.dtype = getattr(jnp, dt_name)
            except Exception:
                self.dtype = jnp.float32

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

        # Ensure other attributes exist with sensible defaults
        if not hasattr(self, "dtype"):
            self.dtype = jnp.float32
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
                ("JAX backend:", lambda: [f"{jax.default_backend()}"]),
                ("JAX devices:", lambda: [f"{jax.devices()}"]),
            ]
        )

        top_right = dict(
            [
                ("Shape (p, q, T) :", lambda: [f"(p = {p}, q = {q}, T = {T})"]),
                (
                    "Diag. R",
                    lambda: [f"{jnp.mean(jnp.diag(self.R)):2f}"
                        if self.R is not None
                        else "None"]
                ),
                (
                    "Diag. Q",
                    lambda: [f"{jnp.mean(jnp.diag(self.Q)):2f}"
                        if self.Q is not None
                        else "None"],
                ),
                (
                    "Diag. F",
                    lambda: [
                        f"{jnp.mean(jnp.diag(self.F)):2f}"
                        if self.F is not None
                        else "None"],
                ),
                (
                    "mean x0",
                    lambda: [
                        f"{jnp.mean(self.x0):2f}" if self.x0 is not None else "None"],
               ),
                (
                    "mean Sigma0",
                    lambda: [
                        f"{jnp.mean(jnp.diag(self.Sigma0)):2f}"
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
        """Return or print a structured summary of the model."""
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
