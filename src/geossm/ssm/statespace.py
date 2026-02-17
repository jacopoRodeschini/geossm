"""
State Space Models Module
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit
import time
from geossm.utils import KeyStream
import numpy as np

from .statespace_results import SSMResults


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
        # Note: The original code had a slight lookahead (y_{t-1} from x_t).
        # This version uses the more standard y_t from x_t.
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

    # This is the function for a single loop iteration
    def kalman_step(carry, step_data):
        # 1. Unpack carry and step_data
        x_prev, P_prev, logL_accum = carry
        yt_slice, Xbeta_slice = step_data

        # PREDICTION
        x_pred = F @ x_prev
        P_pred = F @ P_prev @ F.T + Q

        # RESIDUAL
        nan_mask = jnp.isnan(yt_slice)
        Xb = Xbeta_slice @ beta
        e = yt_slice - Xb - (H @ x_pred)
        e = jnp.where(nan_mask, 0.0, e)

        # MODIFIED H
        Hna_dense = H_dense * (~nan_mask)[:, None]

        # WOODBURY
        invP_pred = solve(P_pred, Iq)
        M = invP_pred + Hna_dense.T @ (invR @ Hna_dense)
        invM = solve(M, Iq)
        invSigmaE = invR - invR @ Hna_dense @ invM @ Hna_dense.T @ invR

        # KALMAN GAIN
        K = P_pred @ Hna_dense.T @ invSigmaE

        # UPDATE STATE
        x_upd = x_pred + K @ e
        P_upd = P_pred - K @ Hna_dense @ P_pred

        # LOG-LIKELIHOOD
        logdetSigmaE = (
            jnp.linalg.slogdet(M)[1]
            + jnp.linalg.slogdet(P_pred)[1]
            + jnp.sum(jnp.log(R_diag))
        )
        logL_accum += logdetSigmaE + e.T @ (invSigmaE @ e)

        # 2. Pack carry for next step and outputs for this step
        next_carry = (x_upd, P_upd, logL_accum)
        outputs = {
            "x_t": x_upd, "P_t": P_upd, "K": K,
            "x_t_1": x_pred, "P_t_1": P_pred, "invP_t_1": invP_pred
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
        kalman_step, initial_carry, scan_inputs)

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
    x_t_1 = jnp.concatenate(
        [jnp.zeros((q, 1), dtype=dtype), x_t_1], axis=1)

    P_t = jnp.concatenate([Sigma0[:, :, None], P_t], axis=2)
    P_t_1 = jnp.concatenate(
        [jnp.zeros(Sigma0.shape, dtype=dtype)[:, :, None], P_t_1], axis=2)
    invP_t_1 = jnp.concatenate(
        [jnp.diag(1/Sigma0.diagonal())[:, :, None], invP_t_1], axis=2)
    logL = -0.5 * final_logL

    jax.block_until_ready(x_t)

    return x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL

@jit
def _smoother_kernelJAX(H, F, x_t, P_t, Klast, x_t_1, P_t_1, invP_t_1):
    
    dtype = x_t.dtype.type()
    q = F.shape[0]

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
        x_t_curr, P_t_curr, x_t_1_next, P_t_1_next, invP_t_1_next, P_t_prev, invP_t_1_curr = inputs

        # --- Core Smoother Logic (from your original loop) ---
        J_t_1 = P_t_curr @ F.T @ invP_t_1_next

        x_T_curr = x_t_curr + J_t_1 @ (x_T_next - x_t_1_next)
        P_T_curr = P_t_curr + J_t_1 @ (P_T_next - P_t_1_next) @ J_t_1.T

        # Lag-one covariance
        J_t_2 = P_t_prev @ F.T @ invP_t_1_curr
        term = P_T_1_next - F @ P_t_curr
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
    P_T_1_last = (jnp.eye(q, dtype=dtype) -
                    Klast @ H) @ F @ P_t[:, :, -2]
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
    invP_t_1_curr_padded = jnp.concatenate(
        [q_q_pad, xs_invP_t_1_curr], axis=2)

    # Now, put them into a tuple and reverse for the backward pass
    # Transposing to (T, ...) shape for scan
    xs = (
        xs_x_t.T,
        jnp.moveaxis(xs_P_t, 2, 0),
        xs_x_t_1.T,
        jnp.moveaxis(xs_P_t_1, 2, 0),
        jnp.moveaxis(xs_invP_t_1, 2, 0),
        jnp.moveaxis(P_t_prev_padded, 2, 0),
        jnp.moveaxis(invP_t_1_curr_padded, 2, 0)
    )
    # Reverse time axis for backward pass
    xs_reversed = jax.tree.map(lambda x: jnp.flip(x, axis=0), xs)

    # --- Run the scan ---
    # The final carry is not needed, we use the stacked outputs
    tStart = time.time()

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

    jax.block_until_ready(x_T)
    return x_T, P_T, P_T_1

@jit
def _compute_expected_values_kernelJAX(H, x_T, P_T, P_T_1, Xbeta, beta): 
    """ JIT-compiled kernel for computing expected values needed for M-step in EM. This is a straightforward implementation that can be optimized further if needed. """
    
    # Slices of the smoothed states
    # x_t terms range from t=1 to T
    # x_{t-1} terms range from t=0 to T-1
    x_t_slice = x_T[:, 1:]      # Shape: [q, T]
    x_tm1_slice = x_T[:, :-1]   # Shape: [q, T]

    # --- 1. Compute predicted observations (y_hat) ---
    # The term Xbeta @ beta can be computed efficiently using einsum.
    # y_hat_t = Xbeta_t @ beta + H @ x_t
    y_hat_covariate_term = jnp.einsum('pkt,k->pt', Xbeta, beta)
    y_hat_state_term = H @ x_t_slice
    y_hat = y_hat_covariate_term + y_hat_state_term

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

    jax.block_until_ready(S00)
    return y_hat, S11, S10, S00
        

# %% State Space Model Class
class StateSpaceModel:
    """
    A class representing a State Space Model with Kalman filtering capabilities.
    """

    def __init__(self, H, R, F, Q, x0=None, Sigma0=None, Xbeta=None, beta=None, xbeta_names=None, dtype=jnp.float32):
        """
        Initialize the State Space Model with system matrices and initial state.
        """
        self.dtype = dtype  # Data type for computations

        self._F = None  # State transition matrix
        self._H = None  # Observation matrix
        self._Q = None  # Process noise covariance
        self._q = None  # State dimension
        self._R = None  # Observation noise covariance
        self._x0 = None  # Initial state estimate
        self._Sigma0 = None  # Initial covariance estimate
        self._Xbeta = None  # Exogenous variables
        self._beta = None  # Coefficients for exogenous variables
        self._xbeta_names = None
        self._T = None  # Time length
        self._p = None  # number of measurement equation
        self._q = None  # number of state equation
        self._b = None  # number of regression coefficent

        self._type = 'Linear (Gaussian)'
        self._order = '(1, 0)'  # Placeholder for ARMA order if needed
        
    
        # Set the initial state starting values if not provided
        if x0 is None:
            x0 = np.zeros(F.shape[0])
        if Sigma0 is None:
            Sigma0 = np.eye(F.shape[0])

        # Set default Xbeta and beta if not provided
        if Xbeta is None:
            Xbeta = np.zeros((H.shape[0], 1, 1))
        if beta is None:
            beta = np.zeros(Xbeta.shape[1])

        self.set(H=H, F=F, Q=Q, R=R, x0=x0, Sigma0=Sigma0, Xbeta=Xbeta, beta=beta, xbeta_names=xbeta_names)

        # define the filtered attribute ?
    @property
    def type(self):
        return self._type
    
    @property
    def order(self):
        return self._order
    
    @property
    def shape(self):
        return (self._p, self._q, self._T)

    def __call__(self, y_t):
        """
        Docstring for __call__

        :param self: Run the estimation of the state == fitler + smoother
        :param y_t: Observed dataset
        """
        return self.smoother(y_t)

    def set(self, H=None, R=None, F=None, Q=None, x0=None, Sigma0=None, Xbeta=None, beta=None, xbeta_names=None):
        """
        Set model parameters and matrices.
        @ return: None
        @ param F: State transition matrix
        @ param H: Observation matrix
        @ param Q: Process noise covariance
        @ param R: Observation noise covariance
        @ param x0: Initial state estimate
        @ param Sigma0: Initial covariance estimate
        @ param Xbeta: Exogenous variables
        @ param beta: Coefficients for exogenous variables
        """
        # Check parameters

        self._update_parameters(
            F=F, H=H, Q=Q, R=R, x0=x0, Sigma0=Sigma0, Xbeta=Xbeta, beta=beta, xbeta_names=xbeta_names)

        flag, msg = self._check_parameters()
        if not flag:
            raise ValueError(msg)

    def _update_parameters(self, H=None, R=None, F=None, Q=None, x0=None, Sigma0=None, Xbeta=None, beta=None,xbeta_names=None ):
        """
        Helper function to update model parameters if provided.
        """

        if H is not None:
            self._H = jnp.asarray(H, dtype=self.dtype)
            self._p = H.shape[0]

        if R is not None:
            self._R = jnp.asarray(R, dtype=self.dtype)

        if F is not None:
            self._F = jnp.asarray(F, dtype=self.dtype)
            self._q = F.shape[0]

        if Q is not None:
            self._Q = jnp.asarray(Q, dtype=self.dtype)

        if x0 is not None:
            self._x0 = jnp.asarray(x0, dtype=self.dtype)

        if Sigma0 is not None:
            self._Sigma0 = jnp.asarray(Sigma0, dtype=self.dtype)

        if Xbeta is not None:
            self._Xbeta = jnp.asarray(Xbeta, dtype=self.dtype)
            # infer time length from Xbeta shape (p, b, T)
            try:
                self._T = int(self._Xbeta.shape[2])
            except Exception:
                self._T = None

        if beta is not None:
            self._beta = jnp.asarray(beta, dtype=self.dtype)
            try:
                self._b = int(self._beta.shape[0])
            except Exception:
                self._b = None
        
        if xbeta_names is not None:
            if len(xbeta_names) != len(self._b):
                raise ValueError(
                    f"Expected {len(self._b)} xbeta names, got {len(xbeta_names)}."
                )
            self._xbeta_names = xbeta_names
            
        else:
            self._xbeta_names = [f"X_{i}" for i in range(self._b)]


        return True

    @ property
    def xbeta_names(self):
        return self._xbeta_names

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
                return tuple(np.asarray(x).shape)
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
                "Number of time steps `T` is not set (inferred from Xbeta).")
            flag = False
        if b is None:
            messages.append("Number of regression coefficients `b` is not set.")
            flag = False

        if not flag:
            return flag, "\n".join(messages)

        # Helper: check positive semidefinite (symmetric) with tolerance
        def is_pos_semidef(mat):
            A = np.asarray(mat)
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                return False
            # symmetry check
            if not np.allclose(A, A.T, atol=1e-8):
                return False
            # eigenvalues >= -tol
            eigs = np.linalg.eigvalsh(A)
            return np.all(eigs >= -1e-8)

        # Sigma0: should be (q, q)
        sigma0_shape = shape_str(self.Sigma0)
        if sigma0_shape not in [(q, q), (q,), (q,)]:
            messages.append(
                f"Sigma0 must be shape ({q},{q}), got {sigma0_shape}.")
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
                f"R must be shape ({p},{p}) (or scalar for p=1), got {R_shape}.")
            flag = False
        else:
            if p > 1:
                if not is_pos_semidef(self.R):
                    messages.append(
                        "R must be symmetric positive semidefinite.")
                    flag = False
            else:
                # scalar case
                Rval = np.asarray(self.R).ravel()[0]
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
            messages.append(
                f"Sigma0 must be shape ({q},{q}), got {Sigma0_shape}.")
            flag = False
        # Check positive semidefinite
        # else:
        #     if not is_pos_semidef(self.Sigma0):
        #         messages.append(
        #             "Sigma0 must be symmetric positive semidefinite.")
        #         flag = False

        # Xbeta: (p, b, T)
        Xbeta_shape = shape_str(self.Xbeta)
        if Xbeta_shape != (p, b, T):
            messages.append(
                f"Xbeta must be shape ({p},{b},{T}), got {Xbeta_shape}.")
            flag = False

        # beta: (b,)
        beta_shape = shape_str(self.beta)
        if beta_shape != (b,):
            messages.append(f"beta must be shape ({b},), got {beta_shape}.")
            flag = False

        return flag, "\n".join(messages)

    def estimate(self, y_t, H=None, R=None, F=None, Q=None, x0=None, Sigma0=None, Xbeta=None, beta=None, xbeta_names = None):


        # run the smoother
        x_T, P_T, P_T_1, logL, tdelta_filter, tdelta_smoother = self.smoother(y_t, H=H, R=R, F=F, Q=Q, x0=x0, Sigma0=Sigma0, Xbeta=Xbeta, beta=beta, xbeta_names=xbeta_names)

        # compute expected values
        y_hat, S11, S10, S00, tdelta_expectation = self.computeExpectedValues(
            x_T, P_T, P_T_1)
        
        results = SSMResults(
            y_hat=y_hat, Xbeta=self.Xbeta, beta=self.beta, xbeta_names=self.xbeta_names,
            x_T=x_T, P_T=P_T, P_T_1=P_T_1, S11=S11, S10=S10, S00=S00, logL=logL,
            tdelta_filter=tdelta_filter, tdelta_smoother=tdelta_smoother, tdelta_expectation=tdelta_expectation
        )
        
        return results

        # return y_hat, x_T, P_T, P_T_1, S11, S10, S00, logL, tdelta_filter, tdelta_smoother, tdelta_expectation

    def filter(self, y_t, H=None, R=None, F=None, Q=None, x0=None, Sigma0=None, Xbeta=None, beta=None, xbeta_names = None) -> tuple:
        """
        Kalman Filter using jax.lax.scan for variable-length inputs.

        ========= References ==========
        | 1. Durbin, J., & Koopman, S. J. (2012). Time Series Analysis by State Space Methods. Oxford University Press. 
        """
        # Update parameters if provided
        self.set(H=H, R=R, F=F, Q=Q, x0=x0, Sigma0=Sigma0, Xbeta=Xbeta, beta=beta, xbeta_names=xbeta_names)
        
        # Run the scan
        tStart = time.time()

        x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL = _filter_kernelJAX(y_t, self.H, self.R, self.F, self.Q, self.x0, 
                                                                      self.Sigma0, self.Xbeta, self.beta)

        tDelta = time.time() - tStart

        # compute expected values (given the filterd values)
        y_hat, S11, S10, S00, tdelta_expectation = self.computeExpectedValues(
            x_t, P_t, P_t_1)
        

        results  = SSMResults(model=self, 
                              y_obs=y_t, Xbeta=self.Xbeta, beta=self.beta, xbeta_names=self.xbeta_names,
                              x_filtered=x_t, P_filtered=P_t, K=K, x_pred=x_t_1, 
                              P_pred=P_t_1, invP_pred=invP_t_1, llf =logL, time_filter=tDelta,
                              y_hat = y_hat, S11=S11, S10=S10, S00=S00, time_expectation=tdelta_expectation)

        return results
        # return (x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL, tDelta)

    def smoother(self, y_t, H=None, R=None, F=None, Q=None, x0=None, Sigma0=None, Xbeta=None, beta=None, xbeta_names = None) -> tuple:
        """
        Kalman smoother using jax.lax.scan for efficient, T-independent compilation.

        description: filtering and smoothing algorithm for linear state space models

        ========= References ==========
        | 1. Durbin, J., & Koopman, S. J. (2012). Time Series Analysis by State Space Methods. Oxford University Press. 
        """
        # Update parameters if provided
        self.set(H=H, R=R, F=F, Q=Q, x0=x0, Sigma0=Sigma0, Xbeta=Xbeta, beta=beta, xbeta_names=xbeta_names)

        # First, run the filter to get necessary inputs for the smoother
        x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL, td_filter = self.filter(
            y_t, H, R, F, Q, x0, Sigma0, Xbeta, beta)
        
        # Now run the smoother
        tStart = time.time() 
        x_T, P_T, P_T_1 = _smoother_kernelJAX(
            self.H, self.F, x_t, P_t, K, x_t_1, P_t_1, invP_t_1)
        
        td_smoother = time.time() - tStart

        return (x_T, P_T, P_T_1, logL, td_filter, td_smoother)


    def computeExpectedValues(self, x_T, P_T, P_T_1) -> tuple:

        # Note: Type conversions are omitted for clarity. It's often better
        # to ensure inputs have the correct dtype before calling a JIT-compiled function.
        tStart = time.time()

        y_hat, S11, S10, S00 = _compute_expected_values_kernelJAX(self.H, x_T, P_T, P_T_1, self.Xbeta, self.beta)

        tDelta = time.time() - tStart

        return (y_hat, S11, S10, S00, tDelta)

    def sim(self, seed=1234, Xbeta=None) -> jnp.ndarray:

        # def sim(keys, R, F, H, Q, x0, Sigma0, Xbeta, beta):
        """
        Simulates a time series from the state-space model using JAX and a Python for-loop.
        This version does NOT use JIT compilation and is therefore slower.

        Args:
            key: is a JAX PRNGKey strem object (next methods).
            ... other model parameters.

        Returns:
            y_t : (p, T) JAX array of simulated observations
            x_t : (q, T+1) JAX array of simulated state vectors [x_0, ..., x_T]
        """
        # Get dimensions from input shapes
        if Xbeta is None:
            Xbeta = self.Xbeta
            T = self.T
        else:
            T = Xbeta.shape[2]
            # check Xbeta shape compatibility
            if Xbeta.shape[0] != self.p:
                raise ValueError(
                    f"Xbeta first dimension must be {self.p}, got {Xbeta.shape[0]}")
            if Xbeta.shape[1] != self.b:
                raise ValueError(
                    f"Xbeta second dimension must be {self.b}, got {Xbeta.shape[1]}")

        if isinstance(seed, KeyStream):
            keys = seed
        else:
            # Initialize PRNGKey stream
            main_key = jax.random.PRNGKey(seed)
            seed, main_key = jax.random.split(main_key)

            keys = KeyStream(seed)

        # Call the simulation kernel to generate the time series
        tStart = time.time()
        y_t_sim, x_t_sim = _sim_kernelJAX(
            keys, self.H, self.R, self.F, self.Q, self.x0, self.Sigma0, Xbeta, self.beta)

        tdelta = time.time() - tStart
        return y_t_sim, x_t_sim, tdelta

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
    def shape(self):
        return (self.p, self.q, self.T)

    # ----------------- Pickle support -----------------
    def __getstate__(self):
        """Return a serializable state for pickling.

        Convert JAX arrays to NumPy arrays and store basic metadata.
        """
        def to_np(x):
            if x is None:
                return None
            try:
                return np.array(x)
            except Exception:
                return x

        state = {
            'F': to_np(self._F),
            'H': to_np(self._H),
            'Q': to_np(self._Q),
            'R': to_np(self._R),
            'x0': to_np(self._x0),
            'Sigma0': to_np(self._Sigma0),
            'Xbeta': to_np(self._Xbeta),
            'beta': to_np(self._beta),
            'T': self._T,
            'p': self._p,
            'q': self._q,
            'b': self._b,
            # store dtype name for robust restoration
            'dtype': getattr(self.dtype, 'name', str(self.dtype)),
        }
        return state

    def __setstate__(self, state):
        """Restore object state from pickled state.

        Arrays are converted back to JAX arrays with the original dtype.
        """
        # Restore dtype first
        dt_name = state.get('dtype', None)
        try:
            self.dtype = jnp.dtype(dt_name) if dt_name is not None else jnp.float32
        except Exception:
            # fallback
            try:
                self.dtype = getattr(jnp, dt_name)
            except Exception:
                self.dtype = jnp.float32

        def to_jax(x):
            if x is None:
                return None
            try:
                return jnp.asarray(x, dtype=self.dtype)
            except Exception:
                return x

        # Restore arrays and metadata
        self._F = to_jax(state.get('F', None))
        self._H = to_jax(state.get('H', None))
        self._Q = to_jax(state.get('Q', None))
        self._R = to_jax(state.get('R', None))
        self._x0 = to_jax(state.get('x0', None))
        self._Sigma0 = to_jax(state.get('Sigma0', None))
        self._Xbeta = to_jax(state.get('Xbeta', None))
        self._beta = to_jax(state.get('beta', None))

        self._T = state.get('T', None)
        self._p = state.get('p', None)
        self._q = state.get('q', None)
        self._b = state.get('b', None)

        # Ensure other attributes exist with sensible defaults
        if not hasattr(self, 'dtype'):
            self.dtype = jnp.float32
        if not hasattr(self, '_F'):
            self._F = None
        if not hasattr(self, '_H'):
            self._H = None
        if not hasattr(self, '_Q'):
            self._Q = None
        if not hasattr(self, '_R'):
            self._R = None
        if not hasattr(self, '_x0'):
            self._x0 = None
        if not hasattr(self, '_Sigma0'):
            self._Sigma0 = None
        if not hasattr(self, '_Xbeta'):
            self._Xbeta = None
        if not hasattr(self, '_beta'):
            self._beta = None

    def summary(self, print_output: bool = True) -> str:
        """Return or print a short summary with key shapes and metrics."""
        st = "\nState Space Model Summary \n"

        st += "------------------------------------ \n"
        st += "State Space formulas:\n"
        st += f"y(t) = X {getattr(self.Xbeta, 'shape', None)} beta + H {self.H.shape} x(t) + e(t) ~N(0, R {self.R.shape}) \n"
        st += f"x(t) = F {self.F.shape} x(t-1) + u(t) ~N(0, Q {self.Q.shape}) \n \n"

        return st

    def __str__(self):
        """String representation of the model."""

        return self.summary(print_output=True)

    def __repr__(self):
        return self.__str__()


