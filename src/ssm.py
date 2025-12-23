"""
State Space Models Module
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit

class StateSpaceModel:
    """
    A class representing a State Space Model with Kalman filtering capabilities.
    """

    def __init__(self, F, H, Q, R, x0, Sigma0, Xbeta, beta, dtype=jnp.float32):
        """
        Initialize the State Space Model with system matrices and initial state.
        """
        self.dtype=dtype # Data type for computations
        self.set(F, H, Q, R, x0, Sigma0, Xbeta, beta)

    def set(self, F, H, Q, R, x0, Sigma0, Xbeta, beta):
        """
        Set model parameters.
        """
        self.F = jnp.asarray(F, dtype=self.dtype)  # State transition matrix
        self.H = jnp.asarray(H, dtype=self.dtype)  # Observation matrix
        self.Q = jnp.asarray(Q, dtype=self.dtype)  # Process noise covariance
        self.q = F.shape[0]  # State dimension
        self.R = jnp.asarray(R, dtype=self.dtype)  # Observation noise covariance
        self.x0 = jnp.asarray(x0, dtype=self.dtype)  # Initial state estimate
        self.Sigma0 = jnp.asarray(Sigma0, dtype=self.dtype)  # Initial covariance estimate
        self.Xbeta = jnp.asarray(Xbeta, dtype=self.dtype)  # Exogenous variables
        self.beta = jnp.asarray(beta, dtype=self.dtype) # Coefficients for exogenous variables
    
    @jit
    def filter(self, y_t, q) -> tuple:
        """
        Kalman Filter using jax.lax.scan for variable-length inputs.
        """
        p, T = y_t.shape

        # Pre-compute constants
        Iq = jnp.eye(q, dtype=self.dtype)
        R_diag = self.R.diagonal()
        invR_diag = jnp.reciprocal(R_diag)
        invR = jnp.diag(invR_diag)
        H_dense = self.H.astype(self.dtype)

        # This is the function for a single loop iteration
        def kalman_step(carry, step_data):
            # 1. Unpack carry and step_data
            x_prev, P_prev, logL_accum = carry
            yt_slice, Xbeta_slice = step_data

            # PREDICTION
            x_pred = self.F @ x_prev
            P_pred = self.F @ P_prev @ self.F.T + self.Q

            # RESIDUAL
            nan_mask = jnp.isnan(yt_slice)
            Xb = Xbeta_slice @ self.beta
            e = yt_slice - Xb - (self.H @ x_pred)
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
        initial_carry = (self.x0, self.Sigma0, self.dtype(0.0))
        # We need to transpose inputs so that T is the leading dimension
        # y_t: [p, T] -> [T, p]
        # Xbeta: [p, b, T] -> [T, p, b]
        scan_inputs = (y_t.T, jnp.moveaxis(self.Xbeta, -1, 0))

        # Run the scan
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
        x_t = jnp.concatenate([self.x0[:, None], x_t], axis=1)
        x_t_1 = jnp.concatenate([jnp.zeros((self.q, 1), dtype=self.dtype), x_t_1], axis=1)

        P_t = jnp.concatenate([self.Sigma0[:, :, None], P_t], axis=2)
        P_t_1 = jnp.concatenate(
            [jnp.zeros(self.Sigma0.shape, dtype=self.dtype)[:, :, None], P_t_1], axis=2)
        invP_t_1 = jnp.concatenate(
            [jnp.diag(1/self.Sigma0.diagonal())[:, :, None], invP_t_1], axis=2)

        logL = self.dtype(-0.5) * final_logL

        return (x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL)
      
    @jit
    def smoother(self, x_t, P_t, Klast, x_t_1, P_t_1, invP_t_1) -> tuple:
        """
        Kalman smoother using jax.lax.scan for efficient, T-independent compilation.
        """

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
        P_T_1_last = (jnp.eye(self.q, dtype=self.dtype) - Klast @ self.H) @ self.F @ P_t[:, :, -2]
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
        q_q_pad = jnp.zeros((self.q, self.q, 1), dtype=self.dtype)
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
            jnp.moveaxis(invP_t_1_curr_padded, 2, 0)
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

        return (x_T, P_T, P_T_1)

    @jit
    def computeExpectedValues(self, x_T, P_T, P_T_1) -> tuple:
    
        # Note: Type conversions are omitted for clarity. It's often better
        # to ensure inputs have the correct dtype before calling a JIT-compiled function.

        # Slices of the smoothed states
        # x_t terms range from t=1 to T
        # x_{t-1} terms range from t=0 to T-1
        x_t_slice = x_T[:, 1:]      # Shape: [q, T]
        x_tm1_slice = x_T[:, :-1]   # Shape: [q, T]

        # --- 1. Compute predicted observations (y_hat) ---
        # The term Xbeta @ beta can be computed efficiently using einsum.
        # y_hat_t = Xbeta_t @ beta + H @ x_t
        y_hat_covariate_term = jnp.einsum('pkt,k->pt', self.Xbeta, self.beta)
        y_hat_state_term = self.H @ x_t_slice
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

        return (y_hat, S11, S10, S00)
    
    def __str__(self):
        return f"StateSpaceModel(H={self.H}, F={self.F}, R={self.R}, Q={self.Q}, x0={self.x0}, Sigma0={self.Sigma0}, beta={self.beta})"