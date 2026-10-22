"""Independent NumPy (float64) reference implementations used by the test suite.

Deliberately written as the textbook dense recursions (no Woodbury identity,
no jitter, explicit missing-data row selection) so they share no code or
numerical shortcuts with the JAX kernels in ``geossm.ssm.statespace``.

Model (diagonal F and R, as in geossm)::

    y_t = H x_t + X_t beta + e_t,   e_t ~ N(0, diag(R))
    x_t = diag(F) x_{t-1} + eta_t,  eta_t ~ N(0, Q)
    x_0 ~ N(x0, Sigma0)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SSMProblem:
    """A small, well-conditioned state-space problem (NumPy float64 arrays)."""

    H: np.ndarray  # (p, q)
    R: np.ndarray  # (p,)   diagonal of the observation-noise covariance
    F: np.ndarray  # (q,)   diagonal of the transition matrix
    Q: np.ndarray  # (q, q)
    x0: np.ndarray  # (q,)
    Sigma0: np.ndarray  # (q, q)
    Xbeta: np.ndarray  # (p, b, T)
    beta: np.ndarray  # (b,)
    y: np.ndarray  # (p, T), may contain NaN

    @property
    def p(self):
        return self.H.shape[0]

    @property
    def q(self):
        return self.H.shape[1]

    @property
    def T(self):
        return self.y.shape[1]

    def with_missing(self, mask):
        """Copy of the problem with ``y[mask] = NaN``."""
        y = self.y.copy()
        y[mask] = np.nan
        return SSMProblem(**{**self.__dict__, "y": y})


def make_problem(seed=0, p=6, q=3, T=10, b=2):
    """Random, well-conditioned problem with non-trivial x0/Sigma0 and covariates."""
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(p, q))
    R = rng.uniform(0.2, 1.0, size=p)
    F = rng.uniform(-0.9, 0.95, size=q)
    A = rng.normal(size=(q, q))
    Q = A @ A.T + 0.5 * np.eye(q)
    x0 = rng.normal(size=q)
    Sigma0 = np.diag(rng.uniform(0.5, 2.0, size=q))
    Xbeta = rng.normal(size=(p, b, T))
    beta = rng.normal(size=b)
    y = rng.normal(size=(p, T)) + np.einsum("pbt,b->pt", Xbeta, beta)
    return SSMProblem(H, R, F, Q, x0, Sigma0, Xbeta, beta, y)


def kalman_filter(prob):
    """Dense Kalman filter, missing entries dropped row-wise.

    Returns
    -------
    dict with
        x_filt (q, T+1), P_filt (q, q, T+1)   index 0 = prior (x0, Sigma0)
        x_pred (q, T+1), P_pred (q, q, T+1)   index 0 unused (zeros)
        llf    observed-data Gaussian log-likelihood INCLUDING -n/2 log(2 pi)
        llf_no_const  same, without the 2 pi constant (geossm's convention before F15)
        n_obs  number of observed entries
    """
    p, q, T = prob.p, prob.q, prob.T
    Fm = np.diag(prob.F)
    x_filt = np.zeros((q, T + 1))
    P_filt = np.zeros((q, q, T + 1))
    x_pred = np.zeros((q, T + 1))
    P_pred = np.zeros((q, q, T + 1))
    x_filt[:, 0], P_filt[:, :, 0] = prob.x0, prob.Sigma0
    llf_no_const, n_obs = 0.0, 0
    for t in range(T):
        x = Fm @ x_filt[:, t]
        P = Fm @ P_filt[:, :, t] @ Fm.T + prob.Q
        x_pred[:, t + 1], P_pred[:, :, t + 1] = x, P
        obs = ~np.isnan(prob.y[:, t])
        if obs.any():
            Ho = prob.H[obs]
            mean = Ho @ x + prob.Xbeta[obs, :, t] @ prob.beta
            S = Ho @ P @ Ho.T + np.diag(prob.R[obs])
            e = prob.y[obs, t] - mean
            llf_no_const += -0.5 * (np.linalg.slogdet(S)[1] + e @ np.linalg.solve(S, e))
            n_obs += int(obs.sum())
            K = np.linalg.solve(S, Ho @ P).T  # P Ho' S^-1 (S, P symmetric)
            x = x + K @ e
            P = P - K @ Ho @ P
        x_filt[:, t + 1], P_filt[:, :, t + 1] = x, P
    return dict(
        x_filt=x_filt,
        P_filt=P_filt,
        x_pred=x_pred,
        P_pred=P_pred,
        llf_no_const=llf_no_const,
        llf=llf_no_const - 0.5 * n_obs * np.log(2 * np.pi),
        n_obs=n_obs,
    )


def rts_smoother(prob, filt=None):
    """Dense Rauch-Tung-Striebel smoother.

    Returns x_smooth (q, T+1), P_smooth (q, q, T+1) and the lag-one smoothed
    covariances P_lag (q, q, T+1) with P_lag[:, :, t] = Cov(x_t, x_{t-1} | y)
    for t = 1..T (index 0 unused), computed as P_{t|T} J_{t-1}' (de Jong 1989).
    """
    filt = kalman_filter(prob) if filt is None else filt
    q, T = prob.q, prob.T
    Fm = np.diag(prob.F)
    xs = np.zeros((q, T + 1))
    Ps = np.zeros((q, q, T + 1))
    P_lag = np.zeros((q, q, T + 1))
    xs[:, T], Ps[:, :, T] = filt["x_filt"][:, T], filt["P_filt"][:, :, T]
    J = [None] * T
    for t in range(T - 1, -1, -1):
        J[t] = filt["P_filt"][:, :, t] @ Fm.T @ np.linalg.inv(filt["P_pred"][:, :, t + 1])
        xs[:, t] = filt["x_filt"][:, t] + J[t] @ (xs[:, t + 1] - filt["x_pred"][:, t + 1])
        Ps[:, :, t] = filt["P_filt"][:, :, t] + J[t] @ (Ps[:, :, t + 1] - filt["P_pred"][:, :, t + 1]) @ J[t].T
    for t in range(1, T + 1):
        P_lag[:, :, t] = Ps[:, :, t] @ J[t - 1].T
    return dict(x_smooth=xs, P_smooth=Ps, P_lag=P_lag)


def sufficient_statistics(prob, smooth):
    """EM sufficient statistics S11, S10, S00 and y_hat from smoothed moments."""
    xs, Ps, Pl = smooth["x_smooth"], smooth["P_smooth"], smooth["P_lag"]
    T = prob.T
    S11 = sum(np.outer(xs[:, t], xs[:, t]) + Ps[:, :, t] for t in range(1, T + 1))
    S00 = sum(np.outer(xs[:, t - 1], xs[:, t - 1]) + Ps[:, :, t - 1] for t in range(1, T + 1))
    S10 = sum(np.outer(xs[:, t], xs[:, t - 1]) + Pl[:, :, t] for t in range(1, T + 1))
    y_hat = np.einsum("pbt,b->pt", prob.Xbeta, prob.beta) + prob.H @ xs[:, 1:]
    return dict(S11=S11, S10=S10, S00=S00, y_hat=y_hat)
