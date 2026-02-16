"""
Convenience container for State Space Model outputs (filter + smoother).

Improved SSMResults: safer defaults, clearer API, lazy conversions between
JAX and NumPy, robust conf-int and diagnostics helpers, and a simple text
summary. Keep this as a lightweight adapter — heavy inference (bootstrap,
Hessian) should live in the fitter.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, Dict

import numpy as np
import jax.numpy as jnp
from scipy.stats import norm
from statsmodels.iolib.summary import Summary
from statsmodels.stats.stattools import jarque_bera, durbin_watson, omni_normtest


ArrayLike = Optional[Any]


@dataclass
class SSMResults:
    # metadata
    model: Optional[Any] = None
    nobs: Optional[int] = None
    param_names: Optional[list] = None

    # main likelihood / info
    llf: Optional[float] = None  # log-likelihood (scalar)
    time_filter: Optional[float] = None
    time_smoother: Optional[float] = None
    time_expectation: Optional[float] = None

    # raw outputs (JAX or NumPy arrays). Keep None default to allow partial results.
    y_obs: ArrayLike = None           # (p, T)
    y_hat: ArrayLike = None           # (p, T), expected observation from smoothed states
    x_filtered: ArrayLike = None      # (q, T+1) including t=0
    P_filtered: ArrayLike = None      # (q, q, T+1)
    x_pred: ArrayLike = None          # (q, T)
    P_pred: ArrayLike = None          # (q, q, T)
    invP_pred: ArrayLike = None
    K: ArrayLike = None               # Kalman gains (maybe last or full history)
    x_smoothed: ArrayLike = None      # (q, T+1)
    P_smoothed: ArrayLike = None      # (q, q, T+1)
    P_lag: ArrayLike = None           # (q, q, T)  lag-one covariances

    # optional sufficient statistics from E-step
    S11: ArrayLike = None
    S10: ArrayLike = None
    S00: ArrayLike = None

    # internal cache
    _residuals: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # ---------- Utility methods ----------
    def _to_numpy(self, arr):
        if arr is None:
            return None
        try:
            # jax DeviceArray -> numpy
            return np.asarray(arr)
        except Exception:
            return arr

    def to_numpy(self) -> "SSMResults":
        """Convert stored arrays to NumPy in-place and return self."""
        for name in ("y_obs", "y_hat", "x_filtered", "P_filtered", "x_pred",
                     "P_pred", "invP_pred", "K", "x_smoothed", "P_smoothed",
                     "P_lag", "S11", "S10", "S00"):
            val = getattr(self, name, None)
            if val is not None:
                setattr(self, name, self._to_numpy(val))
        return self

    # ---------- Residuals / Diagnostics ----------
    def _compute_residuals(self) -> Optional[np.ndarray]:
        """Compute and cache residuals: y_obs - y_hat (converted to numpy)."""
        if self._residuals is not None:
            return self._residuals
        y_obs = self._to_numpy(self.y_obs)
        y_hat = self._to_numpy(self.y_hat)
        if y_obs is None or y_hat is None:
            self._residuals = None
            return None
        try:
            self._residuals = y_obs - y_hat
        except Exception:
            self._residuals = (y_obs - y_hat).astype(float)
        return self._residuals

    @property
    def residuals(self) -> Optional[np.ndarray]:
        return self._compute_residuals()

    # ---------- Error metrics ----------
    def mse(self, kind: str = "global") -> float:
        """Mean squared error. kind in {'global','space','time'}."""
        err = self._compute_residuals()
        if err is None:
            return float("nan")
        if kind == "global":
            valid = err[~np.isnan(err)]
            return float(np.mean(valid ** 2)) if valid.size else float("nan")
        if kind == "space":
            # average across time axis -> shape (p,)
            return np.nanmean(err ** 2, axis=1)
        if kind == "time":
            # average across space axis -> shape (T,)
            return np.nanmean(err ** 2, axis=0)
        raise ValueError("kind must be one of {'global','space','time'}")

    def rmse(self, kind: str = "global") -> float:
        v = self.mse(kind=kind)
        if isinstance(v, np.ndarray):
            return np.sqrt(v)
        return float(np.sqrt(v))

    # ---------- Confidence intervals ----------
    def _std_from_cov(self, cov: np.ndarray) -> np.ndarray:
        """Return standard deviations from covariance array.
        Accepts cov shape (q,q,T) or (q,q) and returns shape (q,T) or (q,1).
        """
        cov = np.asarray(cov)
        if cov.ndim == 3:
            q, _, T = cov.shape
            std = np.zeros((q, T))
            for t in range(T):
                std[:, t] = np.sqrt(np.maximum(np.diag(cov[:, :, t]), 0.0))
            return std
        if cov.ndim == 2:
            return np.sqrt(np.maximum(np.diag(cov), 0.0))[:, None]
        # fallback
        return np.sqrt(np.maximum(cov, 0.0))

    def conf_int_state(self, which: str = "smoothed", alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) CI arrays for states.
        which: 'smoothed'|'filtered' -> uses P_smoothed or P_filtered and corresponding means.
        """
        alpha = float(alpha)
        z = norm.ppf(1 - alpha / 2.0)
        if which == "smoothed":
            mean = self._to_numpy(self.x_smoothed)
            cov = self._to_numpy(self.P_smoothed)
        elif which == "filtered":
            mean = self._to_numpy(self.x_filtered)
            cov = self._to_numpy(self.P_filtered)
        else:
            raise ValueError("which must be 'smoothed' or 'filtered'")

        if mean is None or cov is None:
            raise ValueError("State means or covariances are not available.")

        std = self._std_from_cov(cov)  # (q,T) or (q,1)
        lower = mean - z * std
        upper = mean + z * std
        return lower, upper

    def conf_int_y(self, alpha: float = 0.05, prediction: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Confidence intervals for y_hat. If prediction True include measurement noise R when available."""
        alpha = float(alpha)
        z = norm.ppf(1 - alpha / 2.0)
        y_hat = self._to_numpy(self.y_hat)
        if y_hat is None:
            raise ValueError("y_hat not available.")

        # try using model H and P_smoothed to compute predictive var
        model = getattr(self, "model", None)
        Psm = self._to_numpy(self.P_smoothed)
        if model is not None and hasattr(model, "H") and Psm is not None:
            H = np.asarray(model.H)
            R = np.asarray(getattr(model, "R", 0.0 if not prediction else getattr(model, "R", 0.0)))
            T = Psm.shape[-1]
            p = H.shape[0]
            std = np.zeros((p, T))
            for t in range(T):
                var_y = H @ Psm[:, :, t] @ H.T
                if prediction and getattr(model, "R", None) is not None:
                    var_y = var_y + np.asarray(model.R)
                std[:, t] = np.sqrt(np.maximum(np.diag(var_y), 0.0))
            lower = y_hat - z * std
            upper = y_hat + z * std
            return lower, upper

        # fallback to residual std
        err = self._compute_residuals()
        if err is None:
            raise ValueError("Cannot compute y confidence intervals: no residuals and no model covariance.")
        resid_std = np.nanstd(err)
        lower = y_hat - z * resid_std
        upper = y_hat + z * resid_std
        return lower, upper

    # ---------- Diagnostics & summary ----------
    def diagnostics(self) -> Dict[str, float]:
        """Return a small diagnostics dict computed on residuals (numpy)."""
        err = self._compute_residuals()
        if err is None:
            return {}
        flat = err.flatten()
        flat = flat[~np.isnan(flat)]
        jb, jbpv, skew, kurtosis = jarque_bera(flat, nan_policy="omit")
        omni, omnipv = omni_normtest(flat)
        dw = durbin_watson(flat)
        return {"jb": float(jb), "jb_pvalue": float(jbpv), "omni_pvalue": float(omnipv), "dw": float(dw),
                "skew": float(skew), "kurtosis": float(kurtosis)}

    def summary(self) -> Summary:
        """Return a statsmodels Summary object with a brief report."""
        # Ensure numpy arrays for summary stats
        self.to_numpy()
        err = self._compute_residuals() or np.array([])
        # top-left / top-right small tables
        top_left = [
            ("Model:", getattr(self.model, "__class__", type(self.model)).__name__),
            ("No. Observations:", int(self.nobs) if self.nobs is not None else None),
            ("Log Likelihood:", f"{self.llf:.6g}" if self.llf is not None else None),
            ("Runtime filter (s):", f"{self.time_filter:.4g}" if self.time_filter is not None else None),
            ("Runtime smoother (s):", f"{self.time_smoother:.4g}" if self.time_smoother is not None else None),
        ]
        top_right = [
            ("MSE:", f"{self.mse():.6g}" if err.size else None),
            ("RMSE:", f"{self.rmse():.6g}" if err.size else None),
        ]
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=getattr(self, "y_name", (None,))[0] if getattr(self, "y_name", None) else None,
                             xname=getattr(self, "param_names", None), title="State Space Model Results")
        return smry

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain dict with main results converted to NumPy where possible."""
        self.to_numpy()
        return {
            "y_obs": self.y_obs,
            "y_hat": self.y_hat,
            "x_filtered": self.x_filtered,
            "P_filtered": self.P_filtered,
            "x_smoothed": self.x_smoothed,
            "P_smoothed": self.P_smoothed,
            "P_lag": self.P_lag,
            "llf": self.llf,
            "mse": self.mse("global"),
            "rmse": self.rmse("global"),
            "diagnostics": self.diagnostics(),
            "S11": self.S11, "S10": self.S10, "S00": self.S00
        }