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
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.stats.stattools import durbin_watson, omni_normtest
from datetime import date, datetime



ArrayLike = Optional[Any]


@dataclass
class SSMResults:
    # metadata
    model: Optional[Any]
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

    Xbeta: ArrayLike = None           # (q, q, T)  lag-one covariances
    beta: ArrayLike = None
    xbeta_names: ArrayLike = None

    # optional sufficient statistics from E-step
    S11: ArrayLike = None
    S10: ArrayLike = None
    S00: ArrayLike = None

    today = date.today()

    # internal cache
    _residuals: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # Convert all quantity to numpy array 
    def __post_init__(self):
        self = self.to_numpy()

    # ---------- Utility methods ----------
    def to_numpy(self) -> "SSMResults":
        """Convert stored arrays to NumPy in-place and return self."""
        exclude = {"model"}

        for name in self.__dataclass_fields__:
            if name in exclude:
                continue

            val = getattr(self, name, None)
            if val is not None and isinstance(val, jnp.ndarray):
                setattr(self, name, self._to_numpy(val))
        return self
    

    def _to_numpy(self, arr):
        if arr is None:
            return None
        try:
            # jax DeviceArray -> numpy
            return np.asarray(arr)
        except Exception:
            return arr

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
    def mse(self, which: str = "global") -> float:
        """Mean squared error. kind in {'global','space','time'}."""
        err = self._compute_residuals()
        if which is None:
            return float("nan")
        if which == "global":
            valid = err[~np.isnan(err)]
            return float(np.mean(valid ** 2)) if valid.size else float("nan")
        if which == "space":
            # average across time axis -> shape (p,)
            return np.nanmean(err ** 2, axis=1)
        if which == "time":
            # average across space axis -> shape (T,)
            return np.nanmean(err ** 2, axis=0)
        raise ValueError("which must be one of {'global','space','time'}")

    def rmse(self, which: str = "global") -> float:
        v = self.mse(which=which)
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
    

    def coverage_probability(self, alpha: float = 0.05, which='global'):
        return self._coverage_probability(alpha, which)

    def _coverage_probability(self, alpha: float = 0.05, which='global'):
        """
        Compute empirical coverage probability of prediction intervals.

        -------
        float
            Coverage probability in [0, 1].
        """
        y_true = self.y_obs

        lower, upper = self.conf_int_y(alpha, prediction  = False)
        
        inside = (y_true >= lower) & (y_true <= upper)
        
        
        if which == "global":
            return np.nanmean(inside)
        elif which == "space":
            return np.nanmean(inside, axis = 1)
        elif which == "time":
            return np.nanmean(inside, axis = 0)
        else:
            raise ValueError("which must be 'smoothed' or 'filtered'")


    # ---------- Diagnostics & summary ----------
    def diagnostics(self) -> Dict[str, float]:
        """Return a small diagnostics dict computed on residuals (numpy)."""
        
        err = getattr(self, 'residuals', None)
        if err is None:
            err = self._compute_residuals()
            
        flat = err.flatten()
        flat = flat[~np.isnan(flat)]
        jb, jbpv = jarque_bera(flat, nan_policy="omit")
        
        sk = skew(flat, nan_policy="omit")
        kt = kurtosis(flat, nan_policy="omit")
        
        omni, omnipv = omni_normtest(flat)

        dw = durbin_watson(flat)
        return {"jb": float(jb), "jb_pvalue": float(jbpv), "omni": float(omni), "omni_pvalue": float(omnipv), "dw": float(dw),
                "skew": float(sk), "kurtosis": float(kt)}
    

    def summary(self) -> Summary:
        """Return a statsmodels Summary object with a brief report."""
        # Ensure numpy arrays for summary stats
        
        self.results = np.array([0])
        self.params = self.beta
        self.param_names = self.xbeta_names
        self.bse = np.zeros(len(self.beta))
        self.tvalues = np.zeros(len(self.beta))
        self.pvalues = np.zeros(len(self.beta))

        def conf_int_parmas(alpha=0.05):
            lower = self.params - 1.96 * self.bse
            upper = self.params + 1.96 * self.bse
            return np.column_stack([lower, upper])


        self.conf_int = lambda alpha: conf_int_parmas(alpha)
       
        self.nobs = self.y_obs.size
        self.nspace, self.ntime = self.y_obs.shape
        self.missing = np.sum(np.isnan(self.y_obs))
        self.yname = "prova"

        # self.df_model = str(100)
        # self.df_resid = str(100)
        
        # Compute residual diagnostics
        err = getattr(self, 'residuals', None)
        if err is None:
            err = self._compute_residuals()
        
        stats = self.diagnostics()

        # top-left / top-right small tables
        
        top_left = dict([
            ('Model type:', lambda: [self.model.__class__.__name__]),
            ('Dep. Variable:', lambda: [self.yname]),
            ('Date:', lambda: [self.today]),
            ('Number of Obs:', lambda: [self.nobs]),
            ('Number of points:', lambda: [self.nspace]),
            ('# missing:', lambda: [self.missing]),
            ('Time lenght:', lambda: [self.ntime]),
            ('Log-Likelihood:', lambda: ["%#8.5g" % self.llf]),
            ("Runtime filter (s):", lambda: [f"{self.time_filter:.3g}"]),
            ])
        
        if getattr(self, 'time_smoother', None) is not None:
            top_left["Runtime smoother (s):"] = lambda: [f"{self.time_smoother:.4g}"]

        top_right = {
            'jb:': lambda: f"{stats['jb']:.2f} (pvalue: {stats['jb_pvalue']:.2f})",
            'Omnibus test:': lambda: f"{stats['omni']:.2f} (pvalue: {stats['omni_pvalue']:.2f})",
            'Durbin-Watson:': lambda: f"{stats['dw']:.2f}",
            'Skewness:': lambda: f"{stats['skew']:.2f}",
            'Kurtosis:': lambda: f"{stats['kurtosis']:.2f}",
            'MSE:': lambda: f"{self.mse():.2f}",
            'RMSE:': lambda: f"{self.rmse():.2f}",
            'Coverage Prob.:': lambda: f"{self._coverage_probability():.2f}, (alpha = 0.05)",
        }

        # Generate the dictionaly        
        gen_top_left = []
        for item in top_left.keys():
            gen_top_left.append( (item, list(top_left[item]())))

        gen_top_right = []
        for item in top_right.keys():
            gen_top_right.append( (item, top_right[item]()))
        
        # Generate the summary 
        smry = Summary()
        smry.add_table_2cols(self,title="State Space Model results",
                             gleft = gen_top_left, gright = gen_top_right, yname=None, xname=None)

        # add the model params
        smry.add_table_params(
            self,
            yname=None,
            xname=self.param_names
        )

        return smry
    
    def __str__(self):
        return self.summary()
    
    def __repr__(self):
        return self.summary()

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