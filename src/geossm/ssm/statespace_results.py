"""
Convenience container for State Space Model outputs (filter + smoother).
"""

from __future__ import annotations
from typing import Optional, Any, Tuple, Dict

import numpy as np
from scipy.stats import norm
from statsmodels.iolib.summary import Summary
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.stats.stattools import durbin_watson, omni_normtest
from datetime import date
import jax
from geossm.utils import _select_device

ArrayLike = Optional[Any]


class StateSpaceResults:
    """
    Container for state-space model estimation results.
    """

    def __init__(
        self,
        model: Optional[Any],
        y_hat,
        Sigma_y_hat=None,
        tdelta_hat=None,
        # optional metadata
        params=None,
        params_names=None,
        params_dim=None,
        y_obs=None,
        yname=None,
        Xbeta=None,
        backend=None,
        dtype=None,
        # likelihood / info
        llf: Optional[float] = None,
        time_filter: float = 0.0,
        time_smoother: float = 0.0,
        time_expectation: float = 0.0,
        time_total: float = 0.0,
        # main arrays
        x_filtered: ArrayLike = None,
        P_filtered: ArrayLike = None,
        x_pred: ArrayLike = None,
        P_pred: ArrayLike = None,
        K: ArrayLike = None,
        x_smoothed: ArrayLike = None,
        P_smoothed: ArrayLike = None,
        P_pred_smoothed: ArrayLike = None,
        # sufficient statistics
        S11: ArrayLike = None,
        S10: ArrayLike = None,
        S00: ArrayLike = None,
    ):

        # ---- Update the metadata from the model----
        self.model = model

        # ---- Backend: default to the model's backend, so any JAX
        # computation performed on the results (e.g. the Hessian in
        # LRStateSpaceResults) runs on the same device the model was fit
        # on, unless a different backend is explicitly requested here.
        if backend is not None:
            self._backend = _select_device(backend)
        elif model is not None and getattr(model, "backend", None) is not None:
            self._backend = _select_device(model.backend)
        else:
            self._backend = _select_device("auto")

        # ---- dtype: default to the model's dtype, so results stay in the
        # same precision the model was fit in (e.g. the Hessian computed in
        # LRStateSpaceResults), unless a different dtype is explicitly
        # requested here.
        if dtype is not None:
            self._dtype = np.dtype(dtype)
        elif model is not None and getattr(model, "dtype", None) is not None:
            self._dtype = np.dtype(model.dtype)
        else:
            self._dtype = np.dtype(np.float32)

        # get the observed data from the model if available
        if model is not None:
            self.params = getattr(model, "params", None)
            self.params_names = getattr(model, "params_names", None)
            self.params_dim = getattr(model, "params_dim", None)
            self.yname = getattr(model, "yname", None)
            self.y_obs = getattr(model, "y_t", None)  # observed data (from model)
            self.Xbeta = getattr(model, "Xbeta", None)

        if y_obs is not None:
            self.y_obs = y_obs
        if yname is not None:
            self.yname = yname
        if Xbeta is not None:
            self.Xbeta = Xbeta
        if params is not None:
            self.params = params
        if params_names is not None:
            self.params_names = params_names
        if params_dim is not None:
            self.params_dim = params_dim

        # ---- likelihood ----
        self.llf = llf
        self.time_filter = time_filter
        self.time_smoother = time_smoother
        self.time_expectation = time_expectation
        self.time_total = self.time_filter + self.time_smoother + self.time_expectation

        # ---- arrays ----
        self.y_hat = y_hat
        self.Sigma_y_hat = Sigma_y_hat
        self.tdelta_hat = tdelta_hat
        self.x_filtered = x_filtered
        self.P_filtered = P_filtered
        self.x_pred = x_pred
        self.P_pred = P_pred
        self.K = K
        self.x_smoothed = x_smoothed
        self.P_smoothed = P_smoothed
        self.P_pred_smoothed = P_pred_smoothed

        # ---- sufficient stats ----
        self.S11 = S11
        self.S10 = S10
        self.S00 = S00

        # ---- metadata ----
        self.today = date.today()

        # ---- internal cache ----
        self._residuals: Optional[np.ndarray] = None

        # convert arrays if needed
        self.to_numpy()

    @property
    def backend(self):
        return self._backend

    @property
    def dtype(self):
        return self._dtype

    # Utility method to convert array-like inputs to numpy arrays

    def to_numpy(self):
        """
        Convert array-like attributes to numpy arrays in the results' dtype
        (the model's dtype, by default), so results stay numerically
        consistent with the precision the model was fit in.
        """
        for attr in [
            "y_hat",
            "Sigma_y_hat",
            "x_filtered",
            "P_filtered",
            "x_pred",
            "P_pred",
            "K",
            "x_smoothed",
            "P_smoothed",
            "P_pred_smoothed",
            "S11",
            "S10",
            "S00",
        ]:
            value = getattr(self, attr)
            if value is not None:
                setattr(self, attr, np.asarray(value, dtype=self.dtype))

    def update(self, **kwargs) -> "StateSpaceResults":
        """
        Return a NEW StateSpaceResults with updated fields.
        """

        # Collect all current attributes (excluding private ones if desired)
        current_data = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["today"] and not k.startswith("_")
        }

        # Validate keys
        for key in kwargs:
            if key not in current_data:
                raise AttributeError(
                    f"{key} is not a valid attribute of {self.__class__.__name__}"
                )

        # Update values
        current_data.update(kwargs)

        # Create new instance
        return self.__class__(**current_data)

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
            return float(np.mean(valid**2)) if valid.size else float("nan")
        if which == "space":
            # average across time axis -> shape (p,)
            return np.nanmean(err**2, axis=1)
        if which == "time":
            # average across space axis -> shape (T,)
            return np.nanmean(err**2, axis=0)
        raise ValueError("which must be one of {'global','space','time'}")

    def rmse(self, which: str = "global") -> float:
        v = self.mse(which=which)
        if isinstance(v, np.ndarray):
            return np.sqrt(v)
        return float(np.sqrt(v))

    # ---------- Confidence intervals ----------
    def conf_int_state(
        self, which: str = "smoothed", alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) CI arrays for states.
        which: 'smoothed'|'filtered' -> uses P_smoothed or P_filtered and corresponding means.
        """
        alpha = float(alpha)
        z = norm.ppf(1 - alpha / 2.0)
        if which == "smoothed":
            mean = self.x_smoothed
            cov = self.P_smoothed
        elif which == "filtered":
            mean = self.x_filtered
            cov = self.P_filtered
        else:
            raise ValueError("which must be 'smoothed' or 'filtered'")

        if mean is None or cov is None:
            raise ValueError("State means or covariances are not available.")

        lower = np.zeros_like(mean)
        upper = np.zeros_like(mean)
        for t in range(mean.shape[1]):
            std = np.sqrt(np.diag(cov[:, :, t]))
            lower[:, t] = mean[:, t] - z * std
            upper[:, t] = mean[:, t] + z * std

        return lower, upper

    def conf_int_y(
        self, alpha: float = 0.05, prediction: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Confidence intervals for y_hat. If prediction True include measurement noise R when available."""
        
        alpha = float(alpha)
        z = norm.ppf(1 - alpha / 2.0)
        y_hat = self.y_hat
        if y_hat is None:
            raise ValueError("y_hat not available.")

        p, T = y_hat.shape
        model = getattr(self, "model", None)

        # Sigma_y_hat is snapshotted at fit time by the concrete model
        # subclass (e.g. LRStateSpaceResults), not reconstructed here from
        # the model's H/P_smoothed: `model.H` is mutated in place by every
        # `fit()` call/iteration, so rebuilding from it here could silently
        # use a stale H that no longer matches this particular results
        # object.
        Sigma_y_hat = getattr(self, "Sigma_y_hat", None)
        if Sigma_y_hat is None:
            raise ValueError(
                "Cannot compute y confidence intervals: Sigma_y_hat is not "
                "available on this results object."
            )

        R = np.asarray(getattr(model, "R", 0.0)) if model is not None else np.asarray(0.0)
        R_diag_matrix = np.diag(R) if (prediction and R.ndim > 0) else None

        lower = np.zeros((p, T), dtype=self.dtype)
        upper = np.zeros((p, T), dtype=self.dtype)
        for t in range(T):
            var_y = np.asarray(Sigma_y_hat[:, :, t])
            if prediction and R_diag_matrix is not None:
                var_y = var_y + R_diag_matrix

            std_t = np.sqrt(np.diag(var_y))
            lower[:, t] = y_hat[:, t] - z * std_t
            upper[:, t] = y_hat[:, t] + z * std_t
        return lower, upper
        
    def coverage_probability(self, alpha: float = 0.05, which="global"):
        return self._coverage_probability(alpha, which)

    def _coverage_probability(self, alpha: float = 0.05, which="global"):
        """
        Compute empirical coverage probability of prediction intervals.

        -------
        float
            Coverage probability in [0, 1].
        """
        y_true = self.y_obs

        lower, upper = self.conf_int_y(alpha, prediction=True)

        inside = (y_true >= lower) & (y_true <= upper)

        if which == "global":
            return np.nanmean(inside)
        elif which == "space":
            return np.nanmean(inside, axis=1)
        elif which == "time":
            return np.nanmean(inside, axis=0)
        else:
            raise ValueError("which must be 'smoothed' or 'filtered'")

    @staticmethod
    def _format_runtime_summary(parts) -> Tuple[str, str]:
        """
        Build a single "Runtime (a / b / ...): total (va / vb / ...)" summary
        entry from labeled runtime components (name, seconds). The total is
        always the sum of the given parts, so it can never drift out of
        sync with what's shown next to it.
        """
        label = "Runtime (" + " / ".join(name for name, _ in parts) + "):"
        total = sum(value for _, value in parts)
        rendered = f"{total:.3g} (" + " / ".join(f"{value:.3g}" for _, value in parts) + ")"
        return label, rendered

    # ---------- Diagnostics & summary ----------
    def diagnostics(self) -> Dict[str, float]:
        """Return a small diagnostics dict computed on residuals (numpy)."""

        err = getattr(self, "residuals", None)
        if err is None:
            err = self._compute_residuals()

        flat = err.flatten()
        flat = flat[~np.isnan(flat)]
        jb, jbpv = jarque_bera(flat, nan_policy="omit")

        sk = skew(flat, nan_policy="omit")
        kt = kurtosis(flat, nan_policy="omit")

        omni, omnipv = omni_normtest(flat)

        dw = durbin_watson(flat)
        return {
            "jb": float(jb),
            "jb_pvalue": float(jbpv),
            "omni": float(omni),
            "omni_pvalue": float(omnipv),
            "dw": float(dw),
            "skew": float(sk),
            "kurtosis": float(kt),
        }


    def generate_summary(self):

        self.nobs = self.y_obs.size
        self.nspace, self.ntime = self.y_obs.shape
        self.missing = np.sum(np.isnan(self.y_obs))
        
        # Compute residual diagnostics
        err = getattr(self, "residuals", None)
        if err is None:
            err = self._compute_residuals()

        stats = self.diagnostics()


        # top-left / top-right small tables
        p, q, T = self.model.shape if hasattr(self.model, "shape") else ("N/A", "N/A", "N/A")

        runtime_label, runtime_value = self._format_runtime_summary(
            [
                ("fl.", self.time_filter),
                ("sm.", self.time_smoother),
                ("ex.", self.time_expectation),
            ]
        )

        # Left: identity/config of the model and how it was run.
        top_left = dict(
            [
                ("Model name:", lambda: [self.model.__class__.__name__]),
                ("Model type (order):",
                    lambda: [
                        f"{self.model.type if hasattr(self.model, 'type') else 'N/A'}, {self.model.order if hasattr(self.model, 'order') else 'N/A'}"
                    ],
                ),
                ("Dep. Variables:", lambda: [self.model.y_name if hasattr(self.model, "y_name") else "N/A"]),
                ("Shape (p, q, T) :", lambda: [f"(p = {p}, q = {q}, T = {T})"]),
                ("# missing:", lambda: [self.missing]),
                ("Model backend:", lambda: [f"{self.model.backend}, (dtype {self.model.dtype})"]),
                ("Log-Likelihood:", lambda: ["%#8.5g" % self.llf]),
                (runtime_label, lambda v=runtime_value: [v]),
            ]
        )

        # Right: fit quality and residual diagnostics. Bracketed "(x*)"
        # values are the reference value expected for normal, uncorrelated
        # residuals -- not achievable/meaningful targets for MSE/RMSE, so
        # those are left unannotated.
        alpha_cov = 0.05
        top_right = dict(
            [
                ("MSE:", lambda: [f"{self.mse():.2f}"]),
                ("RMSE:", lambda: [f"{self.rmse():.2f}"]),
                ("Coverage Prob.:", lambda a=alpha_cov: [f"{self._coverage_probability(a):.2f} ({1 - a:.2f}*, alpha = {a})"]),
                ("Jarque-Bera:", lambda: [f"{stats['jb']:.2f} (0*) (pvalue: {stats['jb_pvalue']:.2f})"]),
                ("Omnibus test:", lambda: [f"{stats['omni']:.2f} (0*) (pvalue: {stats['omni_pvalue']:.2f})"]),
                ("Skewness:", lambda: [f"{stats['skew']:.2f} (0*)"]),
                ("Kurtosis (excess):", lambda: [f"{stats['kurtosis']:.2f} (0*)"]),
                ("Durbin-Watson:", lambda: [f"{stats['dw']:.2f} (2*)"]),
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

    #: Footnote explaining the "(x*)" reference values shown next to the
    #: residual diagnostics in `generate_summary()`.
    _DIAGNOSTICS_NOTE = (
        "(x*) reference value expected for normal, uncorrelated residuals "
        "(or, for Coverage Prob., the nominal 1 - alpha)."
    )

    def summary(self) -> Summary:
        """Return a statsmodels Summary object with a brief report."""
        # Ensure numpy arrays for summary stats

        self.results = np.array([0])

        gen_top_left, gen_top_right = self.generate_summary()


        # Generate the summary
        smry = Summary()
        smry.add_table_2cols(
            self,
            title="State Space Model results",
            gleft=gen_top_left,
            gright=gen_top_right,
            yname=None,
            xname=None,
        )
        smry.add_extra_txt([self._DIAGNOSTICS_NOTE])

        return smry

    def __str__(self):
        return str(self.summary())

    def __repr__(self):
        return str(self.summary())

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain dict with main results converted to NumPy where possible."""
        self.to_numpy()
        return {
            "y_obs": self.y_obs,
            "y_hat": self.y_hat,
            "Sigma_y_hat": self.Sigma_y_hat,
            "x_filtered": self.x_filtered,
            "P_filtered": self.P_filtered,
            "x_smoothed": self.x_smoothed,
            "P_smoothed": self.P_smoothed,
            "llf": self.llf,
            "mse": self.mse("global"),
            "rmse": self.rmse("global"),
            "diagnostics": self.diagnostics(),
            "S11": self.S11,
            "S10": self.S10,
            "S00": self.S00,
        }
