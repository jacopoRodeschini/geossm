"""
Convenience container for State Space Model outputs (filter + smoother).

This module provides `SSMResults`, a lightweight, user-friendly object
that stores outputs from `StateSpaceModel.filter` and
`StateSpaceModel.smoother` and provides helpers to convert to NumPy
and adapt to a minimal statsmodels-like interface.
"""
from typing import Optional
import numpy as np
import jax.numpy as jnp
from statsmodels.stats.stattools import jarque_bera, durbin_watson, omni_normtest
from statsmodels.iolib.summary import Summary
try:
    from statsmodels.tsa.statespace.mlemodel import MLEResults
except Exception:
    # Provide a minimal fallback so the module can be imported where
    # statsmodels is not installed. The fallback accepts the same
    # constructor signature used below (model, params).
    class MLEResults(object):
        def __init__(self, model=None, params=None):
            pass
from scipy.stats import t, f, norm


class SSMResults(MLEResults):
    """Container for filter + smoother outputs.

    Attributes (kept as JAX arrays until `.to_numpy()` is called):
    - `x_filtered`, `P_filtered`: filtered state estimates (including t=0)
    - `x_pred`, `P_pred`: one-step ahead predictions
    - `K`: last Kalman gain
    - `x_smoothed`, `P_smoothed`, `P_lag`: smoothed states, covariances, lag-one covariances
    - `loglik`: log-likelihood (scalar)
    - `time_filter`, `time_smoother`: timings in seconds
    - `model`: reference to the `StateSpaceModel` that produced the results

    Methods:
    - `to_numpy()`: returns a copy of stored arrays as NumPy arrays.
    - `summary()`: brief human-readable summary.
    - `as_statsmodels()`: lightweight adapter object with a few familiar attributes.
    """

    # Compatibility checklist: names of methods/stored attributes commonly
    # expected by `statsmodels.tsa.statespace.mlemodel.MLEResults`.
    # Implementations below provide safe placeholders; fill in later
    # with statistically appropriate estimators (Fisher information, etc.).
    compat_methods = [
        'params', 'bse', 'cov_params', 'cov_params_default', 'conf_int',
        'predict', 'get_prediction', 'llf', 'aic', 'bic', 'mle_retvals',
        'model', 'summary'
    ]

    def __init__(
        self,
        model,
        y_obs: jnp.ndarray,
        x_filtered: jnp.ndarray,
        P_filtered: jnp.ndarray,
        K: jnp.ndarray,
        x_pred: jnp.ndarray,
        P_pred: jnp.ndarray,
        invP_pred: jnp.ndarray,
        loglik: jnp.ndarray,
        time_filter: float,
        x_smoothed: jnp.ndarray,
        P_smoothed: jnp.ndarray,
        P_lag: jnp.ndarray,
        time_smoother: float,
        y_hat: jnp.ndarray,
        S11: jnp.ndarray,
        S10: jnp.ndarray,
        S00: jnp.ndarray,
        time_expected: float,   
    ):
        # Initialize parent MLEResults with a params vector if available.
        # Infer params from model.beta when possible; otherwise pass an empty array.
        try:
            params = np.ravel(np.array(getattr(model, 'beta', np.array([]))))
        except Exception:
            params = np.array([])

        # Call the parent initializer (MLEResults) with model and params.
        try:
            super(SSMResults, self).__init__(model, params)
        except TypeError:
            # fallback if parent __init__ has a different signature
            try:
                MLEResults.__init__(self, model, params)
            except Exception:
                pass

        # Ensure .params exists for compatibility with statsmodels
        self.params = params
        self.model = model
        # Placeholder for parameter covariance (numpy array)
        self.param_cov = None
        # Optional storage for parameter std errors if computed elsewhere
        self.beta_std = None
        self.y_obs = y_obs

        # Keep JAX arrays by default for downstream JAX usage
        self.x_filtered = x_filtered
        self.P_filtered = P_filtered
        self.x_pred = x_pred
        self.P_pred = P_pred
        self.invP_pred = invP_pred
        self.K = K

        self.x_smoothed = x_smoothed
        self.P_smoothed = P_smoothed
        self.P_lag = P_lag

        # Log-likelihood as scalar (JAX type)
        self.loglik = loglik

        # Expected values
        self.y_hat = y_hat
        self.S11 = S11
        self.S10 = S10
        self.S00 = S00
        
        # timings (floats)
        self.time_filter = float(time_filter)
        self.time_smoother = float(time_smoother)
        self.time_expected = float(time_expected)

     


    def to_numpy(self):
        """Return a dict with NumPy copies of the stored arrays.

        Useful for serialization, plotting, or interop with non-JAX libs.
        """
        return {
            "x_filtered": np.array(self.x_filtered),
            "P_filtered": np.array(self.P_filtered),
            "x_pred": np.array(self.x_pred),
            "P_pred": np.array(self.P_pred),
            "invP_pred": np.array(self.invP_pred),
            "K": np.array(self.K),
            "x_smoothed": np.array(self.x_smoothed),
            "P_smoothed": np.array(self.P_smoothed),
            "P_lag": np.array(self.P_lag),
            "loglik": np.array(self.loglik).item(),
            "time_filter": self.time_filter,
            "time_smoother": self.time_smoother,
        }

    def summary(self, print_output: bool = True) -> Optional[str]:
        """Return or print a short summary with key shapes and metrics."""
        shp = {
            "x_filtered": tuple(self.x_filtered.shape),
            "P_filtered": tuple(self.P_filtered.shape),
            "x_smoothed": tuple(self.x_smoothed.shape),
            "P_smoothed": tuple(self.P_smoothed.shape),
        }
        s = (
            f"SSMResults summary:\n"
            f"  model: {self.model.__class__.__name__}\n"
            f"  loglik: {float(self.loglik):.6g}\n"
            f"  time_filter: {self.time_filter:.4f}s, time_smoother: {self.time_smoother:.4f}s\n"
            f"  shapes: {shp}\n"
        )
        if print_output:
            print(s)
            return None
        return s

    def as_statsmodels(self):
        """Return a lightweight adapter object with a few attributes familiar from statsmodels.

        This is NOT a full drop-in replacement for statsmodels' results classes, but it
        provides `.smoothed_state`, `.filtered_state`, `.smoothed_state_cov`, and `.llf`.
        If you need full compatibility, subclassing statsmodels' `MLEModelResults`
        would be necessary and depends on the exact statsmodels version.
        """

        class _Adapter:
            pass

        a = _Adapter()
        a.smoothed_state = np.array(self.x_smoothed)
        # statsmodels uses shape (k_states, nobs)
        a.filtered_state = np.array(self.x_filtered)
        a.smoothed_state_cov = np.array(self.P_smoothed)
        a.filtered_state_cov = np.array(self.P_filtered)
        a.llf = float(self.loglik)
        a.model = getattr(self.model, "__class__", None)
        return a


    # Compute residuals
    def _compute_error(self):
        # Convert to numpy arrays for diagnostics
        if getattr(self, 'y_obs', None) is not None and getattr(self, 'y_hat', None) is not None:
            y_obs = np.array(self.y_obs)
            y_hat = np.array(self.y_hat)
            # broadcast shapes if needed
            try:
                self._err = y_obs - y_hat
            except Exception:
                self._err = (y_obs - y_hat).astype(float)
        else:
            self._err = None

    @property
    def residuals(self):
        """Returns residuals (y_obs - y_hat)."""
        return self._err

    # Confidence intervals on filtered state (x_t) and smoothed state (x_T)
    def compute_IC_x_t(self):
        """Computes confidence intervals for the filtered state x_t based on P_t."""
        return self._compute_conf_interval(self.x_filtered, self.P_filtered)

    def compute_IC_x_T(self):
        """Computes confidence intervals for the smoothed state x_T based on P_T."""
        return self._compute_conf_interval(self.x_smoothed, self.P_smoothed)

    def compute_IC_y_hat(self, alpha=0.05):
        """Computes confidence intervals for the predictions y_hat."""
        y_hat = np.array(self.y_hat)
        z_score = norm.ppf(1 - alpha / 2)

        # Prefer predictive variance from model if available
        if hasattr(self.model, 'H') and getattr(self, 'P_pred', None) is not None:
            H = np.array(self.model.H)
            R = np.array(self.model.R) if hasattr(self.model, 'R') else 0.0
            P_pred = np.array(self.P_pred)
            # P_pred shape: (q, q, T). compute var_y per time and observation dimension
            p = H.shape[0]
            T = P_pred.shape[2]
            y_std = np.zeros((p, T))
            for t in range(T):
                var_y = H @ P_pred[:, :, t] @ H.T
                # add R if shaped correctly
                try:
                    var_y = var_y + R
                except Exception:
                    var_y = var_y + np.diag(R) if np.ndim(R) == 1 else var_y + R
                y_std[:, t] = np.sqrt(np.diag(var_y))
            upper = y_hat + z_score * y_std
            lower = y_hat - z_score * y_std
            return lower, upper

        # Fallback: use residual std
        if self._err is None:
            self._compute_error()
        resid_std = np.nanstd(self._err)
        upper = y_hat + z_score * resid_std
        lower = y_hat - z_score * resid_std
        return lower, upper

    def _compute_conf_interval(self, mean, covariance):
        """Helper function to compute confidence intervals based on mean and covariance."""
        alpha = getattr(self, 'alpha', 0.05)
        z_score = norm.ppf(1 - alpha / 2)
        mean_np = np.array(mean)
        cov_np = np.array(covariance)

        # covariance expected shape (q, q, T) and mean shape (q, T)
        if cov_np.ndim == 3:
            T = cov_np.shape[2]
            q = cov_np.shape[0]
            std_err = np.zeros((q, T))
            for t in range(T):
                std_err[:, t] = np.sqrt(np.diag(cov_np[:, :, t]))
        elif cov_np.ndim == 2:
            std_err = np.sqrt(np.diag(cov_np))[:, None]
        else:
            std_err = np.sqrt(cov_np)

        upper_bound = mean_np + z_score * std_err
        lower_bound = mean_np - z_score * std_err
        return lower_bound, upper_bound

    # Compute RMSE and MSE
    def compute_rmse(self):
        """Computes RMSE (Root Mean Squared Error)."""
        return np.sqrt(self.compute_mse())

    def compute_mse(self):
        """Computes MSE (Mean Squared Error)."""
        if self._err is None:
            self._compute_error()
        valid = self._err[~np.isnan(self._err)]
        return np.mean(valid ** 2)

    # Compute R-squared
    def compute_r2(self):
        """Computes R-squared."""
        y = np.array(self.y_obs)
        y_hat = np.array(self.y_hat)
        # flatten to 1D
        yf = y.flatten()
        yhf = y_hat.flatten()
        mask = ~np.isnan(yf)
        y_clean = yf[mask]
        yh_clean = yhf[mask]
        ss_total = np.sum((y_clean - np.mean(y_clean)) ** 2)
        ss_res = np.sum((y_clean - yh_clean) ** 2)
        self.r2 = 1 - ss_res / ss_total if ss_total != 0 else np.nan
        return self.r2

    # Compute AIC and BIC
    def compute_aic(self):
        """Computes the AIC (Akaike Information Criterion)."""
        llf = self._log_likelihood()
        # number of estimated parameters: try to infer from model
        k = getattr(self, 'n_params', None)
        if k is None and hasattr(self.model, 'beta'):
            k = np.size(np.array(self.model.beta))
        k = int(k) if k is not None else 0
        self.aic = 2 * k - 2 * llf
        return self.aic

    def compute_bic(self):
        """Computes the BIC (Bayesian Information Criterion)."""
        # number observations: count non-nan residuals
        if self._err is None:
            self._compute_error()
        N = np.sum(~np.isnan(self._err))
        llf = self._log_likelihood()
        k = getattr(self, 'n_params', None)
        if k is None and hasattr(self.model, 'beta'):
            k = np.size(np.array(self.model.beta))
        k = int(k) if k is not None else 0
        self.bic = np.log(N) * k - 2 * llf
        return self.bic

    # Helper function to compute log-likelihood
    def _log_likelihood(self):
        """Computes the log-likelihood based on residuals."""
        if self._err is None:
            self._compute_error()
        resid = self._err[~np.isnan(self._err)]
        N = resid.size
        if N == 0:
            return np.nan
        # use ML estimate of variance (ddof=0)
        sigma2 = np.sum(resid ** 2) / N
        llf = -0.5 * N * (np.log(2 * np.pi * sigma2) + 1)
        return float(llf)

    def _computeFitPerformance(self):
        # Compute the performance metrics using numpy arrays
        y = np.array(self.y_obs)
        y_hat = np.array(self.y_hat)
        y_flat = y.flatten()
        yh_flat = y_hat.flatten()
        mask = ~np.isnan(y_flat)
        y_clean = y_flat[mask]
        yh_clean = yh_flat[mask]

        self.sst = np.sum((y_clean - np.mean(y_clean)) ** 2)
        self.sse = np.sum((y_clean - yh_clean) ** 2)
        self.ssr = np.sum((yh_clean - np.mean(y_clean)) ** 2)

        # Degrees of freedom: try to infer df_model/df_resid if available
        df_model = getattr(self, 'df_model', None)
        df_resid = getattr(self, 'df_resid', None)
        if df_resid is None:
            # approximate: N - k
            k = getattr(self, 'n_params', None)
            if k is None and hasattr(self.model, 'beta'):
                k = np.size(np.array(self.model.beta))
            N = y_clean.size
            df_resid = max(N - int(k or 0), 1)
        self.df_model = int(df_model) if df_model is not None else max(1, int(getattr(self, 'n_params', 1)))
        self.df_resid = int(df_resid)

        # Estimate error variance
        self.s2 = self.sse / max(self.df_resid, 1)

        # Fit performance
        self.r2 = 1 - self.sse / self.sst if self.sst != 0 else np.nan
        self.r2_adj = 1 - (1 - self.r2) * (self.df_resid / max(self.df_resid - 1, 1))
        self.mse = np.mean((yh_clean - y_clean) ** 2)
        self.rmse = np.sqrt(self.mse)

        # F-statistic and p-value (overall model)
        try:
            self.fvalues = (self.ssr / max(self.df_model, 1)) / (self.sse / max(self.df_resid, 1))
            self.f_pvalues = 1 - f.cdf(self.fvalues, self.df_model, self.df_resid)
            self.f_conf_int = f.interval(1 - getattr(self, 'alpha', 0.05), self.df_model, self.df_resid)
        except Exception:
            self.fvalues = np.nan
            self.f_pvalues = np.nan
            self.f_conf_int = (np.nan, np.nan)

        # t-values and p-values for beta if available
        if hasattr(self, 'model') and hasattr(self.model, 'beta'):
            beta = np.array(self.model.beta)
            if hasattr(self, 'beta_std'):
                beta_std = np.array(self.beta_std)
            elif hasattr(self, '_beta_std'):
                beta_std = np.array(self._beta_std)
            else:
                beta_std = np.ones_like(beta)
            self.tvalues = beta / beta_std
            self.pvalues = 2 * (1 - t.cdf(np.abs(self.tvalues), self.df_resid))

    # ---- Statsmodels-compatible placeholders ----
    def cov_params(self, default: bool = False):
        """Return covariance matrix of parameter estimates.

        Placeholder: returns `self.param_cov` if present, otherwise an
        identity matrix scaled with NaNs so it's obvious it's unimplemented.
        """
        if getattr(self, 'param_cov', None) is not None:
            return np.array(self.param_cov)
        # fallback: return nan-filled matrix sized to params
        k = np.size(self.params) if getattr(self, 'params', None) is not None else 0
        if k == 0:
            return np.array([[]])
        return np.full((k, k), np.nan)

    def cov_params_default(self):
        """Alias for the default covariance estimator (same placeholder)."""
        return self.cov_params(default=True)

    @property
    def bse(self):
        """Standard errors for parameters (sqrt of diagonal of cov_params).

        Returns nan values if cov_params is not implemented.
        """
        cov = self.cov_params()
        if cov.size == 0:
            return np.array([])
        diag = np.diag(cov)
        # guard against negative/NaN entries
        with np.errstate(invalid='ignore'):
            se = np.sqrt(diag)
        return se

    def conf_int(self, alpha=0.05, cols=None):
        """Confidence intervals for parameters using normal approximation.

        Placeholder: uses `self.params` and `self.bse`.
        """
        params = np.asarray(self.params)
        se = np.asarray(self.bse)
        if params.size == 0:
            return np.array([[]])
        z = norm.ppf(1 - alpha / 2)
        lower = params - z * se
        upper = params + z * se
        return np.vstack([lower, upper]).T

    def predict(self, start=None, end=None, exog=None, dynamic=False):
        """Placeholder predict method. Implement forecasting using model state here.

        For now raises NotImplementedError to indicate it's intentionally empty.
        """
        raise NotImplementedError("predict: implementation depends on desired forecast semantics")

    def get_prediction(self, start=None, end=None, exog=None, dynamic=False):
        """Placeholder to return prediction results object (mean, se, conf_int).

        Implement later; this is provided so code that calls `.get_prediction`
        can be adapted to this class.
        """
        raise NotImplementedError("get_prediction: implement to return a prediction results object")

    @property
    def mle_retvals(self):
        """Optional storage for mle return values (optimisation info).

        Kept for compatibility with statsmodels' API.
        """
        return getattr(self, '_mle_retvals', None)

    def summary(self):
        """
        Summarize the Regression Results.

        """

        if self._err is None:
            self._compute_error()
        jb, jbpv, skew, kurtosis = jarque_bera(self._err)
        omni, omnipv = omni_normtest(self._err)

        # TODO: Avoid adding attributes in non-__init__
        self.diagn = dict(jb=jb, jbpv=jbpv, skew=skew, kurtosis=kurtosis,
                          omni=omni, omnipv=omnipv)

        top_left = [('Dep. Variable:', getattr(self, 'y_name', (None,))[0]),
                ('Model:', self.model.__class__.__name__ if hasattr(self, 'model') else None),
                ('Method:', [getattr(self, 'method', None)]),
                ('Date:', None),
                ('Time:', None),
                ('No. Observations:', np.sum(~np.isnan(self._err)) if self._err is not None else None),
                ('N. beta:', [getattr(self, 'n_params', np.size(np.array(getattr(self.model, 'beta', [()]))) )]),
                ('Df Residuals:', getattr(self, 'df_resid', None)),
                ('Df Model:', getattr(self, 'df_model', None))
                ]

        if hasattr(self, 'fCovariance'):
            top_left.append(('Covariance Type:', [self.fCovariance.name]))

        k_constant = getattr(self, 'grid', None)
        k_constant = getattr(k_constant, 'intercept', True) if k_constant is not None else True

        rsquared_type = '' if k_constant else ' (uncentered)'
        top_right = [('R-squared' + rsquared_type + ':',
                  ["%#8.3f" % getattr(self, 'r2', np.nan)]),
                 ('Adj. R-squared' + rsquared_type + ':',
                  ["%#8.3f" % getattr(self, 'r2_adj', np.nan)]),
                 ('MSE' + ':', ["%#8.3f" % getattr(self, 'mse', np.nan)]),
                 ('RMSE' + ':', ["%#8.3f" % getattr(self, 'rmse', np.nan)]),
                 ('Sigma^2_e' + ':', ["%#8.3f" % getattr(self, 's2', np.nan)]),
                 ('F-statistic:', ["%#8.4g" % getattr(self, 'fvalues', np.nan)]),
                 ('Prob (F-statistic):', ["%#6.3g" % getattr(self, 'f_pvalues', np.nan)]),
                 ('Log-Likelihood:', ["%#8.4g" % getattr(self, 'loglik', np.nan)]),
                 ('AIC:', ["%#8.4g" % getattr(self, 'aic', np.nan)]),
                 ('BIC:', ["%#8.4g" % getattr(self, 'bic', np.nan)])
                 ]

        diagn_left = [('Omnibus:', ["%#6.3f" % omni]),
                      ('Prob(Omnibus):', ["%#6.3f" % omnipv]),
                      ('Skew:', ["%#6.3f" % skew]),
                      ('Kurtosis:', ["%#6.3f" % kurtosis])
                      ]

        diagn_right = [('Durbin-Watson:',
                        ["%#8.3f" % durbin_watson(self._err)]
                        ),
                       ('Jarque-Bera (JB):', ["%#8.3f" % jb]),
                       ('Prob(JB):', ["%#8.3g" % jbpv]),
                       ('Cond. No.', ["%#8.3g" % getattr(self, 'condno', np.nan)])
                       ]

        title = "Regression Results"

        # create summary table instance

        smry = Summary()
        yname_safe = getattr(self, 'y_name', (None,))[0]
        xname_safe = getattr(self, 'beta_name', None)
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                     yname=yname_safe, xname=xname_safe, title=title)

        # Add grid & covariance type summary
        # smry.add_table_2cols(self, gleft=top_left, gright=top_right,
        #                      yname=yname, xname=xname,title=title)

        smry.add_table_params(self, yname=yname_safe, xname=xname_safe, alpha=getattr(self, 'alpha', 0.05),
                      use_t=getattr(self, 'use_t', True))

        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                     yname=yname_safe, xname=xname_safe,
                     title="")

        # add warnings/notes, added to text format only
        """
        etext = []
        if not self.k_constant:
            etext.append(
                "R² is computed without centering (uncentered) since the "
                "model does not contain a constant."
            )
        if hasattr(self, 'cov_type'):
            etext.append(self.cov_kwds['description'])
        if self.model.exog.shape[0] < self.model.exog.shape[1]:
            wstr = "The input rank is higher than the number of observations."
            etext.append(wstr)
        if eigvals[-1] < 1e-10:
            wstr = "The smallest eigenvalue is %6.3g. This might indicate "
            wstr += "that there are\n"
            wstr += "strong multicollinearity problems or that the design "
            wstr += "matrix is singular."
            wstr = wstr % eigvals[-1]
            etext.append(wstr)
        elif condno > 1000:  # TODO: what is recommended?
            wstr = "The condition number is large, %6.3g. This might "
            wstr += "indicate that there are\n"
            wstr += "strong multicollinearity or other numerical "
            wstr += "problems."
            wstr = wstr % condno
            etext.append(wstr)

        if etext:
            etext = ["[{0}] {1}".format(i + 1, text)
                     for i, text in enumerate(etext)]
            etext.insert(0, "Notes:")
            smry.add_extra_txt(etext)
        """

        return smry

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.__str__()
# End of src/ssm_results.py