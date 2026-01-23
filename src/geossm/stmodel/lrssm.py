"""
Adapter scaffolding making the project's StateSpaceModel usable with
statsmodels' MLEModel API.

This module provides `StateSpaceMLEModel`, a thin adapter that subclasses
`statsmodels.tsa.statespace.mlemodel.MLEModel` (when available) and forwards
likelihood/prediction calls to the project's `StateSpaceModel` computation
engine. All heavy computations remain in the JAX-based `StateSpaceModel`.

The implementation below is a scaffold with detailed placeholders and a
clear TODO list of methods/attributes required for full compatibility. Fill
the TODOs with concrete code (numerical conversions, parameter mapping,
and likelihood evaluation) later.
"""
from typing import Optional
import numpy as np

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

from ssm import StateSpaceModel

# %% Low Rank State-Space Model adapter to statsmodels MLEModel API
class LRStateSpaceModel(StateSpaceModel):
    """Scaffold adapter to expose a statsmodels-style Model API.

    High-level design:
    - This adapter holds (or can construct) a `StateSpaceModel` (the JAX
      computational engine) and maps statsmodels-style parameter vectors to
      the engine's matrices (F, H, Q, R, beta, etc.).
    - Implement `start_params`, `transform_params`, `untransform_params`,
      and `loglike` (the optimizer will call `loglike` during `fit`).

    Important attributes and methods to implement (placeholders provided):
    Required attributes:
    - `endog` : observed data (numpy array, shape (p, T) or similar)
    - `exog`  : optional exogenous regressors used by the model
    - `k_params` : number of parameters
    - `nobs` : number of observations

    Required methods for MLEModel compatibility:
    - `start_params(self)` -> np.ndarray
    - `transform_params(self, params)` -> np.ndarray
    - `untransform_params(self, params)` -> np.ndarray
    - `loglike(self, params)` -> float
    - (optional but useful) `loglikeobs(self, params)` -> array of per-observation loglike
    - `score(self, params)` -> gradient
    - `hessian(self, params)` -> hessian
    - `update(self, params, **kwargs)` -> set internal matrices from params

    Results/prediction interface to implement or delegate:
    - `predict(self, start=None, end=None, exog=None, dynamic=False)`
    - `get_prediction(self, start=None, end=None, exog=None, dynamic=False)`

    Notes on JAX/NumPy interop:
    - statsmodels optimizers expect NumPy arrays and Python-callable
      `loglike` returning float scalars. If you use JAX for likelihood
      evaluation, ensure the JAX value is converted to a Python float (e.g.
      `float(jax_value)`) — and avoid calling jitted functions with JAX
      tracers from the optimizer.

    TODOs / Implementation checklist (for you):
    - Map `params` vector <-> model matrices (decide parameter ordering)
    - Implement `loglike` by: (a) mapping params -> matrices, (b) calling
      the `StateSpaceModel.filter` / `smoother` as needed, (c) returning
      the summed log-likelihood as float.
    - Provide a `start_params` heuristic (e.g., OLS for beta, empirical
      variances for Q/R diagonals).
    - Implement `predict` and `get_prediction` using the model's
      `computeExpectedValues` and state predictions; return numpy arrays.
    - Implement `cov_params` & `bse` in the `SSMResults` class (we added
      placeholders already) or compute the observed information here.
    """

    def __init__(
        self,
        endog: np.ndarray,
        exog: Optional[np.ndarray] = None,
        ss_model=None,
        k_states: Optional[int] = None,
        **kwargs,
    ):
        """Create adapter instance.

        Parameters
        - endog: observed data (numpy array)
        - exog: exogenous regressors (numpy array) or None
        - ss_model: optional existing `StateSpaceModel` instance to wrap.
                    If not provided, the adapter can construct one when
                    `update(params)` is called.
        - k_states: number of latent states (used to build shapes)
        """
        super(StateSpaceMLEModel, self).__init__(endog=endog, exog=exog)
        self.endog = np.asarray(endog)
        self.exog = np.asarray(exog) if exog is not None else None
        self.ss_model = ss_model  # the JAX computation engine (StateSpaceModel)
        self.k_states = k_states

        # bookkeeping
        self.nobs = int(self.endog.size)
        self.k_params = None  # set once you choose a parameterization
        self.param_names = None

    # ---------- required by statsmodels' optimization flow ----------
    def start_params(self) -> np.ndarray:
        """Return a reasonable starting parameter vector.

        TODO: implement heuristics for beta, and variance parameters.
        """
        # Placeholder: return zeros-sized vector (implement properly)
        if self.k_params is None:
            raise NotImplementedError("k_params not set — define parameterization first")
        return np.zeros(self.k_params)

    def transform_params(self, params: np.ndarray) -> np.ndarray:
        """Transform parameters to an unconstrained space for optimization.

        Typical tasks: ensure variances are positive (log transform),
        correlations are mapped from unconstrained reals to admissible range, etc.
        """
        return np.asarray(params)

    def untransform_params(self, params: np.ndarray) -> np.ndarray:
        """Inverse of `transform_params` — map back to natural parameter space."""
        return np.asarray(params)

    def update(self, params: np.ndarray, **kwargs):
        """Set internal model matrices (F, H, Q, R, beta, ...) from `params`.

        This method should not run expensive filtering — just assign matrices
        or update a template `StateSpaceModel` instance so that `loglike`
        can call it.
        """
        # TODO: map `params` into the ss_model matrices. Example mapping:
        # params = [flatten(F), flatten(H), diag(Q), diag(R), beta]
        raise NotImplementedError("update: implement parameter -> matrix mapping")

    def loglike(self, params: np.ndarray) -> float:
        """Return (negative) log-likelihood for optimization.

        Steps to implement:
        - call `untransform_params` to obtain natural parameters
        - `self.update(natural_params)` to apply them into `self.ss_model`
        - run `self.ss_model.filter(y)` to get `loglik`
        - return a Python float (statsmodels expects a float)
        """
        # Defensive placeholder
        params = np.asarray(params)
        natural = self.untransform_params(params)
        # ensure model is present
        if self.ss_model is None:
            raise NotImplementedError("loglike: no internal StateSpaceModel available — provide one or implement update to construct it")

        # TODO: call update(...) to set models' parameters
        # self.update(natural)

        # TODO: call filter and extract log-likelihood (ensure conversion to float)
        # res = self.ss_model.filter(self.endog)
        # llf = res[-2]  # depends on filter return signature
        # return float(llf)
        raise NotImplementedError("loglike: implement likelihood evaluation using StateSpaceModel.filter")

    def loglikeobs(self, params: np.ndarray):
        """Return array of per-observation log-likelihood contributions.

        Optional; useful for some statistics. Implement if needed.
        """
        raise NotImplementedError("loglikeobs: implement if per-observation contributions are needed")

    def score(self, params: np.ndarray):
        """Gradient (score) of the log-likelihood. Optional but useful.

        Can compute via numerical differentiation of `loglike` or derive
        analytically. Statsmodels will compute numerical derivatives if
        this is not provided.
        """
        raise NotImplementedError("score: implement analytic or numeric gradient")

    def hessian(self, params: np.ndarray):
        """Hessian (observed information). Optional — statsmodels can
        approximate it numerically if missing.
        """
        raise NotImplementedError("hessian: implement if you can provide an analytic Hessian")

    # ---------------- prediction helpers ----------------
    def predict(self, start=None, end=None, exog=None, dynamic=False):
        """Compute predicted (fitted) values or forecasts.

        Must return a numpy array of fitted/forecasted `endog` values.
        TODO: use self.ss_model.computeExpectedValues / smoother outputs.
        """
        raise NotImplementedError("predict: implement using the StateSpaceModel prediction pipeline")

    def get_prediction(self, start=None, end=None, exog=None, dynamic=False):
        """Return a prediction results object with mean, se_mean and conf_int.

        Implement a small container or return a tuple; statsmodels has its
        own PredictionResults class — you may mirror that API.
        """
        raise NotImplementedError("get_prediction: implement to return prediction results")

    # ---------------- utilities ----------------
    def _to_jax(self, arr: np.ndarray):
        """Convert numpy array to a JAX array if needed before calling jitted code."""
        try:
            import jax.numpy as jnp

            return jnp.asarray(arr)
        except Exception:
            return arr

    def _from_jax(self, arr):
        """Convert JAX array to NumPy array when returning results to the user."""
        try:
            import numpy as _np

            return _np.array(arr)
        except Exception:
            return arr


__all__ = ["StateSpaceMLEModel"]



def start_params(self) -> np.ndarray:
        """Return a reasonable starting parameter vector.

        TODO: implement heuristics for beta, and variance parameters.
        """
        # Placeholder: return zeros-sized vector (implement properly)
        if self.k_params is None:
            raise NotImplementedError("k_params not set — define parameterization first")
        return np.zeros(self.k_params)



    @jit
    def predict(self, start=None, end=None, exog=None, dynamic=False):
        """Compute predicted (fitted) values or forecasts.

        Must return a numpy array of fitted/forecasted `endog` values.
        TODO: use self.ss_model.computeExpectedValues / smoother outputs.
        """
        raise NotImplementedError(
            "predict: implement using the StateSpaceModel prediction pipeline")
    @jit
    def get_prediction(self, start=None, end=None, exog=None, dynamic=False):
        """Return a prediction results object with mean, se_mean and conf_int.

        Implement a small container or return a tuple; statsmodels has its
        own PredictionResults class — you may mirror that API.
        """
        raise NotImplementedError(
            "get_prediction: implement to return prediction results")

    def loglike(self, y_obs) -> float:
        """Return (negative) log-likelihood for optimization.

        """
        # Defensive placeholder
        params = np.asarray(params)
        natural = self.untransform_params(params)
        # ensure model is present
        if self.ss_model is None:
            raise NotImplementedError(
                "loglike: no internal StateSpaceModel available — provide one or implement update to construct it")

        # TODO: call update(...) to set models' parameters
        # self.update(natural)

        # TODO: call filter and extract log-likelihood (ensure conversion to float)
        # res = self.ss_model.filter(self.endog)
        # llf = res[-2]  # depends on filter return signature
        # return float(llf)
        raise NotImplementedError(
            "loglike: implement likelihood evaluation using StateSpaceModel.filter")

    def loglikeobs(self, params: jnp.ndarray):
        """Return array of per-observation log-likelihood contributions.

        Optional; useful for some statistics. Implement if needed.
        """
        raise NotImplementedError(
            "loglikeobs: implement if per-observation contributions are needed")

    def score(self, params: jnp.ndarray):
        """Gradient (score) of the log-likelihood. Optional but useful.

        Can compute via numerical differentiation of `loglike` or derive
        analytically. Statsmodels will compute numerical derivatives if
        this is not provided.
        """
        raise NotImplementedError(
            "score: implement analytic or numeric gradient")

    def hessian(self, params: jnp.ndarray):
        """Hessian (observed information). Optional — statsmodels can
        approximate it numerically if missing.
        """
        raise NotImplementedError(
            "hessian: implement if you can provide an analytic Hessian")    

