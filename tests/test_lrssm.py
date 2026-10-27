"""End-to-end `LRStateSpaceModel` workflow on a small synthetic dataset (CPU).

One EM fit is shared by the module (about 10 s including JIT compilation).
"""

import dataclasses
import gc
import inspect
import types
import warnings
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import optimize, stats
from shapely.geometry import Polygon

from geossm.covmodel import spdeAppoxCov
from geossm.stmodel import (FitOptions, LRPredictionResults, LRStateSpaceModel, LRStateSpaceResults,
                            ModelParams, Param)
from geossm.stmodel.lrssm import _ks_value_and_grad, _spde_marginal_precision
from geossm.stmodel.stmodel_results import FittedState, _pack_params, _unpack_params
from tests._known_bugs import known_bug
from tests._printing import assert_printing_is_side_effect_free
from tests._reference import SSMProblem, kalman_filter
from tests._synthetic import DOMAIN, make_panel_gdf, make_square_mesh

N_SITES, N_TIMES, MAX_ITER = 25, 40, 5

# Fits deliberately stop at max_iter (tol_relat=1e-12) to keep runtime fixed.
pytestmark = pytest.mark.filterwarnings("ignore:The EM algorithm reached max_iter:RuntimeWarning")


def _build_model(gdf, mesh, dtype=jnp.float32):
    model = LRStateSpaceModel(gdf, ["y ~ 1 + x"], domain=[DOMAIN], verbose=False,
                              backend="cpu", dtype=dtype)
    cov = spdeAppoxCov(latlon=False, verbose=False).setup(mesh, domain=[DOMAIN], stats=False)
    return model.setup([cov])


def _options(**kw):
    kw = {"max_iter": MAX_ITER, "tol_relat": 1e-12, "verbose": False, **kw}
    return FitOptions(**kw)


@pytest.fixture(scope="module")
def gdf():
    return make_panel_gdf(n_sites=N_SITES, n_times=N_TIMES, missing_frac=0.1, seed=0)


@pytest.fixture(scope="module")
def mesh():
    return make_square_mesh()


@pytest.fixture(scope="module")
def stop_model(gdf, mesh):
    return _build_model(gdf, mesh)


@pytest.fixture(scope="module")
def fitted(gdf, mesh):
    model = _build_model(gdf, mesh)
    return model, model.fit(options=_options())


class TestModelConstruction:
    def test_dimensions(self, fitted):
        model, _ = fitted
        p, q, T = model.shape
        assert (p, T) == (N_SITES, N_TIMES)
        assert q == int(model.cov_function[0].fem_solver.n_inner_points)
        assert model.Xbeta.shape == (p, 2, T)

    @known_bug("F11", "formulas=None (documented 'empty model') raises AttributeError",
               raises=AttributeError)
    def test_empty_model_without_formulas(self, gdf):
        LRStateSpaceModel(gdf, None, None, verbose=False, backend="cpu")

    def test_float64_without_x64_raises(self, gdf):
        # F03: the constructor must not switch on the process-wide x64 flag.
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="jax_enable_x64"):
            LRStateSpaceModel(gdf, ["y ~ 1 + x"], domain=[DOMAIN], verbose=False,
                              backend="cpu", dtype=jnp.float64)
        assert jax.config.jax_enable_x64 is False

    @pytest.mark.parametrize("call", ["fit", "sim"])
    def test_fit_and_sim_before_setup_raise_clearly(self, gdf, call):
        model = LRStateSpaceModel(gdf, ["y ~ 1 + x"], domain=[DOMAIN], verbose=False, backend="cpu")
        with pytest.raises(ValueError, match=r"setup\(cov_fun"):
            model.fit(options=_options()) if call == "fit" else model.sim(verbose=False)

    def test_mismatched_formulas_and_domains_raise(self, gdf):
        with pytest.raises(ValueError, match="must match"):
            LRStateSpaceModel(gdf, ["y ~ 1"], domain=[DOMAIN, DOMAIN], verbose=False, backend="cpu")


class TestFit:
    def test_result_contract(self, fitted):
        model, res = fitted
        p, q, T = model.shape
        assert isinstance(res, LRStateSpaceResults)
        assert res.y_hat.shape == (p, T)
        assert res.Sigma_y_hat.shape == (p, p, T)
        assert res.x_smoothed.shape == (q, T + 1)
        assert res.P_smoothed.shape == (q, q, T + 1)
        assert res.iterations == MAX_ITER and res.convergence is False
        assert len(res.llf_path) == MAX_ITER + 1  # initial entry + one per iteration
        assert [len(v) for v in res.y_hat_list] == [p]

    def test_result_contract_types_shapes_dtypes(self, fitted):
        # Documented contract (StateSpaceResults): array attributes are NumPy
        # arrays in the model dtype; llf is a float.
        model, res = fitted
        p, q, T = model.shape
        b = model.Xbeta.shape[1]
        dt = np.dtype(model.dtype)
        shapes = {"y_obs": (p, T), "Xbeta": (p, b, T), "y_hat": (p, T), "Sigma_y_hat": (p, p, T),
                  "residuals": (p, T), "x_smoothed": (q, T + 1), "P_smoothed": (q, q, T + 1),
                  "S11": (q, q), "S10": (q, q), "S00": (q, q)}
        for name, shape in shapes.items():
            value = getattr(res, name)
            assert type(value) is np.ndarray, name
            assert value.shape == shape and value.dtype == dt, name
        for name in ("llf", "aic", "bic"):
            assert isinstance(getattr(res, name), float), name
        assert all(isinstance(v, float) for v in res.llf_path)
        assert isinstance(res.nobs, int) and isinstance(res.missing, int) and res.nobs + res.missing == p * T
        assert isinstance(res.iterations, int) and isinstance(res.convergence, bool)
        assert res.termination in ("tolerance", "decrease", "invalid", "max_iter")
        for name in ("beta", "s2e", "f", "A", "ks", "x0", "Sigma0"):
            assert np.asarray(getattr(res.params, name).value).dtype == dt, name

    def test_parameter_shapes_and_domains(self, fitted):
        _, res = fitted
        prm = res.params
        assert prm.beta.value.shape == (2,)
        assert prm.A.value.shape == (1, 1)
        assert float(prm.s2e.value[0]) > 0
        assert 0 < float(prm.f.value[0]) < 1
        assert float(prm.ks.value[0]) > 0

    def test_recovers_regression_slope(self, fitted):
        # Data were simulated with slope 0.5 on x.
        _, res = fitted
        assert float(res.params.beta.value[1]) == pytest.approx(0.5, abs=0.05)

    def test_loglikelihood_increases(self, fitted):
        _, res = fitted
        llf = np.asarray(res.llf_path[1:], dtype=float)
        assert np.all(np.isfinite(llf))
        assert np.all(np.diff(llf) > 0)

    def test_llf_path_entry_zero_is_nan(self, fitted):
        # F41: iteration 0 is logged before any E-step, so it has no
        # log-likelihood; nor has iteration 1 a previous one to differ from.
        _, res = fitted
        assert np.isnan(res.llf_path[0])
        assert np.all(np.isfinite(res.llf_path[1:]))
        assert np.isnan(float(res.nstats[1]["deltaL"]))

    def test_refit_is_deterministic(self, fitted):
        model, res = fitted
        again = model.fit(options=_options())
        assert again.llf == res.llf
        for k in ("beta", "s2e", "f", "A", "ks"):
            np.testing.assert_array_equal(np.asarray(getattr(again.params, k).value),
                                          np.asarray(getattr(res.params, k).value))

    def test_fixed_parameter_is_held(self, gdf, mesh):
        model = _build_model(gdf, mesh)
        res = model.fit(params0=ModelParams(f=Param("f", [0.5], fixed=True)), options=_options(max_iter=2))
        np.testing.assert_allclose(np.asarray(res.params.f.value), [0.5])

    def test_fixed_initial_state_reaches_the_filter(self, gdf, mesh):
        model = _build_model(gdf, mesh)
        q = model.shape[1]
        params0 = ModelParams(x0=Param("x0", np.ones(q), fixed=True),
                              Sigma0=Param("Sigma0", 5.0 * np.eye(q), fixed=True))
        model.fit(params0=params0, options=_options(max_iter=2))
        np.testing.assert_allclose(np.asarray(model.x0), np.ones(q), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(model.Sigma0), 5.0 * np.eye(q), rtol=1e-6)

    @pytest.mark.parametrize("missing_frac", [0.0, 0.1], ids=["complete", "missing"])
    def test_reported_loglik_matches_dense_reference(self, mesh, missing_frac):
        """Reported llf equals the observed-data likelihood of an independent
        dense filter run at the returned parameters (whose system matrices
        the model holds after fit, F08) and the user-fixed prior."""
        gdf = make_panel_gdf(n_sites=N_SITES, n_times=N_TIMES, missing_frac=missing_frac, seed=4)
        model = _build_model(gdf, mesh)
        q = model.shape[1]
        x0, Sigma0 = 0.5 * np.ones(q), 3.0 * np.eye(q)
        params0 = ModelParams(x0=Param("x0", x0, fixed=True), Sigma0=Param("Sigma0", Sigma0, fixed=True))
        res = model.fit(params0=params0, options=_options(max_iter=1))
        f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
        np.testing.assert_array_equal(f64(model.beta), f64(res.params.beta.value))
        prob = SSMProblem(H=f64(model.H), R=f64(model.R), F=f64(model.F), Q=f64(model.Q),
                          x0=x0, Sigma0=Sigma0, Xbeta=f64(model.Xbeta), beta=f64(model.beta),
                          y=f64(model.y_train))
        ref = kalman_filter(prob)
        # llf is a sum over observations, so its float32 rounding error grows
        # with n_obs; a relative tolerance would be meaningless when llf ~ 0.
        assert float(res.llf) == pytest.approx(ref["llf"], abs=1e-5 * ref["n_obs"])

    def test_results_refer_to_the_returned_parameters(self, fitted):
        # F08: y_hat, the model's matrices and the params are one iterate.
        model, res = fitted
        f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
        for name in ("beta", "x0", "Sigma0"):
            np.testing.assert_array_equal(f64(getattr(model, name)), f64(getattr(res.params, name).value))
        np.testing.assert_allclose(f64(model.R), np.full(model.shape[0], f64(res.params.s2e.value)[0]), rtol=1e-6)
        np.testing.assert_allclose([c.rescale for c in model.cov_function], f64(res.params.ks.value), rtol=1e-6)
        Xb = np.einsum("pbt,b->pt", f64(model.Xbeta), f64(res.params.beta.value))
        y_hat = Xb + f64(model.H) @ f64(res.x_smoothed)[:, 1:]
        np.testing.assert_allclose(f64(res.y_hat), y_hat, rtol=1e-5, atol=1e-5)

    def test_float32_fit_does_not_depend_on_global_x64(self, gdf, mesh):
        # F03: with x64 enabled (e.g. by other code in the process), a float32
        # model must stay float32 and give bit-identical results.
        out = {}
        for x64 in (False, True):
            jax.config.update("jax_enable_x64", x64)
            model = _build_model(gdf, mesh, dtype=jnp.float32)
            res = model.fit(options=_options(max_iter=2))
            arrays = {name: np.asarray(getattr(res.params, name).value)
                      for name in ("beta", "s2e", "f", "A", "ks", "x0", "Sigma0")}
            arrays.update(y_hat=np.asarray(res.y_hat), x_smoothed=np.asarray(res.x_smoothed),
                          H=np.asarray(model.H), Q=np.asarray(model.Q))
            out[x64] = arrays, res.llf
        for name, a in out[True][0].items():
            assert a.dtype == np.float32, name
            np.testing.assert_array_equal(a, out[False][0][name], err_msg=name)
        assert out[True][1] == out[False][1]  # llf is a Python float (computed in float32)

    def test_max_iter_termination_is_recorded(self, fitted):
        _, res = fitted
        assert res.termination == "max_iter"

    def test_printing_model_and_results_has_no_side_effects(self, fitted):
        model, res = fitted
        assert_printing_is_side_effect_free(model)
        assert_printing_is_side_effect_free(res)

    def test_summary_renders(self, fitted):
        _, res = fitted
        text = str(res.summary())
        assert "LR State Space Model Results" in text


class TestStoppingRule:
    """EM stopping paths (F18), driven by a scripted log-likelihood.

    Real EM does not decrease on this problem, so the E-step is wrapped to
    return the scripted llf of each iteration; everything else is the real
    fit. ``beta_seen[k]`` is the beta the (k+1)-th E-step was evaluated at.
    Calls beyond the script (the final E-step after a normal stop, F08)
    repeat its last value.
    """

    TOL = 1e-3

    @pytest.fixture
    def model(self, stop_model):
        return stop_model

    @pytest.fixture
    def scripted(self, model, monkeypatch):
        real = model._E_step

        def install(llfs):
            beta_seen = []

            def fake(y_obs):
                out = real(y_obs)
                beta_seen.append(np.asarray(model.beta).copy())
                return (*out[:6], llfs[min(len(beta_seen), len(llfs)) - 1], out[7])
            monkeypatch.setattr(model, "_E_step", fake)
            return beta_seen
        return install

    def _fit(self, model, **kw):
        return model.fit(options=_options(tol_relat=self.TOL, **kw))

    @pytest.fixture
    def n_obs(self, model):
        return int(np.sum(~np.isnan(np.asarray(model.y_train))))

    @pytest.mark.parametrize("offset", [0.0, -1e7], ids=["llf~1e3", "llf~1e7"])
    def test_tolerance_is_per_observation(self, model, scripted, n_obs, offset):
        # The same llf changes stop (or not) EM regardless of |llf|: adding a
        # constant to the log-likelihood must not change when EM stops.
        small, large = 0.5 * self.TOL * n_obs, 2.0 * self.TOL * n_obs
        scripted([offset - 1e3, offset - 500.0, offset - 500.0 + small])
        res = self._fit(model, max_iter=5)
        assert (res.termination, res.convergence, res.iterations) == ("tolerance", True, 3)

        scripted([offset - 1e3, offset - 500.0, offset - 500.0 + large])
        res = self._fit(model, max_iter=3)
        assert (res.termination, res.convergence) == ("max_iter", False)

    @pytest.mark.parametrize("force_iter", [False, True])
    def test_small_decrease_is_convergence_and_returns_best(self, model, scripted, n_obs, force_iter):
        beta_seen = scripted([-1e3, -500.0, -500.0 - 0.5 * self.TOL * n_obs])
        res = self._fit(model, max_iter=5, force_iter=force_iter)
        assert (res.termination, res.convergence) == ("tolerance", True)
        assert res.llf == -500.0
        np.testing.assert_array_equal(np.asarray(res.params.beta.value), beta_seen[1])

    def test_final_estep_is_reported_when_it_improves(self, model, scripted):
        # Converges at iteration 3; the 4th E-step evaluates the returned params.
        beta_seen = scripted([-1e3, -500.0, -499.9, -499.8])
        res = self._fit(model, max_iter=5)
        assert (res.termination, res.iterations, res.llf) == ("tolerance", 3, -499.8)
        np.testing.assert_array_equal(np.asarray(res.params.beta.value), beta_seen[3])

    @pytest.mark.parametrize("final", [-499.95, float("nan")], ids=["worse", "nan"])
    def test_final_estep_falls_back_to_best(self, model, scripted, final):
        beta_seen = scripted([-1e3, -500.0, -499.9, final])
        if np.isnan(final):
            with pytest.warns(RuntimeWarning, match="best estimate so far"):
                res = self._fit(model, max_iter=5)
        else:
            res = self._fit(model, max_iter=5)
        assert (res.termination, res.convergence, res.llf) == ("tolerance", True, -499.9)
        np.testing.assert_array_equal(np.asarray(res.params.beta.value), beta_seen[2])
        np.testing.assert_array_equal(np.asarray(model.beta), beta_seen[2])

    def test_large_decrease_stops_without_force_iter(self, model, scripted):
        beta_seen = scripted([-1e3, -500.0, -600.0])
        with pytest.warns(RuntimeWarning, match="decreased"):
            res = self._fit(model, max_iter=5)
        assert (res.termination, res.convergence) == ("decrease", False)
        assert res.llf == -500.0
        np.testing.assert_array_equal(np.asarray(res.params.beta.value), beta_seen[1])

    def test_force_iter_continues_and_nan_restores_best(self, model, scripted):
        # Under force_iter the decreased iterate (-600) must not replace the
        # best one (-500) as the estimate restored on NaN.
        beta_seen = scripted([-1e3, -500.0, -600.0, float("nan")])
        with pytest.warns(RuntimeWarning) as record:
            res = self._fit(model, max_iter=5, force_iter=True)
        messages = [str(w.message) for w in record]
        assert any("decreased" in m and "continuing" in m for m in messages)
        assert any("NaN" in m for m in messages)
        assert (res.termination, res.convergence) == ("invalid", False)
        assert res.llf == -500.0
        np.testing.assert_array_equal(np.asarray(res.params.beta.value), beta_seen[1])


class TestPredict:
    """`predict` (mean and point-wise variance, arrays) and `get_prediction`
    (`LRPredictionResults`); neither modifies the fit results (P2)."""

    def test_predict_on_time_window(self, fitted, gdf):
        model, res = fitted
        times = np.sort(gdf["Time"].unique())
        sub = gdf[gdf["Time"] <= times[9]].drop(columns="y")
        y_pred, var_pred = res.predict(sub, verbose=False)
        assert type(y_pred) is np.ndarray and y_pred.shape == (N_SITES, 10)
        assert type(var_pred) is np.ndarray and var_pred.shape == (N_SITES, 10)
        assert np.all(var_pred >= -1e-6)
        pred = res.get_prediction(sub, verbose=False)
        assert isinstance(pred, LRPredictionResults)
        assert pred.Sigma_y_pred.shape == (N_SITES, N_SITES, 10)
        np.testing.assert_array_equal(pred.y_pred, y_pred)
        assert not hasattr(res, "y_pred")  # the fit results are not modified

    def test_variance_is_the_diagonal_of_the_covariance(self, fitted, gdf):
        # predict computes diag(H P H') without the (p, p, T) covariance.
        _, res = fitted
        newdata = gdf.drop(columns="y")
        _, var_pred = res.predict(newdata, verbose=False)
        pred = res.get_prediction(newdata, verbose=False)
        diag = np.diagonal(pred.Sigma_y_pred, axis1=0, axis2=1).T
        scale = np.abs(diag).max()
        np.testing.assert_allclose(var_pred, diag, rtol=1e-5, atol=1e-6 * scale)
        np.testing.assert_array_equal(pred.var_pred, diag)

    def test_predict_at_new_locations(self, fitted, mesh):
        model, res = fitted
        xy = np.array([[2.0, 2.0], [5.0, 5.0], [8.0, 3.0]])
        new = make_panel_gdf(n_sites=3, n_times=N_TIMES, missing_frac=0.0, seed=3, xy=xy).drop(columns="y")
        y_pred, var_pred = res.predict(new, variance=False, verbose=False)
        assert y_pred.shape == (3, N_TIMES) and var_pred is None
        assert np.all(np.isfinite(y_pred))
        pred = res.get_prediction(new, std=False, verbose=False)
        assert pred.Sigma_y_pred is None and pred.var_pred is None
        np.testing.assert_array_equal(pred.y_pred, y_pred)

    def test_model_predict_is_a_shortcut(self, fitted, gdf):
        model, res = fitted
        newdata = gdf.drop(columns="y")
        for a, b in zip(model.predict(newdata, res, verbose=False), res.predict(newdata, verbose=False)):
            np.testing.assert_array_equal(a, b)

    def test_predict_outside_training_period_raises(self, fitted, gdf):
        model, res = fitted
        later = gdf.copy()
        later["Time"] = later["Time"] + np.timedelta64(365, "D")
        with pytest.raises(ValueError):
            res.predict(later.drop(columns="y"), verbose=False)
        with pytest.raises(ValueError):
            res.get_prediction(later.drop(columns="y"), verbose=False)

    def test_in_sample_prediction_reproduces_fitted_values(self, fitted, gdf):
        _, res = fitted
        y_pred, _ = res.predict(gdf.drop(columns="y"), variance=False, verbose=False)
        np.testing.assert_allclose(y_pred, res.y_hat, rtol=1e-5, atol=1e-5)


@pytest.fixture(scope="module")
def converged(gdf, mesh):
    # float64 fit with a negative definite Hessian (30 iterations), with SEs.
    before = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        model = _build_model(gdf, mesh, dtype=jnp.float64)
        res = model.fit(options=_options(max_iter=30))
        res.compute_cov_params()
    finally:
        jax.config.update("jax_enable_x64", before)
    return model, res


@pytest.mark.usefixtures("x64")
class TestStandardErrors:
    """Hessian and standard errors (F21), float64 for finite differences."""

    def _fit(self, gdf, mesh, max_iter):
        model = _build_model(gdf, mesh, dtype=jnp.float64)
        return model, model.fit(options=_options(max_iter=max_iter))

    def test_hessian_matches_finite_differences(self, converged):
        model, res = converged
        H = np.asarray(res._compute_hessian()[0])

        # Independent check: central second differences of the observed
        # log-likelihood in the same flat parameterisation.
        params = res._inference_params()
        logL = model._observed_logL(res.y_obs, res.Xbeta, params.x0.value, params.Sigma0.value)
        vec0, meta = _pack_params(params, dtype=jnp.float64)
        f = jax.jit(lambda v: logL(_unpack_params(v, params, meta)))
        x0 = np.asarray(vec0)
        h = 1e-3 * np.maximum(np.abs(x0), 0.1)
        n = x0.size
        E = np.eye(n)
        fd = np.empty((n, n))
        for i in range(n):
            for j in range(i, n):
                fpp = float(f(x0 + h[i] * E[i] + h[j] * E[j]))
                fpm = float(f(x0 + h[i] * E[i] - h[j] * E[j]))
                fmp = float(f(x0 - h[i] * E[i] + h[j] * E[j]))
                fmm = float(f(x0 - h[i] * E[i] - h[j] * E[j]))
                fd[i, j] = fd[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
        np.testing.assert_allclose(H, fd, rtol=1e-3, atol=1e-4 * np.abs(H).max())

        se = np.asarray(res.bse_vector)
        assert np.all(np.isfinite(se)) and np.all(se > 0)

    def test_bounded_parameters_have_transformed_intervals(self, converged):
        # Delta method: s2e and ks on log theta, theta * exp(-+ z se / theta);
        # f on atanh theta, tanh(atanh f -+ z se / (1 - f^2)); the other
        # parameters keep theta +- z se.
        _, res = converged
        theta, se, ci = res.theta_hat, np.asarray(res.bse_vector), res.conf_int(alpha=0.05)
        z = stats.t.ppf(0.975, df=res.df_resid)
        names = np.array([n for n, _, start, stop, _ in res._free_meta for _ in range(start, stop)])
        log, ath = np.isin(names, ["s2e", "ks"]), names == "f"
        nat = ~(log | ath)
        assert (log.sum(), ath.sum(), nat.sum()) == (2, 1, 3)
        r = np.exp(z * se[log] / theta[log])
        np.testing.assert_allclose(ci[log], np.column_stack([theta[log] / r, theta[log] * r]), rtol=1e-12)
        assert np.all(ci[log, 0] > 0)
        f, dz = theta[ath], z * se[ath] / (1 - theta[ath] ** 2)
        np.testing.assert_allclose(ci[ath], np.column_stack([np.tanh(np.arctanh(f) - dz),
                                                             np.tanh(np.arctanh(f) + dz)]), rtol=1e-12)
        assert np.all((-1 < ci[ath, 0]) & (ci[ath, 1] < 1))
        np.testing.assert_allclose(ci[nat], np.column_stack([theta[nat] - z * se[nat],
                                                             theta[nat] + z * se[nat]]), rtol=1e-12)
        # t-values and p-values stay on the natural scale.
        np.testing.assert_allclose(res.tvalues, theta / se, rtol=1e-12)

    def test_summary_intervals_match_conf_int(self, converged):
        _, res = converged
        ci = res.conf_int(alpha=0.10)
        rows = {}
        for line in str(res.summary(alpha=0.10)).splitlines():
            tok = line.split()
            if len(tok) == 7 and tok[0] in ("Intercept", "x", "s2e_0", "f_0", "A_0_0", "rescale_0"):
                rows[tok[0]] = [float(tok[5]), float(tok[6])]
        # flat order of theta_hat: beta (Intercept, x), s2e, f, A, ks
        printed = np.array([rows[k] for k in ("Intercept", "x", "s2e_0", "f_0", "A_0_0", "rescale_0")])
        np.testing.assert_allclose(printed, ci, atol=5e-4)

    def test_log_scale_interval_is_positive_where_wald_is_not(self, converged):
        _, res = converged
        _, _, wald = res._stats_from_arrays([0.1], [0.08])
        _, _, logci = res._stats_from_arrays([0.1], [0.08], scale="log")
        assert wald[0, 0] < 0 < logci[0, 0] < 0.1 < logci[0, 1]

    def test_atanh_interval_stays_below_one_where_wald_does_not(self, converged):
        _, res = converged
        _, _, wald = res._stats_from_arrays([0.95], [0.05])
        _, _, ath = res._stats_from_arrays([0.95], [0.05], scale="atanh")
        assert ath[0, 0] < 0.95 < ath[0, 1] < 1 < wald[0, 1]
        # |f| >= 1: atanh is undefined, the natural-scale interval is kept.
        _, _, ci = res._stats_from_arrays([1.02, -1.0], [0.05, 0.05], scale="atanh")
        _, _, wald = res._stats_from_arrays([1.02, -1.0], [0.05, 0.05])
        np.testing.assert_array_equal(ci, wald)

    def test_indefinite_hessian_gives_nan_standard_errors(self, gdf, mesh):
        # After 3 EM iterations this estimate is not at a maximum: the Hessian
        # has a positive eigenvalue, so -H^-1 is not a covariance matrix.
        _, res = self._fit(gdf, mesh, max_iter=3)
        with pytest.warns(RuntimeWarning, match="not negative definite"):
            res.compute_cov_params()
        # Warned once, by compute_cov_params; not again on every access.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.all(np.isnan(np.asarray(res.bse_vector)))
            assert np.all(np.isnan(res.conf_int()))


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize("ks", [0.3, 1.0, 3.0])
def test_scipy_and_jax_marginal_precisions_agree(gdf, mesh, ks):
    # Q^-1 is built with SciPy for the E-step (spdeAppoxCov.precision) and
    # with JAX for the ks M-step and the Hessian (_spde_marginal_precision).
    model = _build_model(gdf, mesh, dtype=jnp.float64)
    (cov,) = model.cov_function
    cov.rescale = ks
    P_scipy = np.asarray(cov.precision(marginal=True))
    P_jax = np.asarray(_spde_marginal_precision(jnp.array([ks]), *model._spde_mesh_arrays()))
    np.testing.assert_allclose(P_jax, P_scipy, rtol=0, atol=1e-12 * np.abs(P_scipy).max())


def _state_G(model, f, ks, T, S11, S10, S00, ldim):
    """State part of the expected complete-data log-likelihood (no constants):
    -1/2 [-T log|Q^-1| + tr(Q^-1 Omega(f))], Q^-1 the SPDE precision at ks."""
    P = np.asarray(_spde_marginal_precision(jnp.asarray(ks, dtype=jnp.float64), *model._spde_mesh_arrays()))
    Fd = np.repeat(np.asarray(f, dtype=float), ldim)
    Omega = S11 - S10 * Fd[None, :] - Fd[:, None] * S10.T + np.outer(Fd, Fd) * S00
    return -0.5 * (-T * np.linalg.slogdet(P)[1] + np.trace(P @ Omega))


@pytest.fixture(scope="module")
def m_step_call(gdf, mesh):
    """Arguments and result of the third M-step of a float64 fit."""
    before = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        model = _build_model(gdf, mesh, dtype=jnp.float64)
        orig, calls = model._M_step, []

        def spy(*args, **kwargs):
            bound = inspect.signature(orig).bind(*args, **kwargs).arguments
            ks_cur = [float(c.rescale) for c in bound["est_covList"]]
            out = orig(*args, **kwargs)
            calls.append((bound, ks_cur, out[0]))
            return out

        model._M_step = spy
        model.fit(options=_options(max_iter=3))
    finally:
        jax.config.update("jax_enable_x64", before)
    bound, ks_cur, new = calls[-1]
    a = {k: np.asarray(v, dtype=float) for k, v in bound.items() if k in ("F", "S11", "S10", "S00", "invQ")}
    bq = np.asarray(bound["block_q"])
    return dict(model=model, orig=orig, bound=bound, ks_cur=np.array(ks_cur), new=new, ldim=bq[1:] - bq[:-1],
                T=bound["y_t"].shape[1], f_cur=a["F"][bq[:-1]], **{k: a[k] for k in ("S11", "S10", "S00", "invQ")})


@pytest.mark.usefixtures("x64")
class TestMStepF40:
    """M-step of f and ks (F40): with Q not proportional to I, f must weight
    the traces by Q^-1, and ks is optimised given the new f (ECM)."""

    def test_score_in_f_matches_the_weighted_formula(self, converged):
        # Fisher identity: d llf / df = d/df E[log p(x, y)] at the current
        # parameters = tr(Q^-1 S10) - f tr(Q^-1 S00), from the E-step at those
        # parameters (res.S10/S00). The weighted f update is its root.
        model, res = converged
        params = res._inference_params()
        logL = model._observed_logL(res.y_obs, res.Xbeta, params.x0.value, params.Sigma0.value,
                                    fitted_state=res._fitted_state)
        vec, meta = _pack_params(params, dtype=jnp.float64)
        score = np.asarray(jax.grad(lambda v: logL(_unpack_params(v, params, meta)))(vec))
        (i,) = [start for n, _, start, _, _ in meta if n == "f"]
        P = np.asarray(_spde_marginal_precision(jnp.asarray(res.params.ks.value, dtype=jnp.float64),
                                                *model._spde_mesh_arrays()))
        f = float(np.asarray(res.params.f.value)[0])
        expected = np.trace(P @ res.S10) - f * np.trace(P @ res.S00)
        assert abs(expected) > 1.0  # not yet converged in f: a non-trivial check
        assert score[i] == pytest.approx(expected, rel=1e-6)

    def test_f_update_maximises_the_expected_loglik(self, m_step_call):
        c = m_step_call
        P, S10, S00 = c["invQ"], c["S10"], c["S00"]
        f_new = float(np.asarray(c["new"].f.value)[0])
        assert f_new == pytest.approx(np.trace(P @ S10) / np.trace(P @ S00), rel=1e-10)
        G = lambda f: _state_G(c["model"], [f], c["ks_cur"], c["T"], c["S11"], S10, S00, c["ldim"])  # noqa: E731
        opt = optimize.minimize_scalar(lambda f: -G(f), bounds=(-1.5, 1.5), method="bounded",
                                       options={"xatol": 1e-10})
        assert f_new == pytest.approx(opt.x, abs=1e-6)
        f_unweighted = np.trace(S10) / np.trace(S00)
        assert abs(f_unweighted - f_new) > 1e-3 and G(f_new) > G(f_unweighted)

    def test_ks_is_updated_given_the_new_f(self, m_step_call):
        c = m_step_call
        G = lambda f, ks: _state_G(c["model"], f, ks, c["T"], c["S11"], c["S10"], c["S00"], c["ldim"])  # noqa: E731
        f_new, ks_new = np.asarray(c["new"].f.value), np.asarray(c["new"].ks.value)
        # Each conditional step increases G (ECM).
        assert G(c["f_cur"], c["ks_cur"]) <= G(f_new, c["ks_cur"]) <= G(f_new, ks_new)
        # The old order (ks given the current f) does worse at the new f.
        covs = c["bound"]["est_covList"]
        for cov, k in zip(covs, c["ks_cur"]):
            cov.rescale = k
        ks_old_order = np.asarray(c["orig"](**{**c["bound"], "f_fixed": True})[0].ks.value)
        assert not np.allclose(ks_old_order, ks_new, rtol=1e-6)
        assert G(f_new, ks_new) > G(f_new, ks_old_order)

    def test_loglik_never_decreases(self, converged):
        _, res = converged
        llf = np.asarray(res.llf_path[1:])  # [0] is nan: no E-step at iteration 0 (F41)
        assert np.all(np.diff(llf) >= -1e-9 * np.abs(llf[1:]))


class TestCompilation:
    def test_refit_does_not_recompile_ks_objective(self, fitted):
        # F20: the ks objective is compiled once per process and mesh size,
        # not once per fit().
        model, _ = fitted
        before = _ks_value_and_grad._cache_size()
        model.fit(options=_options(max_iter=2))
        assert _ks_value_and_grad._cache_size() == before > 0


class TestResultsLifecycle:
    """Each results object is a snapshot of its own fit (F09, F27)."""

    def _model(self, gdf, mesh):
        cov = spdeAppoxCov(latlon=False, verbose=False).setup(mesh, domain=[DOMAIN], stats=False)
        model = LRStateSpaceModel(gdf, ["y ~ 1 + x"], domain=[DOMAIN], verbose=False, backend="cpu")
        return model.setup([cov]), cov

    def test_refit_returns_new_results_and_leaves_earlier_ones_unchanged(self, gdf, mesh):
        model, _ = self._model(gdf, mesh)
        res_a = model.fit(options=_options(max_iter=3))
        lo_a, hi_a = (a.copy() for a in res_a.conf_int_y(prediction=True))
        cov_a = res_a.coverage_probability()
        s2e_a = float(res_a.params.s2e.value[0])

        res_b = model.fit(params0=ModelParams(s2e=Param("s2e", [2.0], fixed=True)), options=_options(max_iter=3))

        # The earlier results are unchanged...
        lo, hi = res_a.conf_int_y(prediction=True)
        np.testing.assert_array_equal(lo, lo_a)
        np.testing.assert_array_equal(hi, hi_a)
        assert res_a.coverage_probability() == cov_a == res_a._compute_coverage_probability(0.05, "global")
        assert float(res_a.params.s2e.value[0]) == s2e_a
        # ...and the new results reflect the new fit: s2e = 2 in the noise
        # part of their prediction intervals.
        z = 1.959963984540054
        var_b = np.diagonal(res_b.Sigma_y_hat, axis1=0, axis2=1).T + 2.0  # (p, T)
        lo_b, hi_b = res_b.conf_int_y(prediction=True)
        np.testing.assert_allclose(hi_b - lo_b, 2 * z * np.sqrt(var_b), rtol=1e-5)

    def test_predict_with_earlier_results_after_refit(self, gdf, mesh):
        model, cov = self._model(gdf, mesh)
        newdata = gdf.drop(columns="y")
        res_a = model.fit(options=_options(max_iter=3))
        y_pred_a = res_a.predict(newdata, variance=False, verbose=False)[0].copy()
        model.fit(params0=ModelParams(s2e=Param("s2e", [2.0], fixed=True)), options=_options(max_iter=3))
        rescale = cov.rescale
        np.testing.assert_array_equal(res_a.predict(newdata, variance=False, verbose=False)[0], y_pred_a)
        assert cov.rescale == rescale  # predict does not modify the covariance functions

    def test_results_survive_setup_with_another_mesh(self, gdf, mesh):
        model, _ = self._model(gdf, mesh)
        newdata = gdf.drop(columns="y")
        res = model.fit(options=_options(max_iter=3))
        H = np.asarray(res._compute_hessian()[0]).copy()
        y_pred = res.predict(newdata, variance=False, verbose=False)[0].copy()
        p, q, T = model.shape
        text = str(res.summary())

        other = spdeAppoxCov(latlon=False, verbose=False).setup(make_square_mesh(mesh_size=2.5),
                                                               domain=[DOMAIN], stats=False)
        model.setup([other])
        assert model.shape[1] != q  # the model now has another latent dimension

        res._hessian = None  # recompute from the results' own fit-time state
        np.testing.assert_array_equal(np.asarray(res._compute_hessian()[0]), H)
        np.testing.assert_array_equal(res.predict(newdata, variance=False, verbose=False)[0], y_pred)
        assert f"q = {q}" in text and f"q = {q}" in str(res.summary())


class TestFittedState:
    """The fit-time configuration kept by results (`FittedState`, F09/F27)."""

    def _fit(self, gdf, mesh, max_iter=3):
        cov = spdeAppoxCov(latlon=False, verbose=False).setup(mesh, domain=[DOMAIN], stats=False)
        model = LRStateSpaceModel(gdf, ["y ~ 1 + x"], domain=[DOMAIN], verbose=False, backend="cpu").setup([cov])
        return model, cov, model.fit(options=_options(max_iter=max_iter))

    def test_fitted_state_is_frozen_and_read_only(self, gdf, mesh):
        model, cov, res = self._fit(gdf, mesh, max_iter=2)
        state = res._fitted_state
        assert isinstance(state, FittedState)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.y_name = ("z",)
        assert not state.points[0].flags.writeable and not state.pdim.flags.writeable
        assert state.cov_function == (cov,) and state.y_name == ("y",) and model.y_name == ["y"]
        assert state.builders == tuple(model.builders) and state.dtype == np.dtype(model.dtype)

    def test_predict_and_hessian_ignore_later_changes_to_the_model(self, gdf, mesh):
        model, _, res = self._fit(gdf, mesh)
        newdata = gdf.drop(columns="y")
        y_pred = res.predict(newdata, variance=False, verbose=False)[0].copy()
        H = np.asarray(res._compute_hessian()[0]).copy()

        # Outside code changes the model configuration the results used: the
        # formula builders (predict crashed with the live ones), points and
        # block sizes (Hessian), and the domain (it only flags points).
        model.domain = [Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])]
        model.builders = model.points = model.pdim = None

        np.testing.assert_array_equal(res.predict(newdata, variance=False, verbose=False)[0], y_pred)
        res._hessian = None
        np.testing.assert_array_equal(np.asarray(res._compute_hessian()[0]), H)

    def test_fitted_state_holds_the_estimates(self, gdf, mesh):
        # P1: the public, read-only fitted_state also holds the estimates
        # (the results' own arrays, not copies).
        _, _, res = self._fit(gdf, mesh, max_iter=2)
        state = res.fitted_state
        assert state is res._fitted_state
        with pytest.raises(AttributeError):
            res.fitted_state = None
        assert state.params is res.params
        assert state.x_smoothed is res.x_smoothed and state.P_smoothed is res.P_smoothed
        assert not state.x_smoothed.flags.writeable and not state.timestamps_hat[0].flags.writeable
        np.testing.assert_array_equal(state.timestamps_hat[0], res.timestamps_hat[0])
        new = res.update(params=res.params.copy())
        assert new.fitted_state.params is new.params and res.fitted_state.params is res.params

    def test_state_attributes_are_views_of_the_fitted_state(self, gdf, mesh):
        # The latent state is held only by the fitted state; the results'
        # attributes read it (same arrays), read-only, in the model dtype.
        model, _, res = self._fit(gdf, mesh, max_iter=2)
        q, T = model.shape[1], model.shape[2]
        shapes = {"x_smoothed": (q, T + 1), "P_smoothed": (q, q, T + 1), "S11": (q, q), "S10": (q, q), "S00": (q, q)}
        for name in FittedState.STATE_FIELDS:
            assert name not in vars(res), name
            assert getattr(res, name) is getattr(res.fitted_state, name), name
            with pytest.raises(AttributeError):
                setattr(res, name, None)
        for name, shape in shapes.items():
            value = getattr(res, name)
            assert type(value) is np.ndarray and value.shape == shape, name
            assert value.dtype == np.dtype(model.dtype) and not value.flags.writeable, name
        assert res.P_pred_smoothed is None

    def test_update_of_the_state_goes_to_the_fitted_state(self, gdf, mesh):
        _, _, res = self._fit(gdf, mesh, max_iter=2)
        x_new = np.zeros_like(res.x_smoothed, dtype=np.float64)
        new = res.update(x_smoothed=x_new)
        assert new.x_smoothed is new.fitted_state.x_smoothed
        assert new.x_smoothed.dtype == res.x_smoothed.dtype and not new.x_smoothed.flags.writeable
        np.testing.assert_array_equal(new.x_smoothed, 0.0)
        assert np.any(res.x_smoothed != 0.0) and new.P_smoothed is res.P_smoothed

    @pytest.mark.filterwarnings("ignore:The Hessian of the log-likelihood is not negative definite")
    def test_results_do_not_reference_the_model(self, gdf, mesh):
        # P1: results keep no reference to the model; the Hessian, predict,
        # back_transform, to_geo and the summary work after it is collected.
        model, _, res = self._fit(gdf, mesh, max_iter=2)
        assert not hasattr(res, "model")
        y_pred = model.predict(gdf.drop(columns="y"), res, verbose=False)[0].copy()
        ref = weakref.ref(model)
        del model
        gc.collect()
        assert ref() is None
        res.compute_cov_params()
        assert res._hessian is not None
        np.testing.assert_array_equal(res.predict(gdf.drop(columns="y"), verbose=False)[0], y_pred)
        pred = res.get_prediction(gdf.drop(columns="y"), verbose=False)
        res.back_transform(lambda v: v)
        pred.back_transform(lambda v: v)
        assert len(res.to_geo()) == len(res.y_hat.ravel()) and len(pred.to_geo()) == pred.y_pred.size
        assert "LRStateSpaceModel" in str(res.summary())


class TestSimulate:
    @pytest.mark.usefixtures("x64")
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["float32", "float64"])
    def test_sim_keeps_model_dtype(self, gdf, mesh, dtype):
        # F16: the internal StateSpaceModel used to default to float32.
        model = _build_model(gdf, mesh, dtype=dtype)
        params = ModelParams(beta=Param("beta", [2.0, 0.5]), s2e=Param("s2e", [0.1]), f=Param("f", [0.8]),
                             A=Param("A", [[1.0]]), ks=Param("ks", [1.0]))
        y_sim, x_sim, info, _ = model.sim(params=params, verbose=False)
        assert info["sim_model"].dtype == jnp.dtype(dtype)
        assert np.asarray(y_sim).dtype == np.dtype(dtype)
        assert np.asarray(x_sim).dtype == np.dtype(dtype)
        p, q, T = model.shape
        assert np.asarray(y_sim).shape == (p, T)
        assert np.all(np.isfinite(np.asarray(y_sim)))


class TestResultsMethods:
    def test_sample_size_counts_observed_values_only(self, fitted):
        # BIC and df_resid use the number of observed values, not p*T (F14).
        model, res = fitted
        y = np.asarray(model.y_train)
        n_obs = int(np.sum(~np.isnan(y)))
        assert (res.nobs, res.missing) == (n_obs, y.size - n_obs)
        assert res.missing > 0
        assert res.bic == pytest.approx(np.log(n_obs) * res.n_params - 2 * res.llf, rel=1e-12)
        assert res.df_resid == n_obs - res.n_params

    def test_n_params_excludes_initial_state(self, gdf, mesh):
        # F36: x0 and Sigma0 counted as one parameter each.
        model = _build_model(gdf, mesh)
        q = model.shape[1]
        res_free = model.fit(options=_options(max_iter=1))
        sizes = {k: np.asarray(getattr(res_free.params, k).value).size for k in ("beta", "s2e", "f", "A", "ks")}
        assert res_free.n_params == sum(sizes.values())
        params0 = ModelParams(x0=Param("x0", np.zeros(q), fixed=True), Sigma0=Param("Sigma0", np.eye(q), fixed=True),
                              f=Param("f", [0.5], fixed=True))
        res_fixed = model.fit(params0=params0, options=_options(max_iter=1))
        assert res_fixed.n_params == sum(sizes.values()) - sizes["f"]

    def test_intervals_and_coverage(self, fitted):
        _, res = fitted
        lo, hi = res.conf_int_y(alpha=0.05, prediction=True)
        assert lo.shape == hi.shape == res.y_hat.shape
        assert np.all(lo <= hi)
        cov = res.coverage_probability(alpha=0.05)
        assert 0.0 <= cov <= 1.0

    def test_back_transform_identity(self, fitted):
        _, res = fitted
        res.back_transform(lambda v: v)
        np.testing.assert_allclose(res.y_hat_back_list[0], res.y_hat_list[0], rtol=1e-6)

    def test_output_names_use_the_response_name(self, gdf, mesh):
        # F39: columns read y_hat_['y'], std_hat_['y'] and the summary [['y']].
        model = _build_model(gdf, mesh)
        res = model.fit(options=_options(max_iter=1))
        pred = res.get_prediction(gdf.drop(columns="y"), verbose=False)
        res.back_transform(lambda v: v)
        pred.back_transform(lambda v: v)
        geo_hat, geo_pred = res.to_geo(), pred.to_geo()
        assert {"y_hat_y", "std_hat_y", "y_hat_back_y", "std_hat_back_y"} <= set(geo_hat.columns)
        assert {"y_pred_y", "std_pred_y", "y_pred_back_y", "std_pred_back_y"} <= set(geo_pred.columns)
        poly = pred.to_poly([DOMAIN], std=True)
        assert {"y", "std_y"} <= set(poly.columns)
        assert not any("[" in str(c) for df in (geo_hat, geo_pred, poly) for c in df.columns)
        for obj in (res, model, pred):
            (line,) = [ln for ln in str(obj.summary()).splitlines() if ln.startswith("Dep. Variables:")]
            assert line.split()[2] == "y"


def test_to_poly_std_with_two_responses():
    # F10. Prediction built directly: the defect was in to_poly's own loop,
    # so no fit is needed. One polygon containing all 4 points, so W = 1/4 * 1'.
    n, T = 4, 3
    pts = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    ts = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
    y_pred = np.vstack([np.ones((n, T)), 2.0 * np.ones((n, T))])
    # Response a: iid unit points. Response b: unit variances, correlation 0.5.
    Sigma_b = 0.5 * np.eye(n) + 0.5 * np.ones((n, n))
    Sigma = np.zeros((2 * n, 2 * n))
    Sigma[:n, :n], Sigma[n:, n:] = np.eye(n), Sigma_b
    state = types.SimpleNamespace(y_name=("a", "b"), dtype=np.dtype(np.float64))
    pred = LRPredictionResults(state, y_pred, Sigma_y_pred=np.tile(Sigma[:, :, None], (1, 1, T)),
                               points=[pts, pts], block_p=[0, n, 2 * n], timestamps=[ts, ts], crs="EPSG:3857")
    out = pred.to_poly([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], std=True)
    np.testing.assert_allclose(out["a"], 1.0)
    np.testing.assert_allclose(out["b"], 2.0)
    # sqrt(w' Sigma w), w = 1/4: a -> sqrt(4/16) = 0.5; b -> sqrt((4 + 12 * 0.5)/16)
    np.testing.assert_allclose(out["std_a"], 0.5)
    np.testing.assert_allclose(out["std_b"], np.sqrt(10 / 16))
