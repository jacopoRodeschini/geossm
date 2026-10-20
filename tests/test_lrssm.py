"""End-to-end `LRStateSpaceModel` workflow on a small synthetic dataset (CPU).

One EM fit is shared by the module (about 10 s including JIT compilation).
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import Polygon

from geossm.covmodel import spdeAppoxCov
from geossm.stmodel import FitOptions, LRStateSpaceModel, LRStateSpaceResults, ModelParams, Param
from geossm.stmodel.lrssm import _ks_value_and_grad
from geossm.stmodel.stmodel_results import _pack_params, _unpack_params
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
    def test_predict_on_time_window(self, fitted, gdf):
        model, res = fitted
        times = np.sort(gdf["Time"].unique())
        sub = gdf[gdf["Time"] <= times[9]].drop(columns="y")
        out = model.predict(sub, res, verbose=False)
        assert out is res  # predict() stores into and returns the passed results (F27)
        assert res.y_pred.shape == (N_SITES, 10)
        assert res.Sigma_y_pred.shape == (N_SITES, N_SITES, 10)
        assert np.all(np.diagonal(res.Sigma_y_pred, axis1=0, axis2=1) >= -1e-6)

    def test_predict_at_new_locations(self, fitted, mesh):
        model, res = fitted
        xy = np.array([[2.0, 2.0], [5.0, 5.0], [8.0, 3.0]])
        new = make_panel_gdf(n_sites=3, n_times=N_TIMES, missing_frac=0.0, seed=3, xy=xy)
        model.predict(new.drop(columns="y"), res, verbose=False, std=False)
        assert res.y_pred.shape == (3, N_TIMES)
        assert res.Sigma_y_pred is None
        assert np.all(np.isfinite(res.y_pred))

    def test_predict_outside_training_period_raises(self, fitted, gdf):
        model, res = fitted
        later = gdf.copy()
        later["Time"] = later["Time"] + np.timedelta64(365, "D")
        with pytest.raises(ValueError):
            model.predict(later.drop(columns="y"), res, verbose=False)

    def test_in_sample_prediction_reproduces_fitted_values(self, fitted, gdf):
        model, res = fitted
        y_hat = res.y_hat.copy()
        model.predict(gdf.drop(columns="y"), res, verbose=False, std=False)
        np.testing.assert_allclose(res.y_pred, y_hat, rtol=1e-5, atol=1e-5)


@pytest.mark.usefixtures("x64")
class TestStandardErrors:
    """Hessian and standard errors (F21), float64 for finite differences."""

    def _fit(self, gdf, mesh, max_iter):
        model = _build_model(gdf, mesh, dtype=jnp.float64)
        return model, model.fit(options=_options(max_iter=max_iter))

    def test_hessian_matches_finite_differences(self, gdf, mesh):
        model, res = self._fit(gdf, mesh, max_iter=30)
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

        res.compute_cov_params()
        se = np.asarray(res.bse_vector)
        assert np.all(np.isfinite(se)) and np.all(se > 0)

    def test_indefinite_hessian_gives_nan_standard_errors(self, gdf, mesh):
        # After 3 EM iterations this estimate is not at a maximum: the Hessian
        # has a positive eigenvalue, so -H^-1 is not a covariance matrix.
        _, res = self._fit(gdf, mesh, max_iter=3)
        with pytest.warns(RuntimeWarning, match="not negative definite"):
            res.compute_cov_params()
        with pytest.warns(RuntimeWarning, match="set to NaN"):  # from bse_vector
            assert np.all(np.isnan(np.asarray(res.bse_vector)))


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
        y_pred_a = res_a.predict(newdata, verbose=False, std=False).y_pred.copy()
        model.fit(params0=ModelParams(s2e=Param("s2e", [2.0], fixed=True)), options=_options(max_iter=3))
        rescale = cov.rescale
        np.testing.assert_array_equal(res_a.predict(newdata, verbose=False, std=False).y_pred, y_pred_a)
        assert cov.rescale == rescale  # predict does not modify the covariance functions

    def test_results_survive_setup_with_another_mesh(self, gdf, mesh):
        model, _ = self._model(gdf, mesh)
        newdata = gdf.drop(columns="y")
        res = model.fit(options=_options(max_iter=3))
        H = np.asarray(res._compute_hessian()[0]).copy()
        y_pred = res.predict(newdata, verbose=False, std=False).y_pred.copy()
        p, q, T = model.shape
        text = str(res.summary())

        other = spdeAppoxCov(latlon=False, verbose=False).setup(make_square_mesh(mesh_size=2.5),
                                                               domain=[DOMAIN], stats=False)
        model.setup([other])
        assert model.shape[1] != q  # the model now has another latent dimension

        res._hessian = None  # recompute from the results' own fit-time state
        np.testing.assert_array_equal(np.asarray(res._compute_hessian()[0]), H)
        np.testing.assert_array_equal(res.predict(newdata, verbose=False, std=False).y_pred, y_pred)
        assert f"q = {q}" in text and f"q = {q}" in str(res.summary())


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


def test_to_poly_std_with_two_responses():
    # F10. Stubbed results object: the defect was in to_poly's own loop, so
    # no fit is needed. One polygon containing all 4 points, so W = 1/4 * 1'.
    res = object.__new__(LRStateSpaceResults)
    n, T = 4, 3
    pts = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    ts = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
    res.y_pred = np.vstack([np.ones((n, T)), 2.0 * np.ones((n, T))])
    # Response a: iid unit points. Response b: unit variances, correlation 0.5.
    Sigma_b = 0.5 * np.eye(n) + 0.5 * np.ones((n, n))
    Sigma = np.zeros((2 * n, 2 * n))
    Sigma[:n, :n], Sigma[n:, n:] = np.eye(n), Sigma_b
    res.Sigma_y_pred = np.tile(Sigma[:, :, None], (1, 1, T))
    res.block_p_pred = np.array([0, n, 2 * n])
    res.points_pred, res.timestamps_pred = [pts, pts], [ts, ts]
    res.crs_pred, res.y_pred_back_list = "EPSG:3857", None
    res.model = types.SimpleNamespace(y_name=["a", "b"])
    out = res.to_poly([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], std=True)
    np.testing.assert_allclose(out["a"], 1.0)
    np.testing.assert_allclose(out["b"], 2.0)
    # sqrt(w' Sigma w), w = 1/4: a -> sqrt(4/16) = 0.5; b -> sqrt((4 + 12 * 0.5)/16)
    np.testing.assert_allclose(out["std_a"], 0.5)
    np.testing.assert_allclose(out["std_b"], np.sqrt(10 / 16))
