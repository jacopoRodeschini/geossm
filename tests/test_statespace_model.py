"""Public contract of `geossm.ssm.StateSpaceModel` and `StateSpaceResults`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geossm.ssm import StateSpaceModel, StateSpaceResults
from tests._known_bugs import known_bug
from tests._reference import kalman_filter, rts_smoother


def _model(prob, dtype=jnp.float32, **overrides):
    kw = dict(H=prob.H, R=prob.R, F=prob.F, Q=prob.Q, x0=prob.x0, Sigma0=prob.Sigma0,
              Xbeta=prob.Xbeta, beta=prob.beta, backend="cpu", dtype=dtype)
    kw.update(overrides)
    return StateSpaceModel(**kw)


class TestConstruction:
    def test_dimensions(self, problem):
        m = _model(problem)
        assert m.shape == (problem.p, problem.q, problem.T)
        assert m.b == problem.beta.size

    def test_dense_R_and_F_are_stored_as_diagonals(self, problem):
        m = _model(problem, R=np.diag(problem.R), F=np.diag(problem.F))
        np.testing.assert_allclose(np.asarray(m.R), problem.R, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(m.F), problem.F, rtol=1e-6)

    def test_inconsistent_shapes_raise(self, problem):
        with pytest.raises(ValueError):
            _model(problem, H=problem.H[:, :-1])

    def test_default_dtype_is_float32(self, problem):
        m = _model(problem)
        assert m.dtype == jnp.float32
        assert m.H.dtype == jnp.float32

    @known_bug("F03", "requesting float64 flips the process-global jax_enable_x64 flag")
    def test_float64_does_not_change_global_x64_flag(self, problem):
        before = jax.config.jax_enable_x64
        try:
            _model(problem, dtype=jnp.float64)
        except ValueError:
            pass  # rejecting float64 without x64 enabled is an acceptable fix
        assert jax.config.jax_enable_x64 == before

    @known_bug("F11", "the default Xbeta has shape (p, 1, 1), so filtering T > 1 fails",
               raises=ValueError)
    def test_filter_without_covariates(self, problem):
        m = StateSpaceModel(H=problem.H, R=problem.R, F=problem.F, Q=problem.Q, backend="cpu")
        res = m.filter(problem.y)
        assert res.x_filtered.shape == (problem.q, problem.T + 1)

    @known_bug("F11", "a wrongly shaped y is reported as an H/R/Xbeta shape error")
    def test_wrong_y_shape_error_names_y(self, problem):
        m = _model(problem)
        with pytest.raises(ValueError, match="y"):
            m.filter(np.zeros((problem.p + 2, problem.T)))


class TestInitialState:
    def test_filter_preserves_user_initial_state(self, problem):
        m = _model(problem)
        m.filter(problem.y)
        np.testing.assert_allclose(np.asarray(m.x0), problem.x0, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(m.Sigma0), problem.Sigma0, rtol=1e-6)

    @pytest.mark.usefixtures("x64")
    def test_filter_uses_user_initial_state(self, problem):
        res = _model(problem, dtype=jnp.float64).filter(problem.y)
        ref = kalman_filter(problem)
        np.testing.assert_allclose(res.x_filtered[:, 1:], ref["x_filt"][:, 1:], atol=1e-8)
        assert float(res.llf) == pytest.approx(ref["llf"], abs=1e-8)

    def test_setting_other_matrices_keeps_initial_state(self, problem):
        m = _model(problem)
        m.Q = 2.0 * problem.Q
        m.set(H=problem.H, R=problem.R)
        np.testing.assert_allclose(np.asarray(m.x0), problem.x0, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(m.Sigma0), problem.Sigma0, rtol=1e-6)

    def test_changing_state_dimension_resets_incompatible_initial_state(self, problem):
        m = _model(problem)
        q_new = problem.q + 1
        m.set(H=np.ones((problem.p, q_new)), F=np.full(q_new, 0.5), Q=np.eye(q_new))
        np.testing.assert_array_equal(np.asarray(m.x0), np.zeros(q_new))
        np.testing.assert_array_equal(np.asarray(m.Sigma0), np.eye(q_new))

    @pytest.mark.usefixtures("x64")
    def test_explicit_initial_state_per_call_is_used(self, problem):
        m = _model(problem, dtype=jnp.float64)
        res = m.smoother(problem.y, x0=problem.x0, Sigma0=problem.Sigma0)
        ref = rts_smoother(problem)
        np.testing.assert_allclose(res.x_smoothed, ref["x_smooth"], atol=1e-8)
        np.testing.assert_allclose(res.P_smoothed, ref["P_smooth"], atol=1e-8)


class TestFilterSmootherResults:
    def test_filter_result_contract(self, problem):
        res = _model(problem).filter(problem.y)
        p, q, T = problem.p, problem.q, problem.T
        assert isinstance(res, StateSpaceResults)
        assert res.x_filtered.shape == (q, T + 1)
        assert res.P_filtered.shape == (q, q, T + 1)
        assert res.x_pred.shape == (q, T + 1)
        assert res.y_hat.shape == (p, T)
        assert isinstance(res.x_filtered, np.ndarray)
        assert res.x_filtered.dtype == np.float32
        assert np.isfinite(res.llf)

    def test_smoother_light_drops_filter_arrays(self, problem):
        res = _model(problem).smoother(problem.y, light=True)
        assert res.x_filtered is None and res.P_filtered is None and res.K is None
        assert res.x_smoothed.shape == (problem.q, problem.T + 1)
        assert res.S11.shape == (problem.q, problem.q)

    def test_estimate_equals_smoother(self, problem):
        m = _model(problem)
        a, b = m.estimate(problem.y), m.smoother(problem.y)
        assert a.llf == b.llf
        np.testing.assert_array_equal(a.x_smoothed, b.x_smoothed)

    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")  # all-NaN time step
    def test_residuals_and_mse_ignore_missing(self, problem_missing, missing_mask):
        res = _model(problem_missing).smoother(problem_missing.y)
        assert np.isnan(res.residuals[missing_mask]).all()
        assert np.isfinite(res.residuals[~missing_mask]).all()
        expected = np.mean(res.residuals[~missing_mask] ** 2)
        assert res.mse() == pytest.approx(expected, rel=1e-6)
        assert res.rmse() == pytest.approx(np.sqrt(expected), rel=1e-6)
        assert res.mse("space").shape == (problem_missing.p,)
        assert res.mse("time").shape == (problem_missing.T,)

    def test_missing_count(self, problem_missing, missing_mask):
        res = _model(problem_missing).smoother(problem_missing.y)
        assert res.missing == int(missing_mask.sum())

    def test_nobs_counts_only_observed_entries(self, problem_missing, missing_mask):
        res = _model(problem_missing).smoother(problem_missing.y)
        assert res.nobs == int((~missing_mask).sum())

    def test_conf_int_state_brackets_mean(self, problem):
        res = _model(problem).smoother(problem.y)
        lo, hi = res.conf_int_state()
        assert lo.shape == hi.shape == res.x_smoothed.shape
        assert np.all(lo <= res.x_smoothed) and np.all(res.x_smoothed <= hi)


class TestPredict:
    def test_predict_shapes(self, problem):
        m = _model(problem)
        res = m.smoother(problem.y)
        y_hat, Sigma = m.predict(m.H, res.x_smoothed, res.P_smoothed, m.Xbeta, m.beta)
        assert y_hat.shape == (problem.p, problem.T)
        assert Sigma.shape == (problem.p, problem.p, problem.T)

    def test_predict_std_false(self, problem):
        m = _model(problem)
        res = m.smoother(problem.y)
        y_hat, Sigma = m.predict(m.H, res.x_smoothed, None, m.Xbeta, m.beta, std=False)
        assert Sigma is None
        np.testing.assert_allclose(np.asarray(y_hat), res.y_hat, rtol=1e-5, atol=1e-5)

    def test_predict_requires_covariance_when_std(self, problem):
        m = _model(problem)
        res = m.smoother(problem.y)
        with pytest.raises(ValueError):
            m.predict(m.H, res.x_smoothed, None, m.Xbeta, m.beta, std=True)


class TestSimulation:
    def test_sim_shapes_and_reproducibility(self, problem):
        m = _model(problem)
        y1, x1, _, _ = m.sim(seed=7, stats=False)
        y2, x2, _, _ = m.sim(seed=7, stats=False)
        assert y1.shape == (problem.p, problem.T)
        assert x1.shape == (problem.q, problem.T + 1)
        np.testing.assert_array_equal(np.asarray(y1), np.asarray(y2))


@known_bug("F12", "summary()/__str__ overwrite model.params and add model.model")
def test_str_does_not_mutate_model(problem):
    m = _model(problem)
    before = np.asarray(m.params).copy()
    str(m)
    np.testing.assert_array_equal(np.asarray(m.params), before)
    assert not hasattr(m, "model")


def test_repr_is_side_effect_free(problem):
    m = _model(problem)
    before = dict(vars(m))
    repr(m)
    assert vars(m).keys() == before.keys()
