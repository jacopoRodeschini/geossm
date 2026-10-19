"""JAX SSM kernels vs the independent dense reference in `_reference.py`.

Tolerances: in float64 the kernels agree with the reference to rounding
(~1e-13 on these problems); ``ATOL64`` leaves ample margin. Until F15 a
Cholesky jitter at every step limited agreement to ~1e-4.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests._reference import SSMProblem, kalman_filter, rts_smoother, sufficient_statistics
from geossm.ssm.statespace import (
    _compute_expected_values_kernelJAX,
    _filter_kernelJAX,
    _smoother_kernelJAX,
)

ATOL64 = 1e-8


def _filter(prob, dtype=jnp.float64):
    a = lambda v: jnp.asarray(v, dtype=dtype)  # noqa: E731
    out = _filter_kernelJAX(
        a(prob.y), a(prob.H), a(prob.R), a(prob.F), a(prob.Q),
        a(prob.x0), a(prob.Sigma0), a(prob.Xbeta), a(prob.beta),
    )
    return dict(zip(["x_filt", "P_filt", "K", "x_pred", "P_pred", "llf"], out))


def _smooth(prob, filt, dtype=jnp.float64):
    out = _smoother_kernelJAX(
        jnp.asarray(prob.H, dtype=dtype), jnp.asarray(prob.F, dtype=dtype),
        filt["x_filt"], filt["P_filt"], filt["K"], filt["x_pred"], filt["P_pred"],
    )
    return dict(zip(["x_smooth", "P_smooth", "P_lag"], out))


@pytest.mark.usefixtures("x64")
class TestFilterAgainstReference:
    def test_output_shapes(self, problem):
        f = _filter(problem)
        q, p, T = problem.q, problem.p, problem.T
        assert f["x_filt"].shape == (q, T + 1)
        assert f["P_filt"].shape == (q, q, T + 1)
        assert f["x_pred"].shape == (q, T + 1)
        assert f["P_pred"].shape == (q, q, T + 1)
        assert f["K"].shape == (q, p)
        assert np.ndim(f["llf"]) == 0

    def test_prior_stored_at_index_zero(self, problem):
        f = _filter(problem)
        np.testing.assert_allclose(f["x_filt"][:, 0], problem.x0)
        np.testing.assert_allclose(f["P_filt"][:, :, 0], problem.Sigma0)

    @pytest.mark.parametrize("key", ["x_filt", "P_filt", "x_pred", "P_pred"])
    def test_states_match_reference(self, problem, key):
        f, ref = _filter(problem), kalman_filter(problem)
        np.testing.assert_allclose(np.asarray(f[key])[..., 1:], ref[key][..., 1:], atol=ATOL64)

    def test_loglik_matches_reference(self, problem):
        # Full Gaussian log-likelihood, including -n_obs/2 log(2 pi).
        f, ref = _filter(problem), kalman_filter(problem)
        np.testing.assert_allclose(float(f["llf"]), ref["llf"], atol=ATOL64)

    @pytest.mark.parametrize("s", [1e-4, 1e2])
    def test_change_of_units_is_exact(self, problem_missing, s):
        # Measuring y (and hence x) in units scaled by s is the same model:
        # states scale by s and llf shifts by exactly -n_obs log(s). Needs no
        # reference code; a jitter with a fixed floor broke it (F15).
        prob = problem_missing
        scaled = SSMProblem(**{**prob.__dict__, "R": prob.R * s**2, "Q": prob.Q * s**2,
                               "x0": prob.x0 * s, "Sigma0": prob.Sigma0 * s**2,
                               "beta": prob.beta * s, "y": prob.y * s})
        f, fs = _filter(prob), _filter(scaled)
        n_obs = int(np.sum(~np.isnan(prob.y)))
        np.testing.assert_allclose(np.asarray(fs["x_filt"]), s * np.asarray(f["x_filt"]), rtol=1e-9, atol=0)
        assert float(fs["llf"]) == pytest.approx(float(f["llf"]) - n_obs * np.log(s), rel=1e-12)

    def test_small_observation_noise_matches_reference(self, problem):
        # R ~ 1e-5: H'R^-1 H dominates the Woodbury matrix. Gave NaN with the
        # per-step jitter (F15) although float64 handles it to rounding.
        prob = SSMProblem(**{**problem.__dict__, "R": problem.R * 1e-5})
        f, ref = _filter(prob), kalman_filter(prob)
        np.testing.assert_allclose(np.asarray(f["x_filt"])[:, 1:], ref["x_filt"][:, 1:], atol=1e-6)
        assert float(f["llf"]) == pytest.approx(ref["llf"], rel=1e-9)

    @pytest.mark.parametrize("key", ["x_filt", "P_filt", "x_pred", "P_pred"])
    def test_states_with_missing_match_reference(self, problem_missing, key):
        f, ref = _filter(problem_missing), kalman_filter(problem_missing)
        np.testing.assert_allclose(np.asarray(f[key])[..., 1:], ref[key][..., 1:], atol=ATOL64)

    def test_fully_missing_timestep_has_no_update(self, problem_missing, missing_mask):
        t = int(np.flatnonzero(missing_mask.all(axis=0))[0])  # time index of an all-NaN column
        f = _filter(problem_missing)
        np.testing.assert_allclose(f["x_filt"][:, t + 1], f["x_pred"][:, t + 1], atol=1e-10)
        np.testing.assert_allclose(f["P_filt"][:, :, t + 1], f["P_pred"][:, :, t + 1], atol=1e-10)

    def test_loglik_with_missing_is_observed_data_likelihood(self, problem_missing):
        f, ref = _filter(problem_missing), kalman_filter(problem_missing)
        np.testing.assert_allclose(float(f["llf"]), ref["llf"], atol=ATOL64)

    def test_loglik_with_missing_does_not_depend_on_R_of_missing_entries(self, problem):
        # Observed-data likelihood must be invariant to the noise variance of a
        # site that is never observed. Pins F02 without any reference code.
        mask = np.zeros_like(problem.y, dtype=bool)
        mask[0, :] = True
        prob_a = problem.with_missing(mask)
        prob_b = problem.with_missing(mask)
        prob_b.R = prob_b.R.copy()
        prob_b.R[0] *= 10.0
        llf_a, llf_b = float(_filter(prob_a)["llf"]), float(_filter(prob_b)["llf"])
        assert llf_a == pytest.approx(llf_b, abs=ATOL64)

    def test_is_deterministic(self, problem):
        a, b = _filter(problem), _filter(problem)
        assert float(a["llf"]) == float(b["llf"])
        np.testing.assert_array_equal(np.asarray(a["P_filt"]), np.asarray(b["P_filt"]))


@pytest.mark.usefixtures("x64")
class TestSmootherAgainstReference:
    @pytest.mark.parametrize("missing", [False, True])
    @pytest.mark.parametrize("key", ["x_smooth", "P_smooth"])
    def test_moments_match_reference(self, problem, problem_missing, missing, key):
        prob = problem_missing if missing else problem
        f = _filter(prob)
        s, ref = _smooth(prob, f), rts_smoother(prob)
        np.testing.assert_allclose(np.asarray(s[key]), ref[key], atol=ATOL64)

    def test_last_smoothed_equals_last_filtered(self, problem):
        f = _filter(problem)
        s = _smooth(problem, f)
        np.testing.assert_allclose(s["x_smooth"][:, -1], f["x_filt"][:, -1])
        np.testing.assert_allclose(s["P_smooth"][:, :, -1], f["P_filt"][:, :, -1])

    @pytest.mark.parametrize("missing", [False, True])
    def test_lag_one_covariance_matches_reference(self, problem, problem_missing, missing):
        prob = problem_missing if missing else problem
        s, ref = _smooth(prob, _filter(prob)), rts_smoother(prob)
        np.testing.assert_allclose(np.asarray(s["P_lag"])[..., 1:], ref["P_lag"][..., 1:], atol=ATOL64)

    def test_smoothed_covariances_symmetric_psd(self, problem):
        s = _smooth(problem, _filter(problem))
        P = np.moveaxis(np.asarray(s["P_smooth"]), -1, 0)
        np.testing.assert_allclose(P, np.swapaxes(P, 1, 2), atol=1e-8)
        assert np.linalg.eigvalsh(0.5 * (P + np.swapaxes(P, 1, 2))).min() > 0


@pytest.mark.usefixtures("x64")
def test_sufficient_statistics_match_reference(problem):
    f = _filter(problem)
    s = _smooth(problem, f)
    y_hat, S11, S10, S00 = _compute_expected_values_kernelJAX(
        jnp.asarray(problem.H), s["x_smooth"], s["P_smooth"], s["P_lag"],
        jnp.asarray(problem.Xbeta), jnp.asarray(problem.beta),
    )
    ref = sufficient_statistics(problem, rts_smoother(problem))
    for name, val in dict(y_hat=y_hat, S11=S11, S10=S10, S00=S00).items():
        scale = max(1.0, np.abs(ref[name]).max())
        np.testing.assert_allclose(np.asarray(val), ref[name], atol=ATOL64 * scale, err_msg=name)


def test_float32_kernel_close_to_float64_reference(problem_missing):
    """float32 on CPU: states within 1e-3 abs, llf within 1e-4 rel of the float64 reference."""
    f = _filter(problem_missing, dtype=jnp.float32)
    assert f["x_filt"].dtype == jnp.float32
    ref = kalman_filter(problem_missing)
    np.testing.assert_allclose(np.asarray(f["x_filt"])[:, 1:], ref["x_filt"][:, 1:], atol=1e-3)
    np.testing.assert_allclose(np.asarray(f["P_filt"])[..., 1:], ref["P_filt"][..., 1:], atol=1e-3)
    assert float(f["llf"]) == pytest.approx(ref["llf"], rel=1e-4)
