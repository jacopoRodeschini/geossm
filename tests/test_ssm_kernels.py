"""JAX SSM kernels vs the independent dense reference in `_reference.py`.

Tolerances: the kernels add a scale-aware Cholesky jitter (1e-6 * mean
diagonal) at every step, which alone produces differences of order 1e-4 on
these problems (finding F15), hence ``atol=1e-3`` in float64.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests._known_bugs import known_bug
from tests._reference import kalman_filter, rts_smoother, sufficient_statistics
from geossm.ssm.statespace import (
    _compute_expected_values_kernelJAX,
    _filter_kernelJAX,
    _smoother_kernelJAX,
)

ATOL64 = 1e-3


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

    def test_loglik_matches_reference_without_2pi_constant(self, problem):
        # geossm's llf omits -n_obs/2 log(2 pi) (documented in the kernel).
        f, ref = _filter(problem), kalman_filter(problem)
        np.testing.assert_allclose(float(f["llf"]), ref["llf_no_const"], atol=ATOL64)

    @pytest.mark.parametrize("key", ["x_filt", "P_filt", "x_pred", "P_pred"])
    def test_states_with_missing_match_reference(self, problem_missing, key):
        f, ref = _filter(problem_missing), kalman_filter(problem_missing)
        np.testing.assert_allclose(np.asarray(f[key])[..., 1:], ref[key][..., 1:], atol=ATOL64)

    def test_fully_missing_timestep_has_no_update(self, problem_missing, missing_mask):
        t = int(np.flatnonzero(missing_mask.all(axis=0))[0])  # time index of an all-NaN column
        f = _filter(problem_missing)
        np.testing.assert_allclose(f["x_filt"][:, t + 1], f["x_pred"][:, t + 1], atol=1e-10)
        np.testing.assert_allclose(f["P_filt"][:, :, t + 1], f["P_pred"][:, :, t + 1], atol=1e-10)

    @known_bug("F02", "log-likelihood adds -1/2 sum_missing log R_ii")
    def test_loglik_with_missing_is_observed_data_likelihood(self, problem_missing):
        f, ref = _filter(problem_missing), kalman_filter(problem_missing)
        np.testing.assert_allclose(float(f["llf"]), ref["llf_no_const"], atol=ATOL64)

    @known_bug("F02", "llf depends on R of never-observed sites")
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
    """float32 on CPU: states within 1e-3 abs of the float64 reference.

    The llf is not compared here: with missing data it carries the F02 term.
    """
    f = _filter(problem_missing, dtype=jnp.float32)
    assert f["x_filt"].dtype == jnp.float32
    ref = kalman_filter(problem_missing)
    np.testing.assert_allclose(np.asarray(f["x_filt"])[:, 1:], ref["x_filt"][:, 1:], atol=1e-3)
    np.testing.assert_allclose(np.asarray(f["P_filt"])[..., 1:], ref["P_filt"][..., 1:], atol=1e-3)
