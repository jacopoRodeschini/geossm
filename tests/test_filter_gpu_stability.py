"""
Regression tests for numerical stability of the Kalman filter/smoother
kernels in geossm.ssm.statespace on GPU with float32.

Background
----------
`_filter_kernelJAX`'s covariance update `P_upd = P_pred - K @ H @ P_pred`
subtracts two similar-magnitude matrices. On GPU, JAX's default (unset)
matmul precision can route float32 `@`/dot ops through reduced-precision
tensor-core kernels (e.g. TensorFloat32). The resulting per-matmul error is
amplified by that cancellation into an outright indefinite P_pred at the
very next step -- the minimum eigenvalue can be off by ~1e-1, not ~1e-7 --
which no reasonable Cholesky jitter can absorb, and the Cholesky then
produces NaN.

The fix (`_highest_matmul_precision` in statespace.py, applied to every
JIT-compiled kernel) forces full matmul precision inside the kernels
regardless of any ambient `jax.default_matmul_precision` the caller has
set, so it can't be silently defeated by unrelated code elsewhere. The
kernels no longer add a Cholesky jitter (F15): it biased results, and all
tests below pass without it.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from geossm.ssm.statespace import _filter_kernelJAX, _smoother_kernelJAX


def _gpu_devices():
    try:
        return jax.devices("gpu")
    except RuntimeError:
        return []


GPU_DEVICES = _gpu_devices()
HAS_GPU = len(GPU_DEVICES) > 0
CPU_DEVICE = jax.devices("cpu")[0]
GPU_DEVICE = GPU_DEVICES[0] if HAS_GPU else None

gpu_required = pytest.mark.skipif(not HAS_GPU, reason="no GPU device available")


def _make_ill_conditioned_ssm(
    seed, p, q, T, dtype,
    min_eig=1e-8, max_eig=2.0,
    f_lo=0.95, f_hi=0.999,
    nan_frac=0.2, obs_frac=0.3,
):
    """
    Build a synthetic, deliberately ill-conditioned low-rank-SSM-like
    problem: Q has eigenvalues spanning many orders of magnitude (mimicking
    a SPDE/Matern precision-derived covariance on a fine, irregular mesh),
    F is a near-unit-root persistent AR(1) diagonal (so P_pred accumulates
    across steps instead of resetting), and H/observations are sparse with
    missing data -- all properties of the real low-rank spatio-temporal
    model this reproduces the bug from (see cs_1_grins_PM10.py).
    """
    rng = np.random.default_rng(seed)

    eigvals = np.sort(np.exp(rng.uniform(np.log(min_eig), np.log(max_eig), size=q)))[::-1]
    Qrot, _ = np.linalg.qr(rng.normal(size=(q, q)))
    Q_np = (Qrot * eigvals) @ Qrot.T
    Q_np = 0.5 * (Q_np + Q_np.T)

    f_diag = rng.uniform(f_lo, f_hi, size=q)
    H_np = rng.normal(size=(p, q)) * (rng.uniform(size=(p, q)) < obs_frac)
    R_np = rng.uniform(0.01, 1.0, size=p)
    x0_np = np.zeros(q)
    Sigma0_np = np.eye(q)
    Xbeta_np = np.zeros((p, 1, T))
    beta_np = np.zeros(1)
    y_np = rng.normal(size=(p, T))
    y_np[rng.uniform(size=y_np.shape) < nan_frac] = np.nan

    to = lambda a: jnp.asarray(a, dtype=dtype)
    return dict(
        Q=to(Q_np), F=to(f_diag), H=to(H_np), R=to(R_np),
        x0=to(x0_np), Sigma0=to(Sigma0_np),
        Xbeta=to(Xbeta_np), beta=to(beta_np), y=to(y_np),
    )


def _run_filter_smoother(mats):
    x_t, P_t, K, x_t_1, P_t_1, logL = _filter_kernelJAX(
        mats["y"], mats["H"], mats["R"], mats["F"], mats["Q"],
        mats["x0"], mats["Sigma0"], mats["Xbeta"], mats["beta"],
    )
    jax.block_until_ready(x_t)
    x_T, P_T, P_T_1 = _smoother_kernelJAX(mats["H"], mats["F"], x_t, P_t, K, x_t_1, P_t_1)
    jax.block_until_ready(x_T)
    return x_t, P_t, logL, x_T, P_T


def _assert_finite(name, arr):
    arr_np = np.asarray(arr)
    assert np.isfinite(arr_np).all(), f"{name} contains NaN/Inf"


@gpu_required
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("T", [50, 500, 2000])
def test_filter_smoother_no_nan_on_gpu(dtype, T):
    """
    Regression test for the GPU + float32 NaN bug: an ill-conditioned but
    valid PSD Q, run through the real filter/smoother kernels on GPU, must
    never produce NaN/Inf regardless of dtype or horizon length. Before the
    `_highest_matmul_precision` fix, float32 reliably went NaN within the
    first couple of timesteps (T=50 already reproduces it).
    """
    if dtype == jnp.float64:
        jax.config.update("jax_enable_x64", True)

    with jax.default_device(GPU_DEVICE):
        mats = _make_ill_conditioned_ssm(seed=0, p=60, q=25, T=T, dtype=dtype)
        x_t, P_t, logL, x_T, P_T = _run_filter_smoother(mats)

    _assert_finite("x_filtered", x_t)
    _assert_finite("P_filtered", P_t)
    _assert_finite("x_smoothed", x_T)
    _assert_finite("P_smoothed", P_T)
    assert np.isfinite(float(logL))


#: min_eig=1e-8, f_hi=0.999 is calibrated (see `find_safe_T` in the test
#: suite's development notes) to sit just inside float32's own precision
#: floor for this (p=80, q=30) problem shape -- confirmed to run clean on
#: *CPU* float32 up to T=500 for every seed used below -- so a failure at
#: these settings is attributable to the GPU-specific reduced-precision-
#: matmul bug this suite targets, not to float32 itself being fundamentally
#: too coarse for the problem. Pushed further out (T=3000), some random
#: draws *do* exceed even CPU's float32 floor; `test_gpu_cpu_parity_at_precision_floor`
#: below covers that regime by checking GPU/CPU agree rather than asserting
#: GPU alone must succeed.
_ADVERSARIAL_MIN_EIG = 1e-8
_ADVERSARIAL_F_HI = 0.999


@gpu_required
@pytest.mark.parametrize("seed", range(5))
def test_filter_smoother_no_nan_adversarial(seed):
    """
    Adversarial sweep: a small minimum eigenvalue, near-unit-root
    persistence, and a high missing-data fraction (some timesteps can be
    entirely NaN) -- across several independent random seeds so this isn't
    tuned to one lucky draw. Calibrated to stay just inside float32's own
    precision floor (see `_ADVERSARIAL_MIN_EIG` above), so this isolates
    the GPU-specific bug rather than float32's inherent limits.
    """
    with jax.default_device(GPU_DEVICE):
        mats = _make_ill_conditioned_ssm(
            seed=seed, p=80, q=30, T=500, dtype=jnp.float32,
            min_eig=_ADVERSARIAL_MIN_EIG, max_eig=5.0,
            f_lo=0.99, f_hi=_ADVERSARIAL_F_HI,
            nan_frac=0.4, obs_frac=0.2,
        )
        x_t, P_t, logL, x_T, P_T = _run_filter_smoother(mats)

    _assert_finite(f"x_filtered (seed={seed})", x_t)
    _assert_finite(f"P_filtered (seed={seed})", P_t)
    _assert_finite(f"x_smoothed (seed={seed})", x_T)
    _assert_finite(f"P_smoothed (seed={seed})", P_T)
    assert np.isfinite(float(logL))


@gpu_required
@pytest.mark.parametrize("seed", range(8))
def test_gpu_cpu_parity_at_precision_floor(seed):
    """
    Push past float32's genuine precision floor (T=3000 at the same
    min_eig/f_hi as above -- confirmed some seeds exceed even CPU's float32
    capability here) and check GPU/CPU *agree*, rather than asserting GPU
    alone must succeed (which would not be a reasonable claim once the
    problem is hard enough that float32 itself, not the GPU-specific bug,
    is the limiting factor).

    This is the real invariant `_highest_matmul_precision` restores: before
    the fix, GPU would NaN far earlier and independently of whether CPU
    could handle the same problem; after the fix, GPU and CPU fail on
    exactly the same seeds (usually within a few timesteps of each other)
    and agree closely whenever both succeed.
    """
    kwargs = dict(
        p=80, q=30, T=3000, dtype=jnp.float32,
        min_eig=_ADVERSARIAL_MIN_EIG, max_eig=5.0,
        f_lo=0.99, f_hi=_ADVERSARIAL_F_HI,
        nan_frac=0.4, obs_frac=0.2,
    )

    with jax.default_device(CPU_DEVICE):
        mats_cpu = _make_ill_conditioned_ssm(seed=seed, **kwargs)
        x_t_cpu, _, logL_cpu, _, _ = _run_filter_smoother(mats_cpu)

    with jax.default_device(GPU_DEVICE):
        mats_gpu = _make_ill_conditioned_ssm(seed=seed, **kwargs)
        x_t_gpu, _, logL_gpu, _, _ = _run_filter_smoother(mats_gpu)

    nan_cpu = not np.isfinite(np.asarray(x_t_cpu)).all()
    nan_gpu = not np.isfinite(np.asarray(x_t_gpu)).all()
    assert nan_cpu == nan_gpu, (
        f"seed={seed}: CPU {'failed' if nan_cpu else 'succeeded'} but "
        f"GPU {'failed' if nan_gpu else 'succeeded'} -- GPU/CPU precision parity broken"
    )

    if not nan_cpu:
        rel_logL = abs(float(logL_gpu) - float(logL_cpu)) / max(abs(float(logL_cpu)), 1.0)
        assert rel_logL < 1e-2, f"seed={seed}: logL relative difference too large: {rel_logL:.3e}"


@gpu_required
def test_gpu_float32_matches_cpu_float64_reference():
    """
    Beyond "no NaN": the GPU float32 result should be numerically close to
    a CPU float64 reference, not merely finite-but-wrong -- i.e. the fix
    doesn't just mask the bug behind a different, still-incorrect number.
    Tolerances are loose (this is a sanity check on the fix, not a
    precision benchmark) but tight enough to catch a badly wrong result.
    """
    jax.config.update("jax_enable_x64", True)

    with jax.default_device(CPU_DEVICE):
        mats64 = _make_ill_conditioned_ssm(seed=1, p=60, q=25, T=500, dtype=jnp.float64)
        _, _, logL_ref, x_T_ref, _ = _run_filter_smoother(mats64)

    with jax.default_device(GPU_DEVICE):
        mats32 = _make_ill_conditioned_ssm(seed=1, p=60, q=25, T=500, dtype=jnp.float32)
        _, _, logL_gpu, x_T_gpu, _ = _run_filter_smoother(mats32)

    rel_logL = abs(float(logL_gpu) - float(logL_ref)) / abs(float(logL_ref))
    assert rel_logL < 1e-2, f"logL relative difference too large: {rel_logL:.3e}"

    x_T_ref_np = np.asarray(x_T_ref)
    x_T_gpu_np = np.asarray(x_T_gpu)
    max_abs_diff = np.max(np.abs(x_T_ref_np - x_T_gpu_np))
    scale = max(np.max(np.abs(x_T_ref_np)), 1.0)
    assert max_abs_diff / scale < 1e-2, (
        f"GPU float32 smoothed state diverges too much from the float64 reference: "
        f"{max_abs_diff:.3e}"
    )


def test_filter_smoother_no_nan_on_cpu_baseline():
    """
    CPU control for the adversarial setup used in the GPU tests above: this
    should already pass on the pre-fix code too (CPU never hit the bug),
    and confirms the adversarial matrices themselves are not simply
    "impossible" to filter.
    """
    with jax.default_device(CPU_DEVICE):
        mats = _make_ill_conditioned_ssm(
            seed=2, p=80, q=30, T=500, dtype=jnp.float32,
            min_eig=_ADVERSARIAL_MIN_EIG, max_eig=5.0,
            f_lo=0.99, f_hi=_ADVERSARIAL_F_HI,
            nan_frac=0.4, obs_frac=0.2,
        )
        x_t, P_t, logL, x_T, P_T = _run_filter_smoother(mats)

    _assert_finite("x_filtered", x_t)
    _assert_finite("P_filtered", P_t)
    _assert_finite("x_smoothed", x_T)
    _assert_finite("P_smoothed", P_T)
    assert np.isfinite(float(logL))
