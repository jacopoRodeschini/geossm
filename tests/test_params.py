"""`Param`, `ModelParams` and `FitOptions` containers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geossm.stmodel import FitOptions, ModelParams, Param
from tests._known_bugs import known_bug


def test_arrays_are_wrapped_as_free_params():
    mp = ModelParams(beta=[1.0, 2.0], s2e=0.5)
    assert isinstance(mp.beta, Param) and mp.beta.name == "beta" and not mp.beta.fixed
    np.testing.assert_allclose(np.asarray(mp.s2e.value), [0.5])  # scalars become 1-D
    assert mp.f.value is None


def test_free_params_excludes_fixed():
    mp = ModelParams(beta=Param("beta", [1.0], fixed=True), s2e=0.5)
    free = mp.free_params()
    assert "beta" not in free and "s2e" in free


def test_size_counts_all_set_values():
    mp = ModelParams(beta=[1.0, 2.0], A=[[1.0, 0.0], [0.0, 1.0]])
    assert mp.size == 6


def test_set_on_fixed_param_is_noop():
    p = Param("beta", [1.0], fixed=True)
    assert p.set([5.0]) is p


def test_freeze_unfreeze_roundtrip():
    p = Param("beta", [1.0])
    assert p.freeze().fixed and not p.freeze().unfreeze().fixed


def test_copy_is_independent():
    mp = ModelParams(beta=[1.0, 2.0], s2e=0.5)
    cp = mp.copy()
    cp.beta.fixed = True
    assert not mp.beta.fixed
    np.testing.assert_array_equal(np.asarray(cp.beta.value), np.asarray(mp.beta.value))


def test_grad_flows_through_free_params_only():
    mp = ModelParams(beta=[1.0, 2.0], s2e=0.5, f=[0.8], A=[[1.0]], ks=[1.0], x0=[0.0], Sigma0=[[1.0]])
    g = jax.grad(lambda m: jnp.sum(m.beta.value ** 2) + jnp.sum(m.s2e.value))(mp)
    np.testing.assert_allclose(np.asarray(g.beta.value), [2.0, 4.0])
    np.testing.assert_allclose(np.asarray(g.s2e.value), [1.0])


def test_jit_with_free_params():
    mp = ModelParams(beta=[1.0, 2.0], s2e=0.5)
    out = jax.jit(lambda m: m.s2e.value * 2)(mp)
    np.testing.assert_allclose(np.asarray(out), [1.0])


@known_bug("F13", "fixed Param stores arrays in pytree aux_data; jit rejects it", raises=ValueError)
def test_jit_with_fixed_param():
    # The failure appears on the second call, when JAX compares the new
    # pytree's aux_data (holding the fixed array) with the cached one.
    f = jax.jit(lambda m: m.s2e.value * 2)
    for s2e in (0.5, 0.7):
        mp = ModelParams(beta=Param("beta", [1.0, 2.0], fixed=True), s2e=s2e)
        np.testing.assert_allclose(np.asarray(f(mp)), [2 * s2e], rtol=1e-6)


def test_fit_options_defaults():
    o = FitOptions()
    assert (o.max_iter, o.tol_relat, o.force_iter, o.std) == (20, pytest.approx(1e-3), False, True)
