"""`Param`, `ModelParams` and `FitOptions` containers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geossm.stmodel import FitOptions, ModelParams, Param


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


def test_jit_with_fixed_param():
    # F13: the second call compares the new pytree's aux_data (holding the
    # fixed array) with the cached one; a raw array there raised.
    f = jax.jit(lambda m: m.s2e.value * 2)
    for s2e in (0.5, 0.7):
        mp = ModelParams(beta=Param("beta", [1.0, 2.0], fixed=True), s2e=s2e)
        np.testing.assert_allclose(np.asarray(f(mp)), [2 * s2e], rtol=1e-6)


def _mixed_params(beta=(1.0, 2.0)):
    """beta and Sigma0 fixed, the rest free."""
    return ModelParams(beta=Param("beta", list(beta), fixed=True), s2e=0.5, f=[0.8], A=[[3.0]],
                       ks=[1.5], x0=[0.0], Sigma0=Param("Sigma0", [[2.0]], fixed=True))


def _loss(m):
    return (jnp.sum(m.beta.value ** 2) * m.s2e.value[0] + m.A.value[0, 0] ** 2
            + m.f.value[0] * jnp.sum(m.Sigma0.value) + jnp.log(m.ks.value[0]))


def test_jit_grad_roundtrip_with_fixed_and_free_params():
    mp = _mixed_params()
    g = jax.jit(jax.grad(_loss))(mp)
    # Free parameters: gradients, identical to the eager ones.
    g_eager = jax.grad(_loss)(mp)
    expected = {"s2e": [5.0], "A": [[6.0]], "f": [2.0], "ks": [1 / 1.5], "x0": [0.0]}
    for name, value in expected.items():
        np.testing.assert_allclose(np.asarray(getattr(g, name).value), value, rtol=1e-6, err_msg=name)
        np.testing.assert_allclose(np.asarray(getattr(g, name).value),
                                   np.asarray(getattr(g_eager, name).value), rtol=1e-6, err_msg=name)
    # Fixed parameters: no leaves, so they come back unchanged and still fixed.
    for name in ("beta", "Sigma0"):
        assert getattr(g, name).fixed
        np.testing.assert_array_equal(np.asarray(getattr(g, name).value), np.asarray(getattr(mp, name).value))
    assert float(jax.jit(_loss)(mp)) == pytest.approx(float(_loss(mp)), rel=1e-6)


def test_flatten_unflatten_roundtrip():
    mp = _mixed_params()
    leaves, treedef = jax.tree.flatten(mp)
    assert len(leaves) == 5  # free values only: s2e, f, A, ks, x0
    back = jax.tree.unflatten(treedef, leaves)
    for name in mp.__dataclass_fields__:
        a, b = getattr(mp, name), getattr(back, name)
        assert (a.name, a.fixed) == (b.name, b.fixed)
        np.testing.assert_array_equal(np.asarray(a.value), np.asarray(b.value))


def test_tree_structure_compares_fixed_values():
    same = jax.tree.structure(_mixed_params()) == jax.tree.structure(_mixed_params())
    different = jax.tree.structure(_mixed_params()) == jax.tree.structure(_mixed_params(beta=(3.0, 4.0)))
    assert same and not different


def test_jit_recompiles_only_when_a_fixed_value_changes():
    f = jax.jit(_loss)
    f(_mixed_params())
    f(_mixed_params())
    assert f._cache_size() == 1
    # A different fixed value is a different constant: recompiled and used.
    out = f(_mixed_params(beta=(3.0, 4.0)))
    assert f._cache_size() == 2
    assert float(out) == pytest.approx(float(_loss(_mixed_params(beta=(3.0, 4.0)))), rel=1e-6)


def test_jit_with_bse_set():
    mp = _mixed_params()
    mp.s2e = Param("s2e", [0.5], bse=[0.01])
    f = jax.jit(lambda m: m.s2e.value * 2)
    np.testing.assert_allclose(np.asarray(f(mp)), [1.0])
    np.testing.assert_allclose(np.asarray(f(mp)), [1.0])


def test_fit_options_defaults():
    o = FitOptions()
    assert (o.max_iter, o.tol_relat, o.force_iter, o.std) == (20, pytest.approx(1e-3), False, True)
