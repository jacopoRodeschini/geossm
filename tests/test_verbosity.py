"""Progress messages and `verbose` (Section O): quiet by default, one flag per
call (`None`: the object's or the fit's verbosity), and `fit()` equal to
`fit(FitOptions())` (F17)."""

import jax.numpy as jnp
import numpy as np
import pytest

from geossm.covmodel import SPDEApproxCov
from geossm.stmodel import FitOptions, LRStateSpaceModel
from tests._synthetic import DOMAIN, make_panel_gdf, make_square_mesh


@pytest.fixture(scope="module")
def gdf():
    return make_panel_gdf(n_sites=12, n_times=15, missing_frac=0.1, seed=0)


@pytest.fixture(scope="module")
def mesh():
    return make_square_mesh()


def _model(gdf, mesh, **kw):
    model = LRStateSpaceModel(gdf, ["y ~ 1 + x"], domain=[DOMAIN], backend="cpu", **kw)
    cov_kw = {k: v for k, v in kw.items() if k == "verbose"}
    return model.setup([SPDEApproxCov(latlon=False, **cov_kw).setup(mesh, domain=[DOMAIN])])


def _lines(capsys):
    return [line for line in capsys.readouterr().out.splitlines() if line.strip()]


def test_quiet_by_default(gdf, mesh, capsys):
    model = _model(gdf, mesh)
    res = model.fit(options=FitOptions(max_iter=2))
    res.compute_cov_params()
    res.get_prediction(gdf)
    res.predict(gdf)
    model.sim(params=res.params)
    assert _lines(capsys) == []


def test_verbose_model_prints_and_calls_override(gdf, mesh, capsys):
    model = _model(gdf, mesh, verbose=True)
    assert _lines(capsys)  # construction and setup
    res = model.fit(options=FitOptions(max_iter=2, verbose=False))
    assert _lines(capsys) == [] and model.verbose is True  # per call, not persisted
    res = model.fit(options=FitOptions(max_iter=2))  # None: the model's verbosity
    assert any("log-likelihood" in line.lower() or "iteration" in line.lower() for line in _lines(capsys))
    res.compute_cov_params()
    assert any("Hessian" in line for line in _lines(capsys))  # follows the fit's verbosity
    res.get_prediction(gdf, verbose=False)
    assert _lines(capsys) == []
    res.get_prediction(gdf)
    assert _lines(capsys)


def test_prediction_verbose_override_on_a_quiet_fit(gdf, mesh, capsys):
    res = _model(gdf, mesh).fit(options=FitOptions(max_iter=2))
    res.predict(gdf, verbose=True)
    assert any("Predicting" in line for line in _lines(capsys))


def test_fit_without_options_equals_default_options(gdf, mesh):
    # F17: options=None used max_iter=100, std=False, unlike FitOptions()
    a = _model(gdf, mesh, dtype=jnp.float32).fit()
    b = _model(gdf, mesh, dtype=jnp.float32).fit(options=FitOptions())
    assert a.llf == b.llf and len(a.llf_path) == len(b.llf_path)
    np.testing.assert_array_equal(np.asarray(a.params.beta.value), np.asarray(b.params.beta.value))
    assert (a.Sigma_y_hat is None) == (b.Sigma_y_hat is None) is False  # std=True in both


def test_variance_report_is_returned_not_printed(gdf, mesh, capsys):
    model = _model(gdf, mesh)
    res = model.fit(options=FitOptions(max_iter=1))
    _, _, info, _ = model.sim(params=res.params, stats=True)
    assert isinstance(info["stats"], dict) and _lines(capsys) == []
