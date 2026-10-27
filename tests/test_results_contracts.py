"""Documented contracts of the result objects (Section H): return types, shapes,
dtypes, mutability, lifecycle and statistical semantics.

`StateSpaceResults` (filter/smoother of `StateSpaceModel`) and
`LRStateSpaceResults` (fit, inference, predict, back_transform, to_geo).
Type/shape/dtype checks right after `LRStateSpaceModel.fit` are in
test_lrssm.py (`TestFit`).
"""

import geopandas as gpd
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geossm.ssm import StateSpaceModel, StateSpaceResults
from geossm.stmodel import LRPredictionResults
from tests.test_lrssm import _build_model, _options
from tests._synthetic import make_panel_gdf, make_square_mesh

# Fits deliberately stop at max_iter (tol_relat=1e-12) to keep runtime fixed.
pytestmark = pytest.mark.filterwarnings("ignore:The EM algorithm reached max_iter:RuntimeWarning")

DTYPES = pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["float32", "float64"])


def _ssm(prob, dtype):
    return StateSpaceModel(H=prob.H, R=prob.R, F=prob.F, Q=prob.Q, x0=prob.x0, Sigma0=prob.Sigma0,
                           Xbeta=prob.Xbeta, beta=prob.beta, backend="cpu", dtype=dtype)


@pytest.fixture(scope="module")
def gdf():
    return make_panel_gdf(n_sites=25, n_times=40, missing_frac=0.1, seed=0)


@pytest.fixture(scope="module")
def mesh():
    return make_square_mesh()


@pytest.fixture(scope="module")
def lr_fit(gdf, mesh):
    """float32 fit shared by the read-only tests below (do not mutate)."""
    model = _build_model(gdf, mesh)
    return model, model.fit(options=_options(max_iter=2))


def _fresh_fit(gdf, mesh, dtype=jnp.float32):
    model = _build_model(gdf, mesh, dtype=dtype)
    return model, model.fit(options=_options(max_iter=1))


# --------------------------------------------------------------- StateSpaceResults
@pytest.mark.usefixtures("x64")
class TestStateSpaceResults:
    @DTYPES
    def test_arrays_shapes_dtypes(self, problem, dtype):
        # Class docstring: array attributes are NumPy arrays in the model dtype;
        # shapes from the constructor docstring.
        p, q, T = problem.p, problem.q, problem.T
        y = np.asarray(problem.y, dtype=np.dtype(dtype))
        res = _ssm(problem, dtype).smoother(y)
        shapes = {"y_obs": (p, T), "y_hat": (p, T), "x_filtered": (q, T + 1), "P_filtered": (q, q, T + 1),
                  "x_pred": (q, T + 1), "P_pred": (q, q, T + 1), "K": (q, p), "x_smoothed": (q, T + 1),
                  "P_smoothed": (q, q, T + 1), "P_pred_smoothed": (q, q, T + 1), "S11": (q, q), "S10": (q, q),
                  "S00": (q, q), "residuals": (p, T)}
        for name, shape in shapes.items():
            value = getattr(res, name)
            assert type(value) is np.ndarray and value.shape == shape and value.dtype == np.dtype(dtype), name
        assert isinstance(res.llf, float) and isinstance(res.nobs, int) and isinstance(res.missing, int)
        assert res.dtype == np.dtype(dtype)

    def test_stored_arrays_are_read_only(self, problem):
        # A results object is a snapshot: its stored arrays cannot be modified
        # in place.
        res = _ssm(problem, jnp.float64).smoother(np.asarray(problem.y))
        for name in ("y_obs", "y_hat", "x_smoothed", "P_smoothed", "S11"):
            with pytest.raises(ValueError, match="read-only"):
                getattr(res, name)[0, 0] = 1.0

    def test_float64_data_in_float32_model(self, problem):
        # F43: the kernel received y as passed (scan carry dtype error under x64).
        res = _ssm(problem, jnp.float32).filter(np.asarray(problem.y, dtype=np.float64))
        assert res.y_hat.dtype == np.float32

    def test_update_returns_a_new_object(self, problem_missing, missing_mask):
        # F47: update() rebuilt the object from all public attributes (TypeError).
        res = _ssm(problem_missing, jnp.float64).smoother(np.asarray(problem_missing.y))
        new = res.update(llf=0.0)
        assert isinstance(new, StateSpaceResults) and new is not res
        assert new.llf == 0.0 and res.llf != 0.0
        y_full = np.nan_to_num(res.y_obs)
        new = res.update(y_obs=y_full)
        assert (new.nobs, new.missing) == (y_full.size, 0)
        assert (res.nobs, res.missing) == (int((~missing_mask).sum()), int(missing_mask.sum()))
        assert np.all(np.isfinite(new.residuals)) and np.isnan(res.residuals).any()
        with pytest.raises(AttributeError):
            res.update(not_an_attribute=1)

    def test_predict_and_sim_return_documented_types(self, problem):
        m = _ssm(problem, jnp.float64)
        res = m.smoother(np.asarray(problem.y))
        y_hat, Sigma = m.predict(m.H, res.x_smoothed, res.P_smoothed, m.Xbeta, m.beta)
        assert isinstance(y_hat, jax.Array) and isinstance(Sigma, jax.Array)
        y_t, x_t, stats, tdelta = m.sim(seed=1, stats=False)
        assert y_t.shape == (problem.p, problem.T) and x_t.shape == (problem.q, problem.T + 1)
        assert stats is None and isinstance(tdelta, float)


# --------------------------------------------------------------- LRStateSpaceResults
class TestLRResultsAfterFit:
    def test_stored_arrays_are_read_only(self, lr_fit):
        _, res = lr_fit
        for name in ("y_obs", "y_hat", "Sigma_y_hat", "x_smoothed", "P_smoothed", "theta_hat"):
            arr = getattr(res, name)
            assert not arr.flags.writeable, name

    def test_inference_outputs_before_and_after_hessian_shapes(self, lr_fit):
        _, res = lr_fit
        k = res.n_params
        assert isinstance(k, int) and isinstance(res.df_resid, int) and res.df_resid == res.nobs - k
        assert res.theta_hat.shape == (k,)
        for name in ("bse_vector", "tvalues", "pvalues"):
            value = getattr(res, name)
            assert type(value) is np.ndarray and value.shape == (k,), name
            assert np.all(np.isnan(value)), name  # NaN until compute_cov_params()
        assert res.conf_int().shape == (k, 2)

    def test_llf_path(self, lr_fit):
        _, res = lr_fit
        assert len(res.llf_path) == res.iterations + 1
        assert np.isnan(res.llf_path[0]) and all(isinstance(v, float) for v in res.llf_path)

    def test_per_variable_views_split_the_stacked_arrays(self, lr_fit):
        model, res = lr_fit
        bp = np.asarray(res.block_p)
        for name, stacked in (("y_hat_list", res.y_hat), ("y_obs_list", res.y_obs),
                              ("residuals_list", res.residuals)):
            views = getattr(res, name)
            assert isinstance(views, list) and len(views) == model.nvar, name
            for i, v in enumerate(views):
                np.testing.assert_array_equal(v, stacked[bp[i]:bp[i + 1]])
                assert np.shares_memory(v, stacked), name  # documented as views
        (S,) = res.Sigma_y_hat_list
        assert S.shape == (bp[1] - bp[0], bp[1] - bp[0], res.y_hat.shape[1])

    def test_coverage_is_over_observed_values(self, lr_fit):
        # F44: missing observations were counted as outside their interval.
        _, res = lr_fit
        lo, hi = res.conf_int_y(alpha=0.05, prediction=True)
        obs = ~np.isnan(res.y_obs)
        assert obs.mean() < 1  # the fixture has missing values
        inside = (res.y_obs >= lo) & (res.y_obs <= hi)
        assert res.coverage_probability(0.05) == pytest.approx(inside[obs].mean())
        np.testing.assert_allclose(res.coverage_probability(0.05, which="space"),
                                   [inside[i][obs[i]].mean() for i in range(obs.shape[0])])

    def test_coverage_rejects_invalid_which(self, lr_fit):
        # F48: the message named 'smoothed'/'filtered'.
        _, res = lr_fit
        with pytest.raises(ValueError, match="global"):
            res.coverage_probability(0.05, which="bad")


class TestLRPredictLifecycle:
    """Predictions do not modify the fit results; each `get_prediction` is an
    independent `LRPredictionResults` (Section B, P2)."""

    def test_predictions_are_independent_objects(self, gdf, mesh):
        _, res = _fresh_fit(gdf, mesh)
        newdata = gdf.drop(columns="y")
        a = res.get_prediction(newdata, verbose=False)
        b = res.get_prediction(newdata[newdata["site"] < 10], verbose=False)
        assert isinstance(a, LRPredictionResults) and a is not b
        assert a.y_pred.shape[0] == 25 and b.y_pred.shape[0] == 10 and a.block_p[-1] == 25
        assert a.fitted_state is res.fitted_state
        for name in ("y_pred", "Sigma_y_pred", "points_pred", "y_pred_back_list"):
            assert not hasattr(res, name), name

    @pytest.mark.usefixtures("x64")
    def test_predictions_are_numpy_in_the_model_dtype(self, gdf, mesh):
        # F45: JAX arrays, and y_pred float64 in a float32 fit under x64.
        _, res = _fresh_fit(gdf, mesh, dtype=jnp.float32)
        newdata = gdf.drop(columns="y")
        pred = res.get_prediction(newdata, verbose=False)
        y_pred, var_pred = res.predict(newdata, verbose=False)
        for name, value in (("y_pred", pred.y_pred), ("Sigma_y_pred", pred.Sigma_y_pred),
                            ("var_pred", pred.var_pred), ("predict mean", y_pred), ("predict variance", var_pred)):
            assert type(value) is np.ndarray and value.dtype == np.float32, name
        assert not pred.y_pred.flags.writeable and not pred.Sigma_y_pred.flags.writeable

    def test_back_transform_of_one_prediction_does_not_affect_another(self, gdf, mesh):
        # F46 (stale back-transformed predictions) cannot occur: each
        # prediction keeps its own back-transformed values.
        _, res = _fresh_fit(gdf, mesh)
        newdata = gdf.drop(columns="y")
        a = res.get_prediction(newdata, verbose=False).back_transform(lambda v: v)
        b = res.get_prediction(newdata[newdata["site"] < 10], verbose=False)
        assert a.y_pred_back_list is not None and b.y_pred_back_list is None
        assert not any("back" in c for c in b.to_geo().columns)
        assert res.y_hat_back_list is None  # the fit results are untouched

    def test_to_geo_columns_and_lengths(self, gdf, mesh):
        model, res = _fresh_fit(gdf, mesh)
        geo = res.to_geo()
        p, _, T = model.shape
        assert isinstance(geo, gpd.GeoDataFrame) and len(geo) == p * T
        assert {"point_id", "timestamp", "y_hat_y", "std_hat_y", "geometry"} <= set(geo.columns)
        pred = res.get_prediction(gdf.drop(columns="y"), verbose=False)
        geo = pred.to_geo()
        assert isinstance(geo, gpd.GeoDataFrame) and len(geo) == pred.y_pred.size
        assert {"point_id", "timestamp", "y_pred_y", "std_pred_y", "geometry"} <= set(geo.columns)
        np.testing.assert_allclose(geo["std_pred_y"].to_numpy() ** 2, pred.var_pred.T.ravel(), rtol=1e-5)


def test_lr_sim_returns_states_with_initial_state(gdf, mesh):
    # x_sim has the initial state, like StateSpaceModel.sim: shape (q, T + 1).
    model, res = _fresh_fit(gdf, mesh)
    y_sim, x_sim, info, tdelta = model.sim(params=res.params, verbose=False)
    p, q, T = model.shape
    assert np.shape(y_sim) == (p, T) and np.shape(x_sim) == (q, T + 1)
    assert isinstance(info, dict) and isinstance(tdelta, float)


def test_lr_update_recomputes_parameter_summaries(lr_fit):
    # F47: update() on LRStateSpaceResults; the derived parameter fields and
    # inference caches follow the replaced params.
    _, res = lr_fit
    params = res.params.copy()
    params.f.fixed = True
    new = res.update(params=params)
    assert new.n_params == res.n_params - 1 and new.theta_hat.size == res.theta_hat.size - 1
    assert new.param_fixed[new.param_names.index("f")] and not res.param_fixed[res.param_names.index("f")]


@pytest.mark.usefixtures("x64")
def test_summary_and_repr_name_the_response(problem, lr_fit):
    # F48: the base summary read `y_name` (absent on a plain StateSpaceModel)
    # and `repr` read `yname` (the default "y" on an LRStateSpaceModel).
    res = _ssm(problem, jnp.float64).smoother(np.asarray(problem.y))
    model, lr = lr_fit
    for obj, name in ((res, "y"), (lr, ", ".join(model.y_name))):
        (line,) = [ln for ln in str(obj.summary()).splitlines() if ln.startswith("Dep. Variable")]
        assert line.split()[2] == name
        assert f"Dep. Variable:   {name}" in repr(obj)


def test_back_transform_without_prediction_covariance_warns(gdf, mesh):
    _, res = _fresh_fit(gdf, mesh)
    pred = res.get_prediction(gdf.drop(columns="y"), std=False, verbose=False)
    with pytest.warns(RuntimeWarning, match="predictions have no covariance"):
        assert pred.back_transform(lambda v: v) is pred
    assert pred.y_pred_back_list is None


def test_to_geo_columns_are_in_the_model_dtype(lr_fit):
    _, res = lr_fit
    hat = res.to_geo()
    assert hat["y_hat_y"].dtype == hat["std_hat_y"].dtype == np.dtype(res.dtype)


@pytest.mark.usefixtures("x64")
def test_results_do_not_reference_the_model(problem, lr_fit):
    # Section B, P1: neither result class stores the model.
    res = _ssm(problem, jnp.float64).smoother(np.asarray(problem.y))
    _, lr = lr_fit
    assert not hasattr(res, "model") and not hasattr(lr, "model")
    (line,) = [ln for ln in str(res.summary()).splitlines() if ln.startswith("Model name:")]
    assert "StateSpaceModel" in line
    # Plain StateSpaceModel results have no FittedState: they store the state.
    assert {"x_smoothed", "P_smoothed", "S11", "S10", "S00"} <= set(vars(res))
