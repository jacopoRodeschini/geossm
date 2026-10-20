"""`DesignMatricesBuilder` / `DesignMatrices`: shapes, alignment, validation, prediction reuse."""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from geossm import DesignMatricesBuilder
from geossm.data_preparation import check_regular_timestamps
from tests._printing import assert_printing_is_side_effect_free
from tests._synthetic import make_panel_gdf, make_sites

N, T = 12, 8


@pytest.fixture
def gdf():
    """Panel whose response encodes (site, time): y = 1000 * site + t."""
    g = make_panel_gdf(n_sites=N, n_times=T, missing_frac=0.0, seed=1)
    t_index = np.repeat(np.arange(T), N)
    g["y"] = 1000.0 * g["site"] + t_index
    return g


def _build(gdf, formula="y ~ 1 + x", **kw):
    return DesignMatricesBuilder(gdf, formula, verbose=False, **kw).build()


def test_shapes_and_intercept(gdf):
    dm = _build(gdf)
    assert dm.X.shape == (N, 2, T) and dm.y.shape == (N, T) and dm.points.shape == (N, 2)
    assert (dm.N, dm.b, dm.T) == (N, 2, T)
    np.testing.assert_allclose(np.asarray(dm.X[:, 0, :]), 1.0)
    assert list(dm.x_names) == ["Intercept", "x"]


@pytest.mark.parametrize("formula, name", [("y ~ 1 + x", "y"), ("np.sqrt(y) ~ 1 + x", "np.sqrt(y)")])
def test_response_name_is_a_string(gdf, formula, name):
    # F39: y_name was the one-element list design_info.column_names.
    builder = DesignMatricesBuilder(gdf, formula, verbose=False)
    assert builder.build().y_name == name
    assert builder.build_predict(gdf, verbose=False, include_response=True).y_name == name


def test_printing_design_matrices_has_no_side_effects(gdf):
    # F12: summary() used to set placeholder model/params/bse/... attributes
    dm = _build(gdf)
    assert_printing_is_side_effect_free(dm)
    assert not any(hasattr(dm, a) for a in ("model", "params", "bse", "tvalues", "pvalues"))


def test_rows_points_and_times_are_aligned(gdf):
    dm = _build(gdf)
    xy = make_sites(N, seed=1)
    for i, pt in enumerate(dm.points):
        site = int(np.argmin(np.linalg.norm(xy - pt, axis=1)))
        np.testing.assert_allclose(np.asarray(dm.y[i]), 1000.0 * site + np.arange(T))
        x_site = gdf.loc[gdf.site == site, "x"].to_numpy()
        np.testing.assert_allclose(np.asarray(dm.X[i, 1]), x_site, rtol=1e-6)
    assert np.all(np.diff(dm.timestamps.astype("int64")) > 0)


def test_input_row_order_does_not_matter(gdf):
    a = _build(gdf)
    b = _build(gdf.sample(frac=1.0, random_state=0))
    order_a = np.lexsort(a.points.T)
    order_b = np.lexsort(b.points.T)
    np.testing.assert_allclose(np.asarray(a.y)[order_a], np.asarray(b.y)[order_b])


def test_missing_response_kept_as_nan(gdf):
    gdf.loc[gdf.index[[3, 7, 20]], "y"] = np.nan
    dm = _build(gdf)
    assert int(np.isnan(np.asarray(dm.y)).sum()) == 3
    assert dm.n_obs == N * T - 3


def test_float64_dtype(gdf, x64):
    dm = _build(gdf, dtype=np.float64)
    assert dm.X.dtype == np.float64 and dm.points.dtype == np.float64


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda g: g.set_crs(None, allow_override=True), "CRS"),
        (lambda g: g.drop(index=g.index[5]), "complete site x time grid"),
        (lambda g: g.drop(columns="x"), "x"),
    ],
    ids=["no-crs", "unbalanced-panel", "missing-column"],
)
def test_invalid_inputs_raise(gdf, mutate, match):
    with pytest.raises(ValueError, match=match):
        _build(mutate(gdf))


def test_irregular_timestamps_raise(gdf):
    times = np.sort(gdf["Time"].unique())
    gdf = gdf[gdf["Time"] != times[3]]
    with pytest.raises(ValueError, match="regularly spaced"):
        _build(gdf)


def test_non_geodataframe_raises(gdf):
    with pytest.raises(ValueError, match="GeoDataFrame"):
        _build(pd.DataFrame(gdf.drop(columns="geometry")))


def test_domain_drops_outside_sites(gdf):
    left_half = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])
    dm = _build(gdf, domain=left_half)
    assert 0 < dm.N < N
    assert np.all(dm.points[:, 0] <= 5)


def test_time_window(gdf):
    times = np.sort(gdf["Time"].unique())
    dm = _build(gdf, tmin=times[2], tmax=times[5])
    assert dm.T == 4


def test_build_predict_reuses_training_standardization(gdf):
    builder = DesignMatricesBuilder(gdf, "y ~ standardize(x)", verbose=False)
    train = builder.build()
    x_train = gdf["x"].to_numpy()
    new = gdf.copy()
    new["x"] = new["x"] * 3.0 + 10.0
    pred = builder.build_predict(new.drop(columns="y"), verbose=False)
    assert pred.y is None
    i = 0  # check one site's covariate row against the training mean/std
    site = int(np.argmin(np.linalg.norm(make_sites(N, seed=1) - pred.points[i], axis=1)))
    x_new = new.loc[new.site == site, "x"].to_numpy()
    expected = (x_new - x_train.mean()) / x_train.std()
    np.testing.assert_allclose(np.asarray(pred.X[i, 1]), expected, rtol=1e-4, atol=1e-5)
    assert train.X.shape[1] == pred.X.shape[1]


def test_build_predict_rejects_other_crs(gdf):
    builder = DesignMatricesBuilder(gdf, "y ~ 1 + x", verbose=False)
    builder.build()
    with pytest.raises(ValueError, match="CRS"):
        builder.build_predict(gdf.to_crs("EPSG:4326"), verbose=False)


def test_build_predict_requires_build(gdf):
    builder = DesignMatricesBuilder(gdf, "y ~ 1 + x", verbose=False)
    with pytest.raises(ValueError, match="build"):
        builder.build_predict(gdf, verbose=False)


@pytest.mark.parametrize(
    "freq, expected",
    [("D", (True, 1, "day")), ("h", (True, 1, "hour")), ("MS", (True, 1, "month"))],
)
def test_check_regular_timestamps(freq, expected):
    ts = pd.date_range("2020-01-01", periods=6, freq=freq)
    assert check_regular_timestamps(ts) == expected


def test_check_regular_timestamps_irregular():
    ts = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"])
    assert check_regular_timestamps(ts) == (False, None, None)
