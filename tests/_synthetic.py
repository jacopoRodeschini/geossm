"""Synthetic spatio-temporal datasets and meshes for the test suite (all deterministic)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pygmsh
from shapely.geometry import Point, Polygon

#: Square study domain in projected coordinates (EPSG:3857 units).
DOMAIN = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
CRS = "EPSG:3857"


def make_sites(n_sites=25, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(1, 9, size=(n_sites, 2))


def make_panel_gdf(n_sites=25, n_times=40, missing_frac=0.1, seed=0, xy=None):
    """Long-format GeoDataFrame (one row per site x day) with a smooth AR(1) latent field.

    Columns: ``y`` (response, with NaN), ``x`` (covariate), ``site`` (int id),
    ``Time`` (daily datetime64) and Point ``geometry`` in ``CRS``.
    ``y = 2 + 0.5 x + z(s, t) + noise``.
    """
    rng = np.random.default_rng(seed)
    xy = make_sites(n_sites, seed) if xy is None else xy
    times = pd.date_range("2020-01-01", periods=n_times, freq="D")
    x = rng.normal(size=(n_sites, n_times))
    smooth = np.sin(xy[:, 0] / 3) + np.cos(xy[:, 1] / 3)
    z = np.zeros((n_sites, n_times))
    a = rng.normal(size=n_sites) * 0.3
    for t in range(n_times):
        a = 0.8 * a + rng.normal(size=n_sites) * 0.3
        z[:, t] = smooth * np.sin(t / 6) + a
    y = 2.0 + 0.5 * x + z + rng.normal(size=(n_sites, n_times)) * 0.3
    y[rng.uniform(size=y.shape) < missing_frac] = np.nan
    return gpd.GeoDataFrame(
        {
            "y": y.T.ravel(),
            "x": x.T.ravel(),
            "site": np.tile(np.arange(n_sites), n_times),
            "Time": np.repeat(times, n_sites),
        },
        geometry=[Point(*xy[i]) for _ in range(n_times) for i in range(n_sites)],
        crs=CRS,
    )


def make_square_mesh(lo=-3.0, hi=13.0, mesh_size=1.5):
    """meshio triangular mesh of the square [lo, hi]^2 (covers DOMAIN plus a buffer)."""
    with pygmsh.geo.Geometry() as geom:
        geom.add_rectangle(lo, hi, lo, hi, 0.0, mesh_size=mesh_size)
        return geom.generate_mesh()
