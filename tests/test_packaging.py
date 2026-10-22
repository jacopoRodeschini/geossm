"""Installed-package checks (F29): bundled datasets load with the declared
dependencies, the version has one source, and `import geossm` stays light."""

import subprocess
import sys

import geopandas as gpd
import pytest

import geossm
from geossm._version import __version__ as source_version


@pytest.mark.parametrize("name", ["agrimonia", "aqclim_points"])
def test_bundled_dataset_loads(name):
    # Needs the package data in the wheel and the Parquet reader (pyarrow),
    # which geopandas does not install by itself.
    df, region = geossm.load_dataset(name)
    assert isinstance(df, gpd.GeoDataFrame) and len(df) > 0
    assert isinstance(region, gpd.GeoDataFrame) and region.crs is not None
    assert name in geossm.list_datasets()


def test_version_has_a_single_source():
    assert geossm.__version__ == source_version


def test_import_does_not_load_plotting():
    # matplotlib is only imported by `plot_mesh` (F29); checked in a fresh
    # interpreter because the test session may have imported it already.
    code = "import sys, geossm; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
