"""Installed-package checks (F29): bundled datasets load with the declared
dependencies, the version has one source, and `import geossm` stays light.
Package structure (F25, Section B P5): public `__all__` and intra-package
imports."""

import ast
import importlib
import pathlib
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


@pytest.mark.parametrize("package", ["geossm", "geossm.ssm", "geossm.stmodel", "geossm.covmodel", "geossm.datasets"])
def test_all_lists_only_public_names(package):
    names = importlib.import_module(package).__all__
    assert [n for n in names if n.startswith("_") and not n.startswith("__")] == []
    assert all(hasattr(importlib.import_module(package), n) for n in names)


def test_modules_import_from_defining_modules():
    # Importing from a package __init__ (e.g. `from geossm import X` inside
    # geossm.stmodel.lrssm) works only because of the import order of the
    # __init__ files (F25): modules import from the module that defines a name.
    root = pathlib.Path(geossm.__file__).parent
    packages = {"geossm"} | {f"geossm.{p.parent.relative_to(root).as_posix().replace('/', '.')}"
                             for p in root.rglob("__init__.py") if p.parent != root}
    offending = []
    for path in root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in packages:
                offending.append(f"{path.relative_to(root)}: from {node.module} import ...")
    assert offending == []
