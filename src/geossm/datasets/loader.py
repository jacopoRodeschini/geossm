"""
Loaders for the example datasets bundled with geossm.

Each dataset is a pair of files packaged under
``geossm/datasets/data`` (a Parquet table of observations) and
``geossm/datasets/shapefiles`` (an optional Shapefile describing the
region the observations cover). See `list_datasets` for the currently
available names.
"""

from importlib import resources
import pandas as pd
import geopandas as geopd

_AVAILABLE = {
    "agrimonia": {"data": "agrimonia.parquet", "geometry": "lombardy"},  # optional
    "aqclim_points": {"data": "GRINS_AQCLIM_points_Italy_2023.parquet", "geometry": "italy_boundary_wgs84"}
}


def load_dataset(name: str, return_geometry: bool = True):
    """
    Load one of the example datasets bundled with geossm.

    Parameters
    ----------
    name : str
        Dataset identifier, e.g. ``"agrimonia"``. See `list_datasets`
        for the full list of names currently bundled with the package.
    return_geometry : bool, default True
        If `True`, also return the dataset's region shapefile (e.g. an
        administrative boundary) as a :class:`geopandas.GeoDataFrame`,
        when one is available for `name`. If `False`, only the
        observations table is returned.

    Returns
    -------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        The dataset's observations, as stored in its Parquet file.
    gdf : geopandas.GeoDataFrame
        The dataset's region shapefile. Only returned when
        `return_geometry` is `True`.

    Raises
    ------
    ValueError
        If `name` is not one of the bundled dataset identifiers.
    """
    if name not in _AVAILABLE:
        raise ValueError(
            f"Unknown dataset '{name}'. " f"Available datasets: {list(_AVAILABLE)}"
        )

    base_data = resources.files("geossm.datasets.data")
    base_shape = resources.files("geossm.datasets.shapefiles")
    dataset = _AVAILABLE[name]

    # ---- Load tabular data ----
    with base_data.joinpath(dataset["data"]).open("rb") as f:
        df = geopd.read_parquet(f)

    if "geometry" in dataset:
        shp_path = base_shape.joinpath(
            f"{dataset['geometry']}/{dataset['geometry']}.shp"
        )
        gdf = geopd.read_file(shp_path)

    if return_geometry:
        return df, gdf
    else:
        return df


def list_datasets():
    """
    List available example datasets bundled with the package.

    Returns
    -------
    str
        Formatted table with dataset name, file extension and size in MB.
    """

    base_data = resources.files("geossm.datasets.data")
    base_shape = resources.files("geossm.datasets.shapefiles")

    header = f"{'Dataset':<15} {'Type':<10} {'Ext':<6} {'Size (MB)':>10}"
    sep = f"{'-' * len(header)}"

    rows = [header, sep]

    # Add a flag if the dataset is larger than 10 MB

    for name in sorted(_AVAILABLE):
        filename = _AVAILABLE[name]["data"]
        path = base_data.joinpath(filename)
        size_mb = path.stat().st_size / 1024**2
        ext = path.suffix.lstrip(".")

        flag = " *" if size_mb > 10 else ""
        rows.append(f"{name:<15} {'data':<10} {ext:<6} {size_mb:>10.3f}{flag}")

    for name in sorted(_AVAILABLE):
        if "geometry" not in _AVAILABLE[name]:
            continue

        filename = _AVAILABLE[name]["geometry"]
        path = base_shape.joinpath(f"{filename}/{filename}.shp")
        size_mb = path.stat().st_size / 1024**2
        ext = path.suffix.lstrip(".")

        flag = " *" if size_mb > 10 else ""
        rows.append(f"{name:<15} {'geometry':<10} {ext:<6} {size_mb:>10.3f}{flag}")

    return "\n".join(rows)
