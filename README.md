<p align="center">
  <!-- Dark mode -->
  <img src="docs/images/logo-white.svg#gh-dark-mode-only" width="320">

  <!-- Light mode -->
  <img src="docs/images/logo-dark.svg#gh-light-mode-only" width="320">
</p>

# Geossm 🌍
> **Geo**statistics with **S**tate **S**pace **M**odels

**geossm** is a Python package for applying **state space models** to **spatial and spatiotemporal data**. It is tailored for modern **geostatistical workflows** and natively operates on `GeoDataFrame` objects from the `geopandas` library.

The package is designed with **scalability** and **modularity** in mind, making it suitable for large spatial and spatiotemporal datasets across environmental, climate, and geospatial applications. 


## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

State space models (SSMs) are powerful statistical tools for modeling dynamic systems. This package extends their application to **geospatial and spatiotemporal contexts**, enabling:

- Efficient filtering and smoothing of spatial processes
- Low-rank approximations for scalability
- Seamless integration with geospatial data workflows
- Support for complex environmental and climate datasets

The package is built on the research presented in the PhD thesis: *A State-Space Modelling Framework in Geostatistics with Application to Environmental Data* by Jacopo Rodeschini.

### Project structure
```
geossm/
├── pyproject.toml         <-- All package config
├── environment.yml        <-- For Conda users
├── README.md
├── LICENSE
├── src/                   <-- The "Source" folder
│   └── geossm/            <-- The actual package folder
│       ├── __init__.py
│       ├── datasets/      <-- Submodule for the datasets
│       ├── ssm/           <-- Submodule for the State-Space model
│       ├── stmodel/       <-- Submodule for the Spatio-temporal model
│       ├── covmodel/      <-- Submodule for the covariance functions
│       ├── data_preparation.py  <-- Design matrix construction utilities
│       └── utils.py       <-- Shared helper functions
└── tests/
```

### Available Datasets
The **geossm** package includes spatio-temporal datasets for benchmarking and testing different modelling approaches.

- **Agrimonia Dataset**: Fassò, Alessandro, et al.  
*"Agrimonia: a dataset on livestock, meteorology and air quality in the Lombardy region, Italy."*  
Scientific Data 10.1 (2023): 143.

```python
import geossm
import geossm.datasets as datasets

# List available datasets
print(datasets.list_datasets())

# Output
Dataset         Type       Ext     Size (MB)
--------------------------------------------
agrimonia       data       csv        10.551 *
agrimonia       geometry   shp         0.005
```
The `*` symbol indicates dataset more than 10MB.

### How to build the regression dataset

1. Create a GeoPandas DataFrame with `geometry` and `time` columns.
2. Create the `Builder` object.
3. Call the `.build()` method.

<img src="docs/images/workflow_data_process.png" alt="data process" width="320">


## Key Features

- **Seamless GeoDataFrame Integration**: Work directly with `geopandas.GeoDataFrame` objects
- **State Space Modeling**: Tools for building, estimating, filtering, and smoothing spatial processes
- **Low-Rank Approximations**: Efficient handling of large-scale spatial data via LRSSM
- **Modular Pipeline**: 
  - Data preprocessing and validation
  - Design matrix construction
  - Model specification and estimation
  - Prediction and simulation
- **Research-Oriented**: Built for extensibility and experimental workflows
- **Multiple Model Types**: Support for linear time-invariant and time-varying SSMs

## Requirements
- **OS**: Linux or macOS (Windows not currently supported)
- **Python**: 3.10 or higher (required by the `jax` dependency)
- **Key Dependencies**:
  - `geopandas` ≥ 1.1.2 (geospatial data handling)
  - `pandas` ≥ 2.2.2 (data manipulation)
  - `numpy` ≥ 2.2.6 (numerical computing)
  - `scipy` ≥ 1.15.3 (scientific computing)
  - `jax` ≥ 0.6.2 (automatic differentiation & optimization)
  - `statsmodels` ≥ 0.14.6 (statistical modeling)
  - `matplotlib` ≥ 3.9.1 (visualization)
  - Additional spatial & mesh packages: `shapely`, `gmsh`, `meshio`, `pygmsh`, `pyproj`

See [pyproject.toml](pyproject.toml) or [environment.yml](environment.yml) for the complete dependency list.

## Installation

### Option 1: From pip (Recommended)

```bash
pip install geossm
```

### Option 2: From Source with Conda

1. **Clone or download the repository**:
```bash
git clone https://github.com/jacopoRodeschini/geossm.git
cd geossm
```

2. **Create the conda environment** (named `geossm`) with all required packages. 
Before creating the environment, make sure your **Conda installation is updated to the latest version** and configured to use the faster `libmamba` solver (recommended for significantly faster dependency resolution).

- Update Conda (recommended)
```bash
# Enable the faster libmamba solver
conda config --set solver libmamba

# Update conda in the base environment
conda update -n base -c defaults conda
```

- Create the environment
Once Conda is updated, create the environment (named `geossm`) using:

```bash
conda env create -f environment.yml
```

3. **Activate the environment**:
```bash
conda activate geossm
```

4. **Install the package in development mode**:
```bash
pip install -e .
```

### Verify Installation

```python
import geossm
print(geossm.__version__)
```

### Remove the package and the environment
Remove the package
```bash
pip uninstall geossm
```

Remove an entire environment
```bash
conda remove -n geossm --all
```

## Quick Start

### Loading Data

```python
import geossm
import geossm.datasets as datasets

# List available datasets
print(datasets.list_datasets())

# Load the Agrimonia dataset
agrimonia_gdf, shapefile = datasets.load_dataset('agrimonia')
print(agrimonia_gdf.head())
print(agrimonia_gdf.columns)
```

### Building a Low-Rank State Space Model

This walks through the core steps: load data, build a spatial mesh, define a
covariance function, and fit the model. The full runnable script (including
plotting) is available at
[examples/example_LRSSM_build.py](examples/example_LRSSM_build.py).

```python
from shapely.geometry import Point, Polygon
import numpy as np
import geopandas as geodf
import pygmsh
import gmsh

import geossm.datasets as df
from geossm.stmodel import LRStateSpaceModel as lrssm
from geossm.covmodel import spdeAppoxCov

# 1. Load a dataset and turn it into a GeoDataFrame
agri, shape = df.load_dataset('agrimonia')

domain = list(shape.geometry[0].geoms)[0].boundary
buffer = list(domain.buffer(0.3).boundary.geoms)[0]
domain_polygon = [Polygon(buffer)]

# 2. Define the model formula and the observation domain
model = lrssm(agri, ['AQ_pm10 ~ 1 + WE_temp_2m'], verbose=True, domain=domain_polygon)


# 3. Build a mesh over the domain for the latent spatial field with gmsh
def build_mesh(poly, lc, points, lc_points=1e22):
    with pygmsh.occ.Geometry() as geom:
        coords = np.array(poly.buffer(lc).simplify(lc).exterior.coords[:-1])
        surface = geom.add_polygon(coords, mesh_size=lc * 0.1)
        for p in points:
            gmsh.model.occ.addPoint(p[0], p[1], 0, lc_points)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        return geom.generate_mesh()


mesh_io = build_mesh(buffer, lc=0.35, points=model.points[0])

# 4. Build the latent covariance function on the mesh and attach it
cov_fun = spdeAppoxCov(latlon=True).setup(mesh_io, domain=domain_polygon)
model = model.setup(cov_fun=[cov_fun])

# 5. Fit the model
results = model.fit()
print(results.summary())
```

> **Note**: `build_mesh` above is simplified for readability. See
> [examples/example_LRSSM_build.py](examples/example_LRSSM_build.py) for the
> full mesh-generation code (mesh-density tuning options) and plotting
> utilities (`FEMSolver.plot_mesh`).

## Examples

The [examples/](examples/) directory contains runnable scripts demonstrating:

- **Data Loading**: [example_datasets_load.py](examples/example_datasets_load.py) — Load and explore geospatial datasets
- **Grid Operations**: [example_datasets_grid.py](examples/example_datasets_grid.py) — Create and manipulate spatial grids
- **SSM Building**: [example_SSM_build.py](examples/example_SSM_build.py) — Construct basic state space models
- **SSM Estimation**: [example_SSM_estimate.py](examples/example_SSM_estimate.py) — Estimate model parameters
- **Filtering & Smoothing**: [example_SSM_filter.py](examples/example_SSM_filter.py), [example_SSM_smooth.py](examples/example_SSM_smooth.py) — Apply Kalman filter and smoother
- **Low-Rank SSM**: [example_LRSSM_build.py](examples/example_LRSSM_build.py), [example_LRSSM_estimate.py](examples/example_LRSSM_estimate.py), [example_LRSSM_simulate.py](examples/example_LRSSM_simulate.py) — Efficient large-scale modeling
- **Mesh & FEM**: [example_mesh_1.py](examples/example_mesh_1.py), [example_spde_pyfem.py](examples/example_spde_pyfem.py), [example_GP_FEM_1.py](examples/example_GP_FEM_1.py) — Finite element methods integration

Run any example with:
```bash
cd examples
python example_datasets_load.py
```

## Documentation

The full documentation — installation, quick start, user guide, examples,
and API reference — is built with [Sphinx](https://www.sphinx-doc.org/)
and published to GitHub Pages at:
**https://jacopoRodeschini.github.io/geossm/**

- **Module Reference**:
  - `geossm.ssm` — Core state space modeling
  - `geossm.stmodel` — Spatiotemporal models (LRSSM, SSM variants)
  - `geossm.datasets` — Built-in datasets and data loaders
  - `geossm.data_preparation` — Data preprocessing utilities
  - `geossm.covmodel` — Covariance model specifications

For detailed information on specific functions and classes, use Python's built-in help:
```python
import geossm
help(geossm.ssm.StateSpaceModel)
```

### Building the docs locally

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/build/html
python -m http.server --directory docs/build/html 8000  # then open http://localhost:8000
```

`docs/build/` is generated output and is not committed (see
[.gitignore](.gitignore)). A GitHub Actions workflow
([.github/workflows/docs.yml](.github/workflows/docs.yml)) builds the docs
on every push/PR touching `docs/`, `src/`, or `examples/`, and deploys them
to GitHub Pages on pushes to `main`.

## Contributing

Contributions are welcome! In short: fork the repository, create a branch off
`develop` (`feature/your-feature` or `fix/your-bug`), and open a Pull Request
into `develop`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide, including the
branching strategy, local dev setup, and the PR checklist. For questions or
bug reports, please open an [Issue](../../issues).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## References

- Rodeschini, J. (2025). *A State-Space Modelling Framework in Geostatistics with Application to Environmental Data*. PhD Thesis, University of Bergamo.

- Rodeschini, J., Tedesco, L., Finazzi, F., Otto, P., & Fassò, A. (2025). *Multivariate Low-Rank State-Space Model with SPDE Approach for High-Dimensional Data*. Spatial Statistics.

## Citation

If you use **GEOSSM** in your research, please cite:

```bibtex
@article{rodeschini2025multivariate,
title = {Multivariate low-rank state–space model with SPDE approach for high-dimensional data},
author = {Jacopo Rodeschini and Lorenzo Tedesco and Francesco Finazzi and Philipp Otto and Alessandro Fassò},
journal = {Spatial Statistics},
volume = {73},
pages = {100971},
year = {2026},
issn = {2211-6753},
doi = {https://doi.org/10.1016/j.spasta.2026.100971},
url = {https://www.sciencedirect.com/science/article/pii/S2211675326000199},
}
```

## Contact

**Author**: Jacopo Rodeschini  

---

<p align="center">
  Made with ❤️ for geospatial data science
</p>
