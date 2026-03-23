<p align="center">
  <picture>
    <!-- Dark mode -->
    <source src="docs/images/logo-dark.svg" media="(prefers-color-scheme: dark)">
    <!-- Light mode -->
    <source src="docs/images/logo-white.svg" media="(prefers-color-scheme: light)">
    <!-- Fallback -->
    <img src="docs/images/logo-white.svg" alt="GEOSSM Logo" width="320">
  </picture>
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

### Progect structure
```
geossm/
├── pyproject.toml      <-- All package config
├── environment.yml     <-- For Conda users
├── README.md
├── LICENSE
├── src/                <-- The "Source" folder
│   └── geossm/         <-- The actual package folder
│       ├── __init__.py
|       ├── datasets    <-- Submodule for the dataset
|       ├── ssm         <-- Submodule for the State-Space model
|       ├── stmodels    <-- Submodule for the Spatio-temporal model
│       └── covmodel    <-- Submodule for the covariance functions
|
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


## 🔍 Key Features

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

- **Python**: 3.8 or higher
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

## 🚀 Installation

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

```python
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import numpy as np

import pygmsh
import gmsh
import geopandas as geodf
import geossm.datasets as df
from geossm.stmodel import LRStateSpaceModel as lrssm
from geossm.covmodel import FEMSolver


# %% Load the agrimonia dataset
agri, shape = df.load_dataset('agrimonia')


# %% From .csv to geopandas
ct = np.array([agri.Longitude.to_numpy(), agri.Latitude.to_numpy()]).T
agri['geometry'] = [Point(p[0], p[1]) for p in ct]  # (x,y) = (lat,lon)

agri = geodf.GeoDataFrame(agri, crs=4326)

domain = list(shape.geometry[0].geoms)[0].boundary
buffer = list(domain.buffer(0.3).boundary.geoms)[0]


# %% Build the model
model = lrssm(agri, ['AQ_pm10 ~ 1 + WE_temp_2m'], verbose=True, domain = [Polygon(buffer)])
print(model)


# %% [Utils] build mesh with gmsh
def buildMesh(poly, lc, points, lc_buffer=None, lc_points=1e22):
    with pygmsh.occ.Geometry() as geom:

        if lc_buffer is None:
            lc_buffer = lc

        coords = np.array(poly.buffer(
            lc_buffer).simplify(lc_buffer).exterior.coords[:-1])
        domain = geom.add_polygon(coords, mesh_size=lc_buffer*0.1)

        # 2. Add physical group for the domain surface (good practice)
        geom.add_physical(domain, label="surface_domain")

        # Add points for the boundary
        embedded_tags = []
        for p in points:
            t = gmsh.model.occ.addPoint(p[0], p[1], 0, lc_points)
            embedded_tags.append(t)

        gmsh.model.occ.synchronize()  # Synchronize OCC entities before using them in fields

        # fix the points
        # gmsh.model.mesh.embed(
        #     0, embedded_tags, 2, domain._id)

        gmsh.option.setNumber("Mesh.Algorithm", 6)

        # CRITICAL: Tell Gmsh NOT to force density based on the internal points
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # Allow triangles to be very large
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        # Only limit the absolute minimum to prevent crashes
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.1)

        # 5. Generate
        gmsh.model.mesh.generate(2)

        gmsh.model.mesh.optimize("Laplace2D")
        gmsh.option.setNumber("Mesh.Smoothing", 10)

        # # This allows the optimizer to move nodes more freely
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        mesh = geom.generate_mesh()

    return mesh

# %% Build the mesh for the AQ_pm10 observed variable
points = model.points[0]
mesh_io = buildMesh(buffer, 0.35, points)
print(mesh_io)

# plot the mesh (use the fem_solver utlities)
fem_solver = FEMSolver(mesh_io, [Polygon(buffer)])

# plot the mesh using the utilities 
fig, ax = plt.subplots(figsize=(8, 8))
fem_solver.plot_mesh(ax=ax)

# %% Set up the lrssm model (univiarte latent)

# add the mesh object and the domain where the laten domain is defined
# if None it is assumed to be the same of the observation  
model = model.setup([mesh_io])

# %% Estimate the Model (default estimation options)
results = model.fit()
print(results) # resutls.summary()


# %% Plot the likelihood curve
fig, ax = plt.subplots()
ax.plot(-np.array(results.llf_path[1:]))
ax.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('Log Likelihood')
ax.set_title('Log Likelihood Curve')
ax.grid()
plt.show()

```

## Examples

The [examples/](examples/) directory contains comprehensive notebooks demonstrating:

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

Full API documentation and tutorials are available at:
- **Source Code**: See the [src/geossm/](src/geossm/) directory
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

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

For questions or bug reports, please open an [Issue](../../issues).

### Branching Strategy

This repository follows a simple branching strategy.

**Main Branches**

| Branch | Purpose |
|------|------|
| `main` | Stable production code. Only tested and released versions live here. |
| `develop` | Active development branch where new work is integrated. |

**Working Branches**

| Branch Pattern | Purpose | Example |
|------|------|------|
| `feature/*` | New features or improvements | `feature/add-smoothing` |
| `fix/*` | Bug fixes | `fix/memory-leak` |

After the work is complete, open a **Pull Request into `develop`**.

**Typical Workflow**

1. **Update your local branches**

```bash
git switch main
git pull origin main
git switch develop
git pull origin develop
```

2. Create a branch from `develop`

```bash
git switch -c feature/your-feature (or fix/your-bug)
```

3. Commit your changes

```bash
git add .
git commit -m "Describe your change"
```

4. Push the branch

```bash
git push -u origin feature/your-feature
```

5. Open a Pull Request into `develop`.
   - After review/approval, merge your PR into `develop`
   - `main` will only be updated when a new release is ready

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
**Email**: jacopo.rodeschini@unibg.it

---

<p align="center">
  Made with ❤️ for geospatial data science
</p>
