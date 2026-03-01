<p align="center">
  <picture>
    <!-- Dark mode -->
    <source srcset="docs/logo-dark.svg" media="(prefers-color-scheme: dark)">
    <!-- Light mode -->
    <source srcset="docs/logo-white.svg" media="(prefers-color-scheme: light)">
    <!-- Fallback -->
    <img src="docs/logo-white.svg" alt="GEOSSM Logo" width="320">
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

### Progect folder
```
geossm/
├── pyproject.toml      <-- All package config
├── environment.yml     <-- For Conda users
├── README.md
├── LICENSE
├── src/                <-- The "Source" folder
│   └── geossm/         <-- The actual package folder
│       ├── __init__.py
│       └── core.py
└── tests/            
```

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
git clone https://github.com/yourusername/geossm.git
cd geossm
```

2. **Create the conda environment** (named `geossm`) with all required packages:
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

### Building a State Space Model

```python
from geossm.ssm import StateSpaceModel
from geossm.stmodel import LRSSM

# Initialize a Low-Rank State Space Model (LRSSM)
model = LRSSM(data=agrimonia_gdf, 
              spatial_column='geometry',
              response_column='your_variable',
              rank=50)

# Fit the model
model.fit()

# Get predictions
predictions = model.predict(agrimonia_gdf)
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
3. Commit your changes (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

For questions or bug reports, please open an [Issue](../../issues).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use GEOSSM in your research, please cite:

```bibtex
@phdthesis{rodeschini2025,
  author = {Rodeschini, Jacopo},
  title = {A State-Space Modelling Framework in Geostatistics with Application to Environmental Data},
  school = {University of Bergamo},
  year = {2025}
}
```

## Contact

**Author**: Jacopo Rodeschini  
**Email**: jacopo.rodeschini@unibg.it

---

<p align="center">
  Made with ❤️ for geospatial data science
</p>
