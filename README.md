<p align="center">
  <img src="docs/logo.svg" alt="GEOSSM Logo" width="320"/>
</p>

# GEOSSM
> **Geo**statistics with **S**tate **S**pace **M**odels

**geossm** is a Python package designed to apply **state space models** to **spatially and temporally referenced data**.  It is tailored for modern **geostatistical workflows** and natively operates on `GeoDataFrame` objects from the `geopandas` library.

The package is designed with scalability and modularity in mind, making it suitable for large spatial and spatiotemporal datasets.


The cornerstone of this package relies on the PhD thesis: A State-Space Modelling Framework in Geostatistics with Application to Environmental Data (Jacopo Rodeschini)

---

## 🚀 Installation
*Coming soon.*

The package will be installable via:
```bash
pip install geossm
```

or from source

1. **Download the repository** as a ZIP file named `Low_Rank_State_Space_Model.zip`, and extract it into a folder named `Low_Rank_State_Space_Model`.

2. **Navigate into the folder**:

```bash
cd Low_Rank_State_Space_Model
```

3. **Create the Conda environment** (named `dev`) from the list of required packages:

```bash
conda env create --file=environment.yml --name dev
```

4. **Activate the environment**:

```bash
conda activate dev
```
5. Install the package
```bash
conda install -e
```

---

## 🔍 Key Features
- Seamless integration with `geopandas` for spatial data handling
- State space modeling framework for spatial and spatiotemporal processes
- Modular pipeline:
  - data preprocessing
  - design matrix construction
  - model specification and fitting
- Designed for extensibility and research-oriented experimentation

---

## 📦 Input Format
- Accepts `geopandas.GeoDataFrame` as the primary input
- Compatible with common geospatial formats:
  - Shapefiles
  - GeoJSON
  - GeoPackage
- Supports spatial and temporal indexing through geometry and time columns

