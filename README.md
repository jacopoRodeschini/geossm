<p align="center">
  <img src="docs/logo.svgz" alt="GEOSSM Logo" width="220"/>
</p>

# GEOSSM
> **Geo**statistics with **S**tate **S**pace **M**odels

**GEOSSM** is a Python package designed to apply **state space models** to **spatially and temporally referenced data**.  
It is tailored for modern **geostatistical workflows** and natively operates on `GeoDataFrame` objects from the `geopandas` library.

The package is designed with scalability and modularity in mind, making it suitable for large spatial and spatiotemporal datasets.

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

---

## 🚀 Installation
*Coming soon.*

The package will be installable via:
```bash
pip install geossm