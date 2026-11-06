#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:38:28 2026

@author: jacopo
"""

# %% [imports] 

# set environment variables for JAX
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", 
    os.path.expanduser("~/.cache/jax"))  # reuse compiled kernels
# os.environ["JAX_PLATFORMS"] = "cpu"

import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import pandas as pd
import geopandas as geodf
from geossm import datasets as df
from geossm.stmodel import LRStateSpaceModel as lrssm
from geossm.stmodel import FitOptions
from geossm.covmodel import SPDEApproxCov
from pathlib import Path


# %% Download the the GRIDDED agrimonia dataset from zenodo repository
# This dataset is useful to make prediction at pixel level 

# link: https://zenodo.org/records/7956006

# Set the path
agri_path = Path('~').expanduser() / Path('Downloads/AGC_Dataset_v_3_0_0.csv')

# Load the file
agri_grid = pd.read_csv(agri_path, na_values=[
                    "      NaN", "        NA", "", "NA", "null", "-", '       NA'])
agri_grid['time'] = pd.to_datetime(agri_grid['Time'], format="%Y-%m-%d")

# Remove columns with object type (non-numeric) to avoid issues during prediction
cols_to_remove = agri_grid.select_dtypes(include=["object"]).columns
agri_grid = agri_grid.drop(columns=cols_to_remove)

# Create Point(lat,lon) for each observation
ct = np.array([agri_grid.Longitude.to_numpy(), agri_grid.Latitude.to_numpy()]).T
agri_grid["geometry"] = [Point(p[0], p[1]) for p in ct]  # (x,y) = (lat,lon)

# Create the geopandas dataset
agri_grid = geodf.GeoDataFrame(agri_grid, crs=4326)

# %% Load the point-wise agrimonia dataset

# Import the Lombardy shape file
agri_points, shape = df.load_dataset("agrimonia")

# project to meters (EPSG:32632) for distance-based operations
agri_points = agri_points.to_crs(epsg=32632)
shape = shape.to_crs(epsg=32632)

domain = list(shape.geometry[0].geoms)[0].buffer(27000)


# %% Add season dummy 
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

agri_points["season"] = agri_points["time"].apply(get_season)
agri_grid["season"] = agri_grid["time"].apply(get_season)

# %% Cut the dataset from 2019 to 2020
tmin = np.datetime64("2019-01-01")
tmax = np.datetime64("2020-12-31")
agri_points = agri_points[(agri_points["time"] >= tmin) & (agri_points["time"] <= tmax)].copy()

# Avoid log-issues problems with the log transformation, we also remove the 0.1 percentile of the AQ_pm10 variable
thr001 = agri_points["AQ_pm10"].quantile(0.001)
index = agri_points[agri_points["AQ_pm10"] <= thr001].index
agri_points.loc[index, "AQ_pm10"] = np.nan

thr001_pm2_5 = agri_points["AQ_pm25"].quantile(0.001)
index_pm2_5 = agri_points[agri_points["AQ_pm25"] <= thr001_pm2_5].index
agri_points.loc[index_pm2_5, "AQ_pm25"] = np.nan


# %% Build the model
import jax.numpy as jnp
formulas = ["np.log(AQ_pm10) ~ 1 + standardize(Altitude) + season + standardize(WE_temp_2m) + standardize(WE_blh_layer_max) + standardize(WE_tot_precipitation) + standardize(EM_nh3_sum) + standardize(EM_nox_sum)", 
"np.log(AQ_pm25) ~ 1 + standardize(Altitude) + season + standardize(WE_temp_2m) + standardize(WE_blh_layer_max) + standardize(WE_tot_precipitation) + standardize(EM_nh3_sum) + standardize(EM_nox_sum)"]

model = lrssm(
    agri_points, formulas, 
    verbose=True, domain=[domain, domain], dtype=jnp.float32, backend="cpu")

print(model)

# %% Build the mesh for the AQ_pm10 observed variable
from geossm.covmodel import buildMesh2d

names = ["AQ_pm10", "AQ_pm25"]
cov_funs = {(name): None for name in names}

infos = []
for pt, n in zip(model.points, names):
    points = pt
    mesh_io, poly, info = buildMesh2d(points, lowrank=0.75, gradation=1, 
        domain = domain, offset=10000, anchor_spacing=.5, return_info=True)
    print(mesh_io)
    infos.append(info)

    # build the latent covariance function on the mesh, then attach it to the model
    cov_fun = SPDEApproxCov(latlon=False).setup(mesh_io, domain=[domain])
    cov_funs[n] = cov_fun

# %% plot the mesh (use the fem_solver utlities)
fig, ax = plt.subplots(1,2, figsize=(16, 8))
cov_funs["AQ_pm10"].fem_solver.plot_mesh(ax=ax[0])
ax[0].plot(model.points[0][:, 0], model.points[0][:, 1], "ro", label="Observation Points")
ax[0].set_title("Mesh for $PM_{10}$")

cov_funs["AQ_pm25"].fem_solver.plot_mesh(ax=ax[1])
ax[1].plot(model.points[1][:, 0], model.points[1][:, 1], "ro", label="Observation Points")
ax[1].set_title("Mesh for $PM_{2.5}$")

# %% Estimate the Model (default estimation options)
model = model.setup(cov_fun=[cov_funs["AQ_pm10"], cov_funs["AQ_pm25"]])

opt = FitOptions(max_iter=50, tol_relat=1e-3)
results = model.fit(options=opt)
print(results)  # resutls.summary()


# %% Compute the covariance matrix of the estimated parameters
# it's optional since it requires the computation of the hessian
# It can takes a while
results.compute_cov_params()
print(results)  # results.summary()
print(results.to_latex())

# %% Model prediction 
import jax.numpy as jnp
# Compute the prediction at the agri_grid points (pixel level)
agri_grid = agri_grid.to_crs(epsg=32632)  # project to meters (EPSG:32632) for distance-based operations
prediction = results.get_prediction(agri_grid, verbose=True)
prediction = prediction.back_transform(g_invs=[jnp.exp, jnp.exp])  # back-transform the predicted values to the original scale

print(prediction)  # prediction.summary()

# %% histogram of the predicted value

y_pred = prediction.y_pred_back_list[0].flatten(order='F')  # shape (n_points, n_time_steps)

fig, ax = plt.subplots(figsize=(8, 6))
# Plot the histogram of the predicted values
ax.hist(y_pred, bins=30, color='skyblue', edgecolor='black')
ax.set_title("Histogram of Predicted $PM_{10}$ Values")
ax.set_xlabel("$PM_{10}$ (µg/m³)")
ax.set_ylabel("Frequency")
# %% Plot the results using imshow (for each time step)
from matplotlib.colors import Normalize
from pyproj import Transformer

index = np.where(y_pred > np.nanquantile(y_pred, 0.999))  # Get indices of values above the 99.9th percentile
y_pred[index] = np.nan  # Replace values above the 99.9th percentile with NaN
y_pred = y_pred.reshape(prediction.y_pred_list[0].shape, order='F')  # Reshape back to (n_points, n_time_steps)
# y_pred = y_pred  # Apply the exponential transformation to revert the log transformation
  # Replace values above the 99.9th percentile with NaN
# Define month boundaries (assuming daily data for a year)
# Days: Jan(31), Feb(28), Mar(31), Apr(30), May(31), Jun(30), 
#       Jul(31), Aug(31), Sep(30), Oct(31), Nov(30), Dec(31)

month_ranges = [
    (0, 31, "January"),
    (31, 59, "February"),
    (59, 90, "March"),
    (90, 120, "April"),
    (120, 151, "May"),
    (151, 181, "June"),
    (181, 212, "July"),
    (212, 243, "August"),
    (243, 273, "September"),
    (273, 304, "October"),
    (304, 334, "November"),
    (334, 365, "December")
]

# Calculate monthly averages
monthly_avg = []
month_names = []
for start, end, name in month_ranges:
    avg = np.nanmean(y_pred[:, start:end], axis=1)
    monthly_avg.append(avg)
    month_names.append(name)

# Plot
fig, axs = plt.subplots(3, 4, figsize=(14, 10))

# Create normalization for consistent coloring
vmin = np.nanmin([np.nanmin(m) for m in monthly_avg])
vmax = np.nanmax([np.nanquantile(m, 0.995) for m in monthly_avg])
norm = Normalize(vmin=vmin, vmax=vmax)


pred_crs = prediction.crs if prediction.crs is not None else agri_grid.crs
pts = np.asarray(prediction.points[0])
to_ll = Transformer.from_crs(pred_crs, 4326, always_xy=True)
to_xy = Transformer.from_crs(4326, pred_crs, always_xy=True)

lon, lat = to_ll.transform(pts[:, 0], pts[:, 1])
lon, lat = np.round(lon, 6), np.round(lat, 6)  # remove round-trip noise


def _regular_axis(c):
    """Regular axis (centres) covering c, step = median spacing of unique values."""
    u = np.unique(c)
    step = np.median(np.diff(u))
    n = int(np.rint((u[-1] - u[0]) / step)) + 1
    return u[0] + step * np.arange(n), step


lon_c, dlon = _regular_axis(lon)
lat_c, dlat = _regular_axis(lat)
col_idx = np.rint((lon - lon_c[0]) / dlon).astype(int)
row_idx = np.rint((lat - lat_c[0]) / dlat).astype(int)

# Cell corners in lon/lat, then projected to the plotting CRS
lon_e = np.append(lon_c - dlon / 2, lon_c[-1] + dlon / 2)
lat_e = np.append(lat_c - dlat / 2, lat_c[-1] + dlat / 2)
X_e, Y_e = to_xy.transform(*np.meshgrid(lon_e, lat_e))

# Mask of the pixels whose centre lies outside the domain. The test must be done
# in the same CRS as the shape: project the cell centres (lon/lat) to pred_crs.
import shapely

X_c, Y_c = to_xy.transform(*np.meshgrid(lon_c, lat_c))
domain_geom = shape.to_crs(pred_crs).geometry.union_all()
outside = ~shapely.contains_xy(domain_geom, X_c, Y_c)  # shape (n_lat, n_lon)

# Plot with white background for NaN values
cmap = plt.cm.YlOrBr.copy()
cmap.set_bad(color='white')

for i, (avg_data, month_name) in enumerate(zip(monthly_avg, month_names)):
    ax = axs[i // 4, i % 4]

    # Fill the (lat, lon) raster
    Z = np.full((lat_c.size, lon_c.size), np.nan)
    Z[row_idx, col_idx] = avg_data

    Z[outside] = np.nan  # mask pixels outside the domain

    im = ax.pcolormesh(
        X_e, Y_e, np.ma.masked_invalid(Z),
        cmap=cmap,
        norm=norm,
        shading="flat",
    )
    shape.boundary.plot(ax=ax, color="black", linewidth=1)
    ax.set_aspect("equal")

    # Style
    ax.set_title(month_name, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

# Add shared colorbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label="PM10 (µg/m³)")

plt.tight_layout(rect=[0, 0, 0.88, 1])
