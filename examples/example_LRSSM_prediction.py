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
domain = list(shape.geometry[0].geoms)[0].buffer(0.3)


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

# Remove the outliers (99.9 percentile of the AQ_pm10 variable)

# replace the outlier with NaN
# thr999 = agri_points["AQ_pm10"].quantile(0.999)

# Avoif log-issues problems with the log transformation, we also remove the 0.1 percentile of the AQ_pm10 variable
thr001 = agri_points["AQ_pm10"].quantile(0.001)
index = agri_points[agri_points["AQ_pm10"] <= thr001].index
agri_points.loc[index, "AQ_pm10"] = np.nan


# %% Build the model

model = lrssm(
    agri_points, ["np.log(AQ_pm10) ~ 1 + standardize(Altitude) + standardize(WE_temp_2m)*season + standardize(WE_blh_layer_max) + standardize(WE_tot_precipitation) + (EM_nh3_sum + EM_nox_sum)*season"], 
    verbose=True, domain=[domain]
    )

print(model)

# %% Build the mesh for the AQ_pm10 observed variable
from geossm.covmodel import buildMesh2d
points = model.points[0]
mesh_io, poly = buildMesh2d(points, lowrank=0.75,gradation=1, offset = 1, anchor_spacing=1)
print(mesh_io)

# build the latent covariance function on the mesh, then attach it to the model
cov_fun = SPDEApproxCov(latlon=True).setup(mesh_io, domain=[domain])

# plot the mesh (use the fem_solver utlities)
fig, ax = plt.subplots(figsize=(8, 8))
cov_fun.fem_solver.plot_mesh(ax=ax)
ax.plot(points[:, 0], points[:, 1], "ro", label="Observation Points")
ax.set_title("Mesh for $PM_{10}$")


# %% Estimate the Model (default estimation options)
model = model.setup(cov_fun=[cov_fun])

opt = FitOptions(max_iter=50, tol_relat=1e-3)
results = model.fit(options=opt)
print(results)  # resutls.summary()

# %% Compute the covariance matrix of the estimated parameters

# it's optional since it requires the computation of the hessian
# It can takes a while

results.compute_cov_params()
print(results)  # results.summary()

# %% Residual in-sample residuals (observed - predicted) 

# temp = results.residuals.flatten(order='F')  # Flatten in column-major order to match the time series structure
temp = results.y_hat_list[0].flatten(order='F')
std = np.nanstd(temp)
mean = np.nanmean(temp)

# Plot the histogram of the residuals values
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(temp, bins=30, color='skyblue', edgecolor='black')
ax.set_title("Histogram of Residual $PM_{10}$ Values")
ax.set_xlabel("$PM_{10}$ (µg/m³)")
ax.set_ylabel("Frequency")
ax.axvline(x=mean + 4 * std, color='red', linestyle='--', label='Outlier Thresholds')
ax.axvline(x=mean - 4 * std, color='red', linestyle='--')
ax.legend()

# %% Model prediction 

# Compute the prediction at the agri_grid points (pixel level)
prediction = results.get_prediction(agri_grid, verbose=True)
print(prediction)  # prediction.summary()

# %% histogram of the predicted values 
y_pred = prediction.y_pred_list[0].flatten(order='F')  # shape (n_points, n_time_steps)

fig, ax = plt.subplots(figsize=(8, 6))
# Plot the histogram of the predicted values
ax.hist(y_pred, bins=30, color='skyblue', edgecolor='black')
ax.set_title("Histogram of Predicted $PM_{10}$ Values")
ax.set_xlabel("$PM_{10}$ (µg/m³)")
ax.set_ylabel("Frequency")
# %% Plot the results using imshow (for each time step)
from matplotlib.colors import Normalize

index = np.where(y_pred > np.nanquantile(y_pred, 0.999))  # Get indices of values above the 99.9th percentile
y_pred[index] = np.nan  # Replace values above the 99.9th percentile with NaN
y_pred = y_pred.reshape(prediction.y_pred_list[0].shape, order='F')  # Reshape back to (n_points, n_time_steps)

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

for i, (avg_data, month_name) in enumerate(zip(monthly_avg, month_names)):
    ax = axs[i // 4, i % 4]

    # Get unique sorted coordinates
    x_unique = np.sort(np.unique(prediction.points[0][:, 0]))
    y_unique = np.sort(np.unique(prediction.points[0][:, 1]))
    
    # Create mapping dictionaries
    x_to_idx = {x: idx for idx, x in enumerate(x_unique)}
    y_to_idx = {y: idx for idx, y in enumerate(y_unique)}
    
    # Create 2D grid
    rows, cols = len(y_unique), len(x_unique)
    Z = np.full((rows, cols), np.nan)
    
    # Fill grid
    x_indices = np.array([x_to_idx[x] for x in prediction.points[0][:, 0]])
    y_indices = np.array([y_to_idx[y] for y in prediction.points[0][:, 1]])
    Z[y_indices, x_indices] = avg_data
    
    # Mask pixels outside boundary
    for row_idx in range(rows):
        for col_idx in range(cols):
            x_coord = x_unique[col_idx]
            y_coord = y_unique[row_idx]
            point = Point(x_coord, y_coord)
            if not shape.geometry.iloc[0].contains(point):
                Z[row_idx, col_idx] = np.nan
    
    # Set extent
    extent = [x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()]
    
    # Plot with white background for NaN values
    cmap = plt.cm.YlOrBr.copy()
    cmap.set_bad(color='white')
    
    im = ax.imshow(
        Z,
        cmap=cmap,
        origin="lower",
        extent=extent,
        interpolation="none",
        aspect="auto",
        norm=norm
    )
    shape.boundary.plot(ax=ax, color="black", linewidth=1)

    # Style
    ax.set_title(month_name, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

# Add shared colorbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label="PM10 (µg/m³)")

plt.tight_layout(rect=[0, 0, 0.88, 1])

# %% Export the results to a shapefile
geo_fitted = results.to_geo()
# geo_fitted.to_file("fitted.shp")