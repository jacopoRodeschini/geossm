#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""
import numpy as np
import geopandas as geodf
from geossm import DesignMatricesBuilder
import geossm.datasets as df
from shapely.geometry import Point

# %% List all availabel dataset

print(df.list_datasets())

# %% Import the Agrimonia dataset

Agrimonia, shape = df.load_dataset("agrimonia")
print(Agrimonia.columns)

# if you do not want the shapefile associate
# Agrimonia, _ = df.load_dataset('agrimonia')

# %% From .csv to geopandas

ct = np.array([Agrimonia.Longitude.to_numpy(), Agrimonia.Latitude.to_numpy()]).T
Agrimonia["geometry"] = [Point(p[0], p[1]) for p in ct]  # (x,y) = (lat,lon)

Agrimonia = geodf.GeoDataFrame(Agrimonia, crs=4326)

# %% Crate the regression matrix

# set a threshold for the PM10 variable to avoid log(0)
Agrimonia["AQ_pm10"] = np.where(Agrimonia["AQ_pm10"] <= 0, np.nan, Agrimonia["AQ_pm10"])

# create the design matrix builder
builder = DesignMatricesBuilder(Agrimonia, "np.log(AQ_pm10) ~ 1 + standardize(WE_temp_2m)", verbose=True)


# %% build the design matrices
dataset = builder.build(verbose=True)

# print the dataset class
print(dataset)

# %% histogram of the target variable
import matplotlib.pyplot as plt 

fig, ax = plt.subplots(figsize=(10, 6))
plt.hist(dataset.y.flatten(), bins=30, color='blue', alpha=0.7)
plt.xlabel('log(PM10)')
plt.ylabel('Frequency')
plt.grid(True, linestyle="--", alpha=0.6)
plt.title('Histogram of log(PM10)')

# %% Observation and covariates matrix

# Observation matrix
y = np.array(dataset.y) # or dataset.y[:, 0] if you want a 1D array
# Covariates matrix
X = np.array(dataset.X) # shape (n_samples, n_covariates, n_time_steps)


# %% Linear spatio-temporal regression 

# mask for missing values in the target variable
mask = ~np.isnan(y)
  
X_masked = np.where(mask[:, None, :], X, 0)
y_masked = np.where(mask, y, 0)

cond_xy = np.einsum('nbt,nt->b', X_masked, y_masked)
cond_xx = np.einsum('nbt,nct->bc', X_masked, X_masked)

# solve for beta using np.linalg.solve for better numerical stability
beta = np.linalg.solve(cond_xx, cond_xy)  
s2e = np.mean((y_masked - np.einsum('nbt,b->nt', X_masked, beta))**2)

std_beta = np.sqrt(np.diag(s2e * np.linalg.inv(cond_xx)))

print("Estimated coefficients:")
for i in range(len(beta)):
    print(f"Beta {i}: {beta[i] - 1.96 * std_beta[i]} - {beta[i] + 1.96 * std_beta[i]}")
  
# %% Plot the residuals moving average across days
residuals = y_masked - np.einsum('nbt,b->nt', X_masked, beta)
residuals = residuals.reshape(-1, X.shape[2])  # reshape to (n_samples, n_time_steps)

# group the residuals by week (assuming daily data and 7 days in a week)
# residuals_by_week = [residuals[:, i:i+3].flatten() for i in range(0, residuals.shape[1], 3)]

# Moving average of residuals across days
window_size = 5  # size of the moving average window (e.g., 3 days)
residuals_ma = [np.mean(residuals[:, i:i+window_size]) for i in range(0, residuals.shape[1], window_size)]


plt.figure(figsize=(12, 6))
plt.plot(residuals_ma)
plt.xlabel('Weeks')
plt.ylabel('Residuals')
plt.title('Boxplot of Residuals Across Weeks')
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
