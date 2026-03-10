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

builder = DesignMatricesBuilder(Agrimonia, "AQ_pm10 ~ 1 + WE_temp_2m", verbose=True)


# %% build the design matrices
dataset = builder.build(verbose=True)

# print the dataset class
print(dataset)

# %% Observation and covariates matrix

# Observation matrix
y = dataset.y # or dataset.y[:, 0] if you want a 1D array
# Covariates matrix
X = dataset.X # shape (n_samples, n_covariates, n_time_steps)

# %% Linear regression on the average

# Average across time
y_avg = np.nanmean(y, axis=1)

# Average covariates across time
X_avg = np.nanmean(X, axis=2)

# OLS estimates
beta = np.linalg.solve(X_avg.T @ X_avg, X_avg.T @ y_avg)
print("Estimated coefficients:", beta.flatten())

# %% Linear spatio-temporal regression 

mask = ~np.isnan(y)

X_masked = np.where(mask[:, None, :], X, 0)
y_masked = np.where(mask, y, 0)

cond_xy = np.einsum('nbt,nt->b', X_masked, y_masked)
cond_xx = np.einsum('nbt,nct->bc', X_masked, X_masked)

beta = np.linalg.solve(cond_xx, cond_xy)  
print("Estimated coefficients:", beta.flatten())  

