#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 14:34:05 2026

@author: jacopo
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import geopandas as geodf

# %% import and check the version
import geossm
from geossm import data_preparation as grid
import geossm.datasets as df

from shapely.geometry import Point


# %% List all availabel dataset

print(df.list_datasets())

# %% Import the Agrimonia dataset

Agrimonia, shape = df.load_dataset('agrimonia')
print(Agrimonia.columns)

# if you do not want the shapefile associate
# Agrimonia, _ = df.load_dataset('agrimonia')

# %% From .csv to geopandas

ct = np.array([Agrimonia.Longitude.to_numpy(), Agrimonia.Latitude.to_numpy()]).T
Agrimonia['geometry'] = [Point(p[0], p[1]) for p in ct]  # (x,y) = (lat,lon)

Agrimonia = geodf.GeoDataFrame(Agrimonia, crs=4326)

# %% Crate the regression matrix

dataset = grid(Agrimonia, 'AQ_pm10 ~ 1 + WE_temp_2m')

# print the dataset class
print(dataset)

# Get the design matrix
matrix = dataset()
print(matrix)

# %% Observation and covariates matrix

# Observation matrix
y = matrix.y
# Covariates matrix
X = matrix.X

# %% Do the linear regression on the average across time

# Average across time
y_avg = np.nanmean(y, axis=1)

# Average covariates across time
X_avg = np.nanmean(X, axis=2)

# OLS estimates
beta = np.linalg.solve(X_avg.T @ X_avg, X_avg.T @ y_avg)
print("Estimated coefficients:", beta.flatten())
