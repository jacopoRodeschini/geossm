#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""
# %% Import libraries
import numpy as np
import geopandas as geodf
from geossm import DesignMatricesBuilder as grid
import geossm.datasets as df
from shapely.geometry import Point

# %% List all availabel dataset

print(df.list_datasets())

# %% Import the Agrimonia dataset

Agrimonia, shape = df.load_dataset("agrimonia")
print(Agrimonia.columns)

# if you do not want the shapefile associate
# Agrimonia, _ = df.load_dataset('agrimonia')

# %% Crate the regression matrix

builder = grid(Agrimonia, "np.sqrt(AQ_pm10) ~ 1 + WE_temp_2m")

# Get the design matrix
dataset = builder.build()
print(dataset)

# %% Observation and covariates matrix

# Observation matrix
y = dataset.y
# Covariates matrix
X = dataset.X

# %% Do the linear regression on the average across time

# Average across time
y_avg = np.nanmean(y, axis=1)

# Average covariates across time
X_avg = np.nanmean(X, axis=2)

# OLS estimates
beta = np.linalg.solve(X_avg.T @ X_avg, X_avg.T @ y_avg)
print("Estimated coefficients:", beta.flatten())
