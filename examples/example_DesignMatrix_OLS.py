#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
@title: Example of estimating the parameters of the fixed effect.
"""
import numpy as np
import geopandas as geodf
from geossm import DesignMatricesBuilder
import geossm.datasets as df
from shapely.geometry import Point

# %% List all availabel dataset

print(df.list_datasets())

# %% Import the Agrimonia dataset

agrimonia, shape = df.load_dataset("agrimonia")
print(agrimonia.columns)

# if you do not want the shapefile associate
# agrimonia, _ = df.load_dataset('agrimonia')

# %% Create the regression matrix

# set a threshold for the PM10 variable to avoid log(0)
agrimonia["AQ_pm10"] = np.where(agrimonia["AQ_pm10"] <= 0, np.nan, agrimonia["AQ_pm10"])

# create the design matrix builder
builder = DesignMatricesBuilder(agrimonia, "np.log(AQ_pm10) ~ 1 + standardize(WE_temp_2m)", verbose=True)

# build the design matrices
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

# Compute X'@ y and X'@ X using einsum for better performance
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
residuals = residuals.reshape(X.shape[0], X.shape[2]) # reshape to (n_samples, n_time_steps)
residuals = residuals.mean(axis=0) # Compute daily mean residuals across all samples

# OLS residuals mean = 0 (by definition)
print(residuals.mean())

# Moving average of residuals across days
window_size = 5  # size of the moving average window (e.g., 5 days)
residuals_ma = np.convolve(residuals, np.ones(window_size)/window_size, mode='valid')


plt.figure(figsize=(12, 6))
plt.plot(residuals_ma)
plt.xlabel('Days')
plt.ylabel('Residuals')
plt.title('Moving Average of Residuals Across Days')
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

