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
from shapely import Point

# %% import and check the version
import geossm

# print("Version: ", geossm.__version__)
print("Load from: ", geossm.__file__)

# %% Create the State Space Model (SSM)
# Type = linear time-invariant SSM

if geossm.__file__:
    import geossm.datasets as df


# %% List all availabel dataset

print(df.list_datasets())

# %% Import the Agrimonia dataset

agrimonia, shape = df.load_dataset("agrimonia")
print(type(agrimonia))
print(agrimonia.columns)
agrimonia.crs

# %% Aggregate the monthly data to get the average yield per location

# convert to categorical
agrimonia['IDStations'] = agrimonia['IDStations'].astype('category')

mean_pm10_space = agrimonia.groupby(["IDStations"]).agg(
    {"AQ_pm10": "mean", "geometry": lambda x: x.iloc[0]}
)

mean_pm25_space = agrimonia.groupby(["IDStations"]).agg(
    {"AQ_pm25": "mean", "geometry": lambda x: x.iloc[0]}
)

# %% Plot monthly dataset (scatter plot)

# --- Create subplots (1 row, 2 columns) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# Ensure same color scale across both plots
vmin = min(mean_pm10_space["AQ_pm10"].min(), mean_pm25_space["AQ_pm25"].min())
vmax = max(mean_pm10_space["AQ_pm10"].max(), mean_pm25_space["AQ_pm25"].max())

# --- PM10 ---
sc1 = axes[0].scatter(
    mean_pm10_space.geometry.values.x,
    mean_pm10_space.geometry.values.y,
    c=mean_pm10_space["AQ_pm10"],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)
axes[0].set_title("Annual Mean $PM_{10}$")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].grid(True, linestyle="--", alpha=0.6)

# --- PM2.5 ---
sc2 = axes[1].scatter(
    mean_pm25_space.geometry.values.x,
    mean_pm25_space.geometry.values.y,
    c=mean_pm25_space["AQ_pm25"],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)
axes[1].set_title("Annual Mean $PM_{2.5}$")
axes[1].set_xlabel("Longitude")
axes[1].grid(True, linestyle="--", alpha=0.6)

# --- One shared colorbar ---
cbar = fig.colorbar(sc2, ax=axes, orientation="vertical", fraction=0.035, pad=0.02)
cbar.set_label(r"$\mu g / m^3$", fontsize=12)
