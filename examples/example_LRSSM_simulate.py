#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:38:28 2026

@author: jacopo
"""
# %% [imports]

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
from datetime import date, timedelta
from geossm.covmodel import SPDEApproxCov, buildMesh2d
from geossm.stmodel import LRStateSpaceModel as lrssm
import geossm.stmodel as st

# %% Simulate random point in a convex space

# set the domain as a circle with radius 10
center = (0, 0)
radius = 1
circle = Point(center).buffer(radius)

points = np.random.uniform(-radius, radius, (500, 2))

# take the mask of the points that are inside the circle
mask = np.array([circle.contains(Point(p)) for p in points])
points = points[mask]

# plot the points and the domain
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], color="blue", label="Random Points")
plt.plot(*circle.boundary.xy, color="red", label="Domain Boundary")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Random Points in a Circular Domain")
plt.legend()
plt.axis("equal")
plt.show()

# %% Create the covariance function

# build the discretisation mesh 
mesh_io, poly = buildMesh2d(points, domain=circle, lowrank=0.75, 
            offset=0.2,bandwidth=0.01)
print(mesh_io)

# build the covariance function using the SPDE approximation
cov_fun = SPDEApproxCov(latlon=False).setup(mesh_io, domain=circle)

# Plot the mesh behind the cov. function
fig, ax = plt.subplots(figsize=(8, 8))
cov_fun.fem_solver.plot_mesh(ax=ax)
ax.plot(points[:, 0], points[:, 1], "ro", label="Observation Points")
ax.set_title("Mesh for $PM_{10}$")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
plt.show()

# %% Build the model (which allows only for geopandas)

def build_gdf(points, T=50):

    # Create the spatio-temporal covariates (e.g. temperature and humidity)
    n = points.shape[0]
   
    # repeat the points T times for the temporal dimension
    _points = np.tile(points, (T, 1, 1)).reshape(-1, 2) 

    d = np.linspace(0, 2 * np.pi, n)

    t2m = np.sin(d).reshape(-1, 1) + np.random.normal(0, 1, size=n * T).reshape(n, T)
    t2m = t2m.reshape(-1, )

    humidity = 1 / 2 * np.sin(d).reshape(-1, 1) + np.random.normal(0, 1, size=n * T).reshape(n, T)
    humidity = humidity.reshape(-1, )

    tstart = date(2020, 1, 1)
    time = np.sort(np.tile([tstart + timedelta(days=d) for d in range(T)], n))

    # Create a GeoDataFrame from the points
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(_points[:, 0], _points[:, 1]))
    gdf["temperature"] = t2m
    gdf["humidity"] = humidity
    gdf["Time"] = time
    gdf.crs = "EPSG:4326"  # Set the coordinate reference system

    return gdf

# %% build the lrssm and set the covariance function

gdf = build_gdf(points, T=50)
model = lrssm(
    df=gdf, 
    formulas=["humidity ~ 1 + temperature", "temperature ~ 1"], 
    domain=[circle, circle], 
    verbose=True)

# Set up the model cov. (univariate)
model = model.setup(cov_fun=[cov_fun])
print(model)

# %% [Case 1] Create the model parameters and simulate the model

params = st.ModelParams(beta=[0.1, 0.5, 2], A=np.array([[3],[2]]), s2e=[2, 2], ks=[20], f=[0.9])

# The simulation does not change the model propoerties
y_sim, x_sim, info, tdelta = model.sim(params=params, stats=True, verbose=True)

print(y_sim.shape)
print(x_sim.shape)

# %% [Case 2] Simulate the model with specific covariates 
theta = 0.3
ks = np.sqrt(8)/theta

beta=[0.1, 0.5, 2, 1.2, 3]
params = st.ModelParams(beta=beta, A=np.array([[3],[2]]), s2e=[2, 2], ks=[ks], f=[0.9])

y_sim, x_sim, info, tdelta = model.sim(
        formulas=["1 + temperature", "1 + humidity + I(humidity ** 2)"], 
        params=params, stats=True, verbose=True)

Xbeta = info['Xbeta']
x_names = info['xbeta_names']

# %% Plot the simulated data for the time = 1, 10, 20

pt = np.array(model.points[0])
block = info['block_p']


fix, ax = plt.subplots(2,3, figsize=(18, 12))
for i, t in enumerate([0, 3, 6, 9, 12, 19]):
    row = i // 3
    col = i % 3
    ax[row, col].scatter(pt[:, 0], pt[:, 1], c=y_sim[block[1]: block[2], t], 
        cmap="viridis", marker="o", s=70, label=f"Time Step {t+1}")
    ax[row, col].plot(*circle.boundary.xy, color="red", label="Domain Boundary")
    ax[row, col].set_xlabel("X-axis")
    ax[row, col].set_ylabel("Y-axis")
    ax[row, col].set_title(f"Simulated $PM_{{10}}$ at Time Step {t+1}")
    ax[row, col].legend()
    ax[row, col].axis("equal")
"""plt.colorbar(
    ax[1, 2].collections[0],
    ax=ax,
    orientation="vertical",
    fraction=0.035,
    pad=0.02,
    label="Simulated $PM_{10}$",
)"""
plt.show()
plt.close()
