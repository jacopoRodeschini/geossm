#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""
# %% [Imports]
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geossm.datasets as df
from geossm.stmodel import LRStateSpaceModel as lrssm
from geossm.covmodel import SPDEApproxCov, buildMesh2d


# %% List the available dataset in the geossm packace

print(df.list_datasets())

# %% Load the agrimonia dataset

agri, shape = df.load_dataset("agrimonia")
domain = list(shape.geometry[0].geoms)[0].buffer(0.3)
# %% Plot the AQ_pm10 dataset and the domain

mean_pm10_scatter = agri.groupby(["IDStations"]).agg(
    {"AQ_pm10": "mean", "geometry": lambda x: x.iloc[0]}
)


x_domain, y_domain = domain.exterior.coords.xy
fig, ax = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)

# Ensure same color scale across both plots
vmin = mean_pm10_scatter["AQ_pm10"].min()
vmax = mean_pm10_scatter["AQ_pm10"].max()

# --- PM10 ---
sc1 = ax.scatter(
    mean_pm10_scatter.geometry.values.x,
    mean_pm10_scatter.geometry.values.y,
    c=mean_pm10_scatter["AQ_pm10"],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)

ax.plot(x_domain, y_domain, "b--", label="Domain")
ax.set_title("Annual Mean $PM_{10}$")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.6)


cbar = fig.colorbar(sc1, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
cbar.set_label(r"$\mu g / m^3$", fontsize=12)
plt.show()


# %% Build the LRSSM model

agri['AQ_pm10'] = np.where(agri['AQ_pm10'] <= 0.1, np.nan, agri['AQ_pm10'])
model = lrssm(agri, ["np.log(AQ_pm10) ~ 1 + standardize(WE_temp_2m)"], 
        verbose=True, domain=[domain])

print(model)

# %% Build the mesh for the AQ_pm10 observed variable

points = model.points[0]
mesh_io, poly, info = buildMesh2d(points, domain=domain, offset=0.3, 
    gradation=1,lowrank=0.75, eps=0.05, return_info=True)

print(mesh_io)
print(info['min_angle'])
print(info['h_max_D'], info['h_min_D'])


# %% plot the mesh (use the fem_solver utlities)
from geossm.covmodel import FEMSolver

fem_solver = FEMSolver(mesh_io, [domain])

# plot the mesh using the utilities
fig, ax = plt.subplots(figsize=(8, 8))
fem_solver.plot_mesh(ax=ax)
ax.plot(points[:, 0], points[:, 1], "ro", label="Observation Points")
ax.set_title("Mesh for $PM_{10}$")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
# plt.show()


# %% Set up the lrssm model (univiarte latent)

# build the latent covariance function on the mesh, then attach it to the model
cov_fun = SPDEApproxCov(latlon=True).setup(mesh_io, domain=domain)
model = model.setup(cov_fun=[cov_fun])
print(model)
print(model.backend)
print(model.Xbeta.device)
