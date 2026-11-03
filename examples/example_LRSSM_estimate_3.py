#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
@title: Example of estimating the parameters of a LRSSM from simulated data with random points in a convex domain (regular grid) and a Matern covariance function with nu=1. The model is fitted using the maximum likelihood estimation (MLE) method. 
The example also includes a summary of the variance of the simulated data and the theoretical variance 
of the model, as well as plots of the simulated response variable and the latent state for one point, 
and the simulated response variable for multiple time steps.
"""

# %% Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import polygon
import geopandas as gpd
from datetime import date, timedelta

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["JAX_PLATFORMS"] = "cpu"

# %% Simulate random point in a convex space (regular grind) and plot them

# set the domain as a square [0, 1] x [0, 1]
domain = polygon.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

# points = np.random.uniform(0, 1, (200, 2))
# Create random grid points
n = 100
points = np.random.uniform(0, 1, (n, 2))

# plot the points and the domain
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], color="blue", label="Random Points")
plt.plot(*domain.boundary.xy, color="red", label="Domain Boundary")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Random Points in a Circular Domain")
plt.legend()
plt.axis("equal")
plt.show()


# %% Create the covariance function 
from geossm.covmodel import buildMesh2d
from geossm.covmodel import SPDEApproxCov as matern_spde

# Create the mesh 
mesh_io, poly, info = buildMesh2d(points, domain=domain, offset=0.3,
        gradation=0.5, bandwidth=0.1, lowrank=0.5, return_info=True)

print(mesh_io)
 
# Create the covariance function
cov_fun = matern_spde(latlon=False, nu=1, var=1, rescale=4)
cov_fun = cov_fun.setup(mesh_io, domain=domain)

print(cov_fun.summary())
cov_fun.fem_solver.plot_mesh()

fig, ax = plt.subplots(figsize=(8, 8))
cov_fun.fem_solver.plot_mesh(ax=ax)
ax.plot(points[:, 0], points[:, 1], "ro", label="Observation Points")
ax.set_title("Mesh for $PM_{10}$")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
plt.show()


# %% Build the geopandas dataframe

# Create the spatio-temporal covariates (e.g. temperature and humidity)
n = points.shape[0]
T = 50

omega = 2 * np.pi / 30

# repeat the points T times for the temporal dimension
_points = np.tile(points, (T, 1, 1)).reshape(-1, 2) 

# Create the covariates
t = np.repeat(np.linspace(0, T, num=T), n)
t2m = 4*np.sin(omega * t).reshape(T,n).T  + np.random.normal(0, 1, size=n * T).reshape(n, T)
t2m = t2m.reshape(-1,order='F')

humidity = 2 * (np.sin(omega * t) + 0.5*np.cos(2* omega * t)).reshape(T,n).T  + np.random.normal(0, 1, size=n * T).reshape(n, T)
humidity = humidity.reshape(-1, order='F')

tstart = date(2020, 1, 1)
time = np.sort(np.tile([tstart + timedelta(days=d) for d in range(T)], n))

# create the response variable as a linear combination of the covariates plus some noise
beta = [2, 0.5, 0.3]  # coefficients for the covariates
s2e = 1  # variance of the noise
y = beta[0] + beta[1] * t2m + beta[2] * humidity + np.random.normal(0, np.sqrt(s2e), size=n * T).reshape(n, T).reshape(-1, )


# Create a GeoDataFrame from the points
gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(_points[:, 0], _points[:, 1]))
gdf["temperature"] = t2m
gdf["humidity"] = humidity
gdf['y'] = y
gdf["Time"] = time
gdf.crs = "EPSG:4326"  # Set the coordinate reference system

# %% Build the lrssm and set the covariance function
from geossm import stmodel as st

model = st.LRStateSpaceModel(df=gdf, formulas=None, domain=[domain], 
    verbose=True, backend="gpu")

model = model.setup(cov_fun=[cov_fun])
print(model)

# Set up the model cov. 
# print(model)

# %% Simulate the model
theta = 0.3
ks = np.sqrt(8)/theta
params = st.ModelParams(beta=[3], A=np.array([[1.5]]), s2e=[2], ks=[ks], f=[0.7])

# Print the var. statistics (verbose = True)
y_sim, x_sim, info, tdelta = model.sim(["1"], params=params, stats=True, verbose=True)


# %% Plot the simulated response variable for one point
T = info['T'][0]
pt = np.array(info['points'][0])

times = np.arange(0, T, T//9)  # Select 9 time steps evenly spaced across the simulation
vmin, vmax = y_sim.min(), y_sim.max()  # or compute over the selected times

fig, ax = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)
for i, t in enumerate(times[:-1]):
    row, col = divmod(i, 3)
    im = ax[row, col].scatter(pt[:, 0], pt[:, 1], c=y_sim[:, t], cmap="coolwarm", vmin=vmin, vmax=vmax)

    # ax[row, col].plot(*domain.boundary.xy, color="red")
    ax[row, col].set_title(f"Simulated $PM_{{10}}$ at Time Step {t+1}")

# single shared colorbar for all subplots
cbar = fig.colorbar(im, ax=ax)
plt.show()


# %% Estimate the model parameters from the simulated data (full rank)

# 0) Create the geopandas dataframe with the simulated data
gdf["y_sim"] = y_sim.flatten(order='F')  # Flatten in column-major order to match the time series structure

# 1) Create the covariance matrix
est_cov_fun = matern_spde(latlon=False, nu=1, var=1, rescale=2)
est_cov_fun = est_cov_fun.setup(mesh_io, domain=domain)

# 2) Create the model
model = st.LRStateSpaceModel(
    df=gdf,
    formulas=["y_sim ~ 1"],
    domain=[domain],
    verbose=True)


# 3) Set up the model cov.
model = model.setup(cov_fun=[est_cov_fun])
print(model)

# 4) fit the model 

# %% Debug pls 
opt = st.FitOptions(max_iter=50, tol_relat=1e-5)

results = model.fit(options=opt)


# %% Estimate the LOW-RANK model

# 0) Create mesh with fewer points to reduce the rank of the model (85%)
inx = np.random.choice(points.shape[0], size=85, replace=False)
points_lr = points[inx, :]
lr_mesh_io, poly, info = buildMesh2d(points_lr, domain=domain, 
        offset=0.3, gradation=1, lowrank=0.75, eps=0.05, return_info=True)
print(lr_mesh_io)

# 1) Create the covariance matrix
lr_cov_fun = matern_spde(latlon=False, nu=1, var=1, rescale=2)
lr_cov_fun = lr_cov_fun.setup(lr_mesh_io, domain=domain)
print(lr_cov_fun.summary())

# 2) Create the model
lr_model = st.LRStateSpaceModel(
    df=gdf,
    formulas=["y_sim ~ 1"],
    domain=[domain],
    verbose=True)


# 3) Set up the model cov.
lr_model = lr_model.setup(cov_fun=[lr_cov_fun])
print(lr_model)

# 4) fit the low-rank model 
opt = st.FitOptions(max_iter=50, tol_relat=1e-5)

lr_results = lr_model.fit(options=opt)

# %% Compute the hessian (using the hessian matrix) 
hessian_matrix, tdelta = lr_results.get_hessian()


cov_params= lr_results.compute_cov_params()

print(lr_results)
