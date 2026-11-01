#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
@title: Example of estimating the parameters of a Covariance Model for study the 
Air quality in Italy (PM10) using the Agrimonia dataset.

The example is run on a CPU device.
"""

# %% [Imports]
from matplotlib.pyplot import title
import matplotlib.pyplot as plt
import numpy as np
import geossm.datasets as df
from geossm.covmodel import FEMSolver
from geossm.covmodel.utils import buildMesh2d

# %% Load the agrimonia dataset

agri, shape = df.load_dataset("agrimonia")

# Get the unique points from the dataset
points = np.array([[geom.x, geom.y] for geom in agri.geometry.unique()])

# Create the domain for the mesh using the shapefile geometry
domain = list(shape.geometry[0].geoms)[0]

# Buffer the domain to ensure that the mesh covers the area of interest
domain = domain.buffer(0.3)

# %% Plot the domain and the points
x, y = domain.exterior.xy

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, color='black', linewidth=1, label='Domain')
ax.scatter(points[:, 0], points[:, 1], color='red', s=10, label='Points')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Domain and Points")
ax.legend()


# %% Build the mesh for the AQ_pm10 observed variable

"""
Build the mesh for the AQ_pm10 observed variable
parameters:
- points    = observed locations
- lowrank   = ratio betwen the observed points and the number of vertices in the mesh,
- domain    = the domain of the mesh (the area of interest),
- bandwidth = Gaussian kernel bandwidth `b`. Defaults to Scott's rule,
- gradation = Smaller values give smoother transitions.
- offset    = Buffer added around `D`'s convex hull.
"""

# Use help(buildMesh2d) to see the parameters of the buildMesh2d function

mesh_io, poly, info = buildMesh2d(points, domain=domain, offset=0.3, 
    gradation=0.5, bandwidth=0.1, lowrank=0.5, return_info=True)

print(mesh_io)
print(info['min_angle'])
print(info['h_max_D'], info['h_min_D'])


# %% plot the mesh using the fem_solver utilities

# Create the FEM solver
fem_solver = FEMSolver(mesh_io, [domain], verbose=True)

fig, ax = plt.subplots(figsize=(8, 8))
fem_solver.plot_mesh(ax=ax)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
ax.set_title("Mesh and Points")

# %% Create the Mater covariance function 
from geossm.covmodel import SPDEApproxCov as MaternCov

# Use help(MaternCov) to see the parameters of the MaternCov class

# nu != 1, and var != 1.0 are not valid for the SPDE approximation of the Matern covariance function
covf = MaternCov(latlon=True, nu=1, var=1.0, rescale=1.0) 
covf = covf.setup(mesh_io, domain=domain)
print(covf.summary()) 
