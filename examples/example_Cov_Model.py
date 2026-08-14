#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""

# %% [Imports]
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import numpy as np
import pygmsh
import gmsh
import geopandas as geodf
import geossm.datasets as df
from geossm.covmodel import FEMSolver
from geossm.covmodel.utils import buildMesh2d

# %% Load the agrimonia dataset

agri, shape = df.load_dataset("agrimonia")
points = np.array([[geom.x, geom.y] for geom in agri.geometry.unique()])

# %% From .csv to geopandas

domain = list(shape.geometry[0].geoms)[0]

# %% Build the mesh for the AQ_pm10 observed variable

mesh_io, buffer = buildMesh2d(points, boundary=domain.boundary, offset=0.3, 
    max_edge=0.35, min_edge=0.05, lowrank=0.75, density_neighbors=4)
print(mesh_io)

# %% plot the mesh (use the fem_solver utlities)
fem_solver = FEMSolver(mesh_io, [domain])

# plot the mesh using the utilities
fig, ax = plt.subplots(figsize=(8, 8))
fem_solver.plot_mesh(ax=ax)
# export the figure in pdf
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
# plt.savefig("mesh.pdf", bbox_inches="tight")

# %% Create the Mater covariance function 
from geossm.covmodel.covmodels import spdeAppoxCov as MaternCov


covf = MaternCov([domain], latlon=True, nu=1, var=1.0, rescale=1.0)
covf = covf.setup(mesh_io)

print(covf.summary()) 
