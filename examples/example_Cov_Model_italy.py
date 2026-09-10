#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""

# %% [Imports]
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import geopandas as geodf
import geossm.datasets as df
from geossm.covmodel import FEMSolver
from geossm.covmodel.utils import buildMesh2d, buildMesh2d_density
from shapely import ops

# %% Import the grins dataset

grins_data, _ = df.load_dataset('aqclim_points', return_geometry=True)
points = np.array([[geom.x, geom.y] for geom in grins_data.geometry.unique()])

# compute only the continetal italian (connected mesh)
shapepath = '/home/jrodeschini/geopy_casestudy/geopy_grins/EApplication/Ita_region/Reg01012024_g_WGS84.shp'

sh_italy = geodf.read_file(shapepath)
sh_italy.to_crs(4326, inplace=True)

def getLargestPoly(geometry):
    if isinstance(geometry, Polygon):
        return geometry
    elif isinstance(geometry, MultiPolygon):
        # Return the polygon with the largest area
        # (itarable, key = scalar ordering function)
        return max(geometry.geoms, key=lambda p: p.area)
    else:
        return geometry  # Just in case

# remove minly island
ct_italy = sh_italy.copy()

# get the largest polygon
ct_italy['geometry'] = ct_italy['geometry'].apply(
    lambda geo: getLargestPoly(geo))

# ct_italy.plot()
italy_union = ops.unary_union(ct_italy.geometry)

# Build a low-rank mesh: reduce the vertex budget to ~30% of len(points),
# while keeping more vertices where points are dense and fewer where they
# are sparse
lr = [0.5, 0.75]

# meshes = [buildMesh2d(points, max_edge=0.4, min_edge=0.1, lowrank=r, density_neighbors=8) for r in  lr]

meshes = [buildMesh2d_density(points, lowrank=r, snap_to_points=False, cutoff=0.01) for r in  lr]

for i, (mesh_lr, buffer) in enumerate(meshes):
    print(f"low-rank mesh: {len(mesh_lr.points)} vertices "
        f"(target {round(lr[i] * len(points))})")

# %% Build the mesh for the AQ_pm10 observed variable

fig, ax = plt.subplots(1, len(meshes), figsize=(8, 8))

# Create the FEM solver for the mesh 
for i, (mesh_lr, buffer) in enumerate(meshes):
    fem_solver = FEMSolver(mesh_lr, list(italy_union.geoms))

    # plot the mesh using the utilities
    fem_solver.plot_mesh(ax=ax[i])
    ax[i].plot(points[:, 0], points[:, 1], "x", markersize=2, alpha=0.5)

    # export the figure in pdf
    ax[i].set_xlabel("Longitude")
    ax[i].set_ylabel("Latitude")
    ax[i].set_title(f"Low-rank mesh (r={lr[i]})")
    ax[i].set_aspect('equal', adjustable='box')

# %% Build the covaraince function 
from geossm.covmodel.covmodels import spdeAppoxCov as MaternCov

fig, ax = plt.subplots(1, len(meshes), figsize=(8, 8))

for i, (mesh_lr, buffer) in enumerate(meshes):
    cov_fun = MaternCov(italy_union, latlon=True, nu=1, var=1, rescale=4)
    cov_fun = cov_fun.setup(mesh_lr)

    # print(f"Covariance function for low-rank mesh (r={lr[i]}):")
    # print(cov_fun.summary())

    cov_fun.fem_solver.plot_mesh(ax=ax[i])
    ax[i].plot(points[:, 0], points[:, 1], "x", markersize=2, alpha=0.5)
    # export the figure in pdf
    ax[i].set_xlabel("Longitude")
    ax[i].set_ylabel("Latitude")
    ax[i].set_title(f"Low-rank mesh (r={lr[i]})")
    ax[i].set_aspect('equal', adjustable='box')



 
