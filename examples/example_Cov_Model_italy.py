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
from geossm.covmodel.utils import buildMesh2d
from shapely import ops

# %% Import the grins dataset

grins_data, shape = df.load_dataset('aqclim_points', return_geometry=True)
points = np.array([[geom.x, geom.y] for geom in grins_data.geometry.unique()])

# shape = MULTILINE string
print(shape)


# %% Fix the shape (MULTILINESTRINGS) into a list of poligons

from shapely.geometry import MultiLineString
from shapely.ops import polygonize

shape["geometry"] = shape.geometry.apply(
    lambda geom: MultiLineString(
        sorted(geom.geoms, key=lambda g: g.length, reverse=True)[:3]
    )
    if geom.geom_type == "MultiLineString"
    else geom
)

# create the union of the polygons in the shape dataframe
italy_union = ops.unary_union(shape.geometry) # you can also simply the geometry using the simplify method, but this will not preserve the topology of the polygons

# List of polygons in the union of the polygons in the shape dataframe
domain = list(polygonize(italy_union))

# Continental Italy and two main islands (Sicily and Sardinia) are the three 
# disjoint polygons
print(f"n. domain: {len(domain)}") 

# %% Build the mesh for the AQ_pm10 observed variable

# Build a low-rank mesh: reduce the vertex budget to ~30% of len(points),
# while keeping more vertices where points are dense and fewer where they
# are sparse
lr = [0.5, 0.75]

meshes_io = [
        buildMesh2d(points, domain=domain, offset=1, 
            gradation=0.5, bandwidth=0.05, lowrank=r, return_info=True) 
        for r in  lr]

for i, (mesh_lr, buffer, info) in enumerate(meshes_io):
    print(f"low-rank mesh: {info['n_inside']} vertices "
        f"(target {round(lr[i] * len(points))})")

# %% Plot the derived meshes and the observed points

fig, ax = plt.subplots(1, len(meshes_io), figsize=(8 * len(meshes_io), 8))

for i, (mesh_lr, buffer, info) in enumerate(meshes_io):
    fem_solver = FEMSolver(mesh_lr, domain, verbose=False)

    # plot the mesh using the FEM solver utilities
    fem_solver.plot_mesh(ax=ax[i])
    ax[i].plot(points[:, 0], points[:, 1], "x", markersize=2, alpha=0.5)
    ax[i].set_xlabel("Longitude")
    ax[i].set_ylabel("Latitude")
    ax[i].set_title(f"Low-rank mesh (r={lr[i]})")
    ax[i].set_aspect('equal', adjustable='box')

# %% Build the covaraince function 
from geossm.covmodel.covmodels import SPDEApproxCov as MaternCov


for i, (mesh_lr, buffer, info) in enumerate(meshes_io):
    cov_fun = MaternCov(latlon=True, nu=1, var=1, rescale=4)
    cov_fun = cov_fun.setup(mesh_lr, domain=domain)

    print(f"Covariance function for low-rank mesh (r={lr[i]}):")
    print(cov_fun.summary())



 
