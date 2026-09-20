#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""

# %% [Imports]

import numpy as np
import matplotlib.pyplot as plt

from geossm.covmodel.utils import buildMesh2d, buildMeshGrid2d
import geossm.datasets as df 


# %% Import the grins dataset

grins_data, shape = df.load_dataset('aqclim_points', return_geometry=True)
points = np.array([[geom.x, geom.y] for geom in grins_data.geometry.unique()])


#%% get the larges poly 

# %% compute only the continetal italian (connected mesh)
from shapely.geometry import Polygon, MultiPolygon

shape.to_crs(4326, inplace=True)
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
# ct_italy = sh_italy[~sh_italy['COD_REG'].isin([19, 20])]
ct_italy = shape.copy()

# get the largest polygon
ct_italy['geometry'] = ct_italy['geometry'].apply(
    lambda geo: getLargestPoly(geo))
ct_italy.plot()


# %% Build a low-rank mesh: reduce the vertex budget to ~30% of len(points),
# while keeping more vertices where points are dense and fewer where they
# are sparse
lr = [0.5, 0.75]

temp = [buildMesh2d(points, max_edge=0.4, min_edge=0.1, lowrank=r, density_neighbors=8) for r in  lr]

for i, (mesh_lr, buffer) in enumerate(temp):
    print(f"low-rank mesh: {len(mesh_lr.points)} vertices "
        f"(target {round(lr[i] * len(points))})")

# %% Plot: observed points, full mesh, low-rank mesh, and grid mesh


fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

axs[0].plot(points[:, 0], points[:, 1], "x", markersize=3)
axs[0].set_title(f"Observed points (n={len(points)})")

for i, (mesh_lr, buffer) in enumerate(temp):
    axs[i + 1].triplot(
        mesh_lr.points[:, 0], mesh_lr.points[:, 1],
        mesh_lr.cells_dict["triangle"], linewidth=0.5,
    )
    axs[i + 1].set_title(f"Low-rank mesh, lowrank={lr[i]} ({len(mesh_lr.points)} vertices)")

ct_italy.plot(ax=axs[0], color="black", linewidth=1)
ct_italy.plot(ax=axs[1], color="black", linewidth=1)
ct_italy.plot(ax=axs[2], color="black", linewidth=1)

for ax in axs:
    ax.set_aspect("equal")

axs[0].set_ylabel("Latitude")
for ax in axs:
    ax.set_xlabel("Longitude")

plt.tight_layout()
plt.show()
