#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""

# %% [Imports]

import numpy as np
import matplotlib.pyplot as plt

from geossm.covmodel.utils import buildMesh2d
import geossm.datasets as df 


# %% Import the grins dataset

grins_data, shape = df.load_dataset('aqclim_points')
points = np.array([[geom.x, geom.y] for geom in grins_data.geometry.unique()])

# %% fix the shape into a list of poligons

from shapely.geometry import MultiLineString
from shapely.ops import polygonize
from shapely import ops

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


# %% Build a low-rank mesh: reduce the vertex budget to ~30% of len(points),
# while keeping more vertices where points are dense and fewer where they
# are sparse
lr = [0.5, 0.75]

temp = [buildMesh2d(points, lowrank=r, domain=domain, offset=0.1,
        eps=0.1, gradation=0.5, bandwidth=0.1,anchor_spacing=0.3) for r in  lr]

for i, (mesh_lr, buffer) in enumerate(temp):
    print(f"low-rank mesh: {len(mesh_lr.points)} vertices "
        f"(target {round(lr[i] * len(points))})")

# %% Plot: observed points, full mesh, low-rank mesh, and grid mesh
from geopandas import GeoDataFrame
dm = GeoDataFrame(geometry=domain, crs=shape.crs)

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

axs[0].plot(points[:, 0], points[:, 1], "x", markersize=3)
axs[0].set_title(f"Observed points (n={len(points)})")

for i, (mesh_lr, buffer) in enumerate(temp):
    axs[i + 1].triplot(
        mesh_lr.points[:, 0], mesh_lr.points[:, 1],
        mesh_lr.cells_dict["triangle"], linewidth=0.5,
    )
    axs[i + 1].set_title(f"Low-rank mesh, lowrank={lr[i]} ({len(mesh_lr.points)} vertices)")

dm.boundary.plot(ax=axs[0], color="black", linewidth=1)
dm.boundary.plot(ax=axs[1], color="black", linewidth=1)
dm.boundary.plot(ax=axs[2], color="black", linewidth=1)

for ax in axs:
    ax.set_aspect("equal")

axs[0].set_ylabel("Latitude")
for ax in axs:
    ax.set_xlabel("Longitude")

plt.tight_layout()
plt.close()
