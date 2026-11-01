#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""

# %% [Imports]

import numpy as np
import matplotlib.pyplot as plt

import shapely
from shapely.geometry import MultiPoint

from geossm.covmodel.utils import buildMesh2d

# %% Create a set of observed locations with an inhomogeneous density:
# a dense cluster around (0.3, 0.3) plus a sparse background over [0, 1]^2

np.random.seed(0)

n_cluster = 250
n_background = 60

domain = shapely.geometry.box(0, 0, 1, 1)

pts_cluster = np.random.normal(loc=[0.3, 0.3], scale=0.06, size=(n_cluster, 2))
pts_background = np.random.uniform(0, 1, size=(n_background, 2))

points = np.clip(np.vstack([pts_cluster, pts_background]), 0, 1)

# %% Build a "full" mesh (one vertex density everywhere, like inla.mesh.2d
# without any thinning)

mesh_full, boundary_full = buildMesh2d(points, offset=0.1, lowrank=1.0, anchor_spacing=0.2)
print(f"full mesh: {len(mesh_full.points)} vertices")


mesh_lr, _ = buildMesh2d(points, offset=0.1, lowrank=0.3, anchor_spacing=0.2)
print(f"low-rank mesh: {len(mesh_lr.points)} vertices")


# %% Plot: observed points, full mesh, low-rank mesh, grid mesh, and the
# penalized (data-overlapping) mesh

def isinner(p, poly):
    return poly.contains(shapely.geometry.Point(p))

x, y = boundary_full.exterior.xy
xd, yd = domain.exterior.xy

fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

axs[0].plot(points[:, 0], points[:, 1], "x", markersize=3)

axs[0].set_title(f"Observed points (n={len(points)})")

axs[1].triplot(
    mesh_full.points[:, 0], mesh_full.points[:, 1],
    mesh_full.cells_dict["triangle"], linewidth=0.5,
)
axs[1].set_title(f"Full mesh ({sum(isinner(p, domain) for p in mesh_full.points)} vertices)")

axs[2].triplot(
    mesh_lr.points[:, 0], mesh_lr.points[:, 1],
    mesh_lr.cells_dict["triangle"], linewidth=0.5,
)
axs[2].set_title(f"Low-rank mesh, lowrank=0.3 ({sum(isinner(p, domain) for p in mesh_lr.points)} vertices)")

for ax in axs:
    ax.set_aspect("equal")

for ax in axs:
    ax.plot(x, y, color="red", linewidth=0.7)
    ax.plot(xd, yd, color="black", linewidth=1)

plt.tight_layout()
plt.show()
plt.close()
