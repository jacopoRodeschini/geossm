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


# %% Build a "full" mesh (one vertex density everywhere, like inla.mesh.2d
# without any thinning)

mesh_full = buildMesh2d(points, max_edge=0.4)
print(f"full mesh: {len(mesh_full.points)} vertices")

# %% Build a low-rank mesh: reduce the vertex budget to ~30% of len(points),
# while keeping more vertices where points are dense and fewer where they
# are sparse
lr = 1
mesh_lr = buildMesh2d(points, max_edge=0.4, lowrank=lr)
print(f"low-rank mesh: {len(mesh_lr.points)} vertices "
      f"(target {round(lr * len(points))})")

# %% Plot: observed points, full mesh, low-rank mesh, and grid mesh

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

axs[0].plot(points[:, 0], points[:, 1], "x", markersize=3)
axs[0].set_title(f"Observed points (n={len(points)})")

axs[1].triplot(
    mesh_full.points[:, 0], mesh_full.points[:, 1],
    mesh_full.cells_dict["triangle"], linewidth=0.5,
)
axs[1].set_title(f"Full mesh ({len(mesh_full.points)} vertices)")

axs[2].triplot(
    mesh_lr.points[:, 0], mesh_lr.points[:, 1],
    mesh_lr.cells_dict["triangle"], linewidth=0.5,
)
axs[2].set_title(f"Low-rank mesh, lowrank={lr} ({len(mesh_lr.points)} vertices)")

for ax in axs:
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()
