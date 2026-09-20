#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""

# %% [Imports]

import numpy as np
import matplotlib.pyplot as plt

import shapely
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint

from geossm.covmodel.utils import buildMesh2d, buildMesh2d_density, buildMeshGrid2d

# %% Create a set of observed locations with an inhomogeneous density:
# a dense cluster around (0.3, 0.3) plus a sparse background over [0, 1]^2

np.random.seed(0)

n_cluster = 250
n_background = 60

pts_cluster = np.random.normal(loc=[0.3, 0.3], scale=0.06, size=(n_cluster, 2))
pts_background = np.random.uniform(0, 1, size=(n_background, 2))

points = np.clip(np.vstack([pts_cluster, pts_background]), 0, 1)

# %% Build a "full" mesh (one vertex density everywhere, like inla.mesh.2d
# without any thinning)

mesh_full, boundary_full = buildMesh2d(points, max_edge=0.08)
print(f"full mesh: {len(mesh_full.points)} vertices")

# %% Build a low-rank mesh: reduce the vertex budget to ~30% of len(points),
# while keeping more vertices where points are dense and fewer where they
# are sparse

mesh_lr, boundary_lr = buildMesh2d(points, max_edge=0.08, lowrank=0.3)
# `lowrank` only targets vertices inside the interest domain (here, since no
# explicit `boundary` was given, the convex hull of `points` itself) -- the
# returned `boundary_lr` is the *buffered* outer meshing domain instead.
interest_domain = MultiPoint(points).convex_hull
n_inside = shapely.contains_xy(
    interest_domain, mesh_lr.points[:, 0], mesh_lr.points[:, 1]
).sum()
print(f"low-rank mesh: {len(mesh_lr.points)} vertices total, "
      f"{n_inside} inside the interest domain "
      f"(target {round(0.3 * len(points))})")

# %% Build a regular grid mesh: a structured lattice, cropped to the
# (buffered) convex hull of the points, with its resolution solved
# automatically so it also lands at ~30% of len(points) vertices

mesh_grid, buffer = buildMeshGrid2d(points=points, offset=0.05, lowrank=0.3)
print(f"grid mesh: {len(mesh_grid.points)} vertices "
      f"(target {round(0.3 * len(points))})")

# %% Build a "penalized" low-rank mesh: interior vertices are landmark
# points selected (via k-means) to match the density of `points`, then
# snapped onto the nearest actual point so they coincide with real
# observations instead of merely being near them

mesh_pen, boundary_pen = buildMesh2d_density(points, lowrank=0.3, max_edge=0.08, seed=0)
n_inside_pen = shapely.contains_xy(
    interest_domain, mesh_pen.points[:, 0], mesh_pen.points[:, 1]
).sum()
n_overlap = (
    cKDTree(points).query(mesh_pen.points[:, :2])[0] < 1e-9
).sum()
print(f"penalized mesh: {len(mesh_pen.points)} vertices total, "
      f"{n_inside_pen} inside the interest domain "
      f"(target {round(0.3 * len(points))}), {n_overlap} coincide exactly "
      "with an observed point")

# %% Plot: observed points, full mesh, low-rank mesh, grid mesh, and the
# penalized (data-overlapping) mesh

fig, axs = plt.subplots(1, 5, figsize=(24, 5), sharex=True, sharey=True)

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
axs[2].set_title(f"Low-rank mesh, lowrank=0.3 ({len(mesh_lr.points)} vertices)")

axs[3].triplot(
    mesh_grid.points[:, 0], mesh_grid.points[:, 1],
    mesh_grid.cells_dict["triangle"], linewidth=0.5,
)
axs[3].set_title(f"Grid mesh, lowrank=0.3 ({len(mesh_grid.points)} vertices)")

axs[4].triplot(
    mesh_pen.points[:, 0], mesh_pen.points[:, 1],
    mesh_pen.cells_dict["triangle"], linewidth=0.5,
)
axs[4].plot(points[:, 0], points[:, 1], "r.", markersize=2, alpha=0.4)
axs[4].set_title(f"Penalized mesh, lowrank=0.3 ({len(mesh_pen.points)} vertices)")

for ax in axs:
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()
