#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import pygmsh
import gmsh
import geopandas as geodf
import geossm.datasets as df
from geossm.stmodel import LRStateSpaceModel as lrssm
from geossm.covmodel import FEMSolver


# %% List the available dataset in the geossm packace

print(df.list_datasets())

# %% Load the agrimonia dataset

agri, shape = df.load_dataset("agrimonia")

# %% From .csv to geopandas

ct = np.array([agri.Longitude.to_numpy(), agri.Latitude.to_numpy()]).T
agri["geometry"] = [Point(p[0], p[1]) for p in ct]  # (x,y) = (lat,lon)

agri = geodf.GeoDataFrame(agri, crs=4326)

# %% Plot the AQ_pm10 dataset and the domain

domain = list(shape.geometry[0].geoms)[0].boundary
buffer = list(domain.buffer(0.3).boundary.geoms)[0]

mean_pm10_scatter = agri.groupby(["IDStations"]).agg(
    {"AQ_pm10": "mean", "geometry": lambda x: x.iloc[0]}
)

# --- Extract coordinates ---

x_domain, y_domain = domain.coords.xy
x_buff, y_buff = buffer.coords.xy

# --- Create subplots (1 row, 2 columns) ---
fig, ax = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)

# Ensure same color scale across both plots
vmin = mean_pm10_scatter["AQ_pm10"].min()
vmax = mean_pm10_scatter["AQ_pm10"].max()

# --- PM10 ---
sc1 = ax.scatter(
    mean_pm10_scatter.geometry.values.x,
    mean_pm10_scatter.geometry.values.y,
    c=mean_pm10_scatter["AQ_pm10"],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)

ax.plot(x_buff, y_buff, "b--", label="Boundary")
ax.plot(x_domain, y_domain, "b--", label="Domain")
ax.set_title("Annual Mean $PM_{10}$")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.6)


cbar = fig.colorbar(sc1, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
cbar.set_label(r"$\mu g / m^3$", fontsize=12)
plt.show()


# %% Build the LRSSM model

domain = [Polygon(buffer)]
model = lrssm(agri, ["AQ_pm10 ~ 1 + WE_temp_2m"], verbose=True, domain=domain)

print(model)

# %% [Utils] build mesh with gmsh


def buildMesh(poly, lc, points, lc_buffer=None, lc_points=1e22):
    with pygmsh.occ.Geometry() as geom:

        if lc_buffer is None:
            lc_buffer = lc

        coords = np.array(
            poly.buffer(lc_buffer).simplify(lc_buffer).exterior.coords[:-1]
        )
        domain = geom.add_polygon(coords, mesh_size=lc_buffer * 0.1)

        # 2. Add physical group for the domain surface (good practice)
        geom.add_physical(domain, label="surface_domain")

        # Add points for the boundary
        embedded_tags = []
        for p in points:
            t = gmsh.model.occ.addPoint(p[0], p[1], 0, lc_points)
            embedded_tags.append(t)

        gmsh.model.occ.synchronize()  # Synchronize OCC entities before using them in fields

        # fix the points
        gmsh.model.mesh.embed(0, embedded_tags, 2, domain._id)

        gmsh.option.setNumber("Mesh.Algorithm", 6)

        # CRITICAL: Tell Gmsh NOT to force density based on the internal points
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # Allow triangles to be very large
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        # Only limit the absolute minimum to prevent crashes
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.1)

        # 5. Generate
        gmsh.model.mesh.generate(2)

        gmsh.model.mesh.optimize("Laplace2D")
        gmsh.option.setNumber("Mesh.Smoothing", 10)

        # # This allows the optimizer to move nodes more freely
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        mesh = geom.generate_mesh()

    return mesh


# %% Build the mesh for the AQ_pm10 observed variable

points = model.points[0]
mesh_io = buildMesh(buffer, 1, points)
print(mesh_io)

# plot the mesh (use the fem_solver utlities)
fem_solver = FEMSolver(mesh_io, [Polygon(buffer)])


# plot the mesh using the utilities
fig, ax = plt.subplots(figsize=(8, 8))
fem_solver.plot_mesh(ax=ax)
ax.plot(points[:, 0], points[:, 1], "ro", label="Observation Points")
ax.set_title("Mesh for $PM_{10}$")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
plt.show()


# %% Set up the lrssm model (univiarte latent)

# add the mesh object and the domain where the laten domain is defined
# if None it is assumed to be the same of the observation
model = model.setup(mesh_obj=[mesh_io], domain_latent=domain)
print(model)
