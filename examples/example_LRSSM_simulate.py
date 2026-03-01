#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:38:28 2026

@author: jacopo
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import pygmsh
import gmsh
import geopandas as gpd 
from datetime import date, timedelta


# %% import the geossm package

import geossm

# print("Version: ", geossm.__version__)
print("Load from: ", geossm.__file__)


# %% Import the Matern model based on the SPDE approach R^2

if geossm.__file__:
    # import covariance model
    from geossm.covmodel.covmodels import spdeAppoxCov

    # import lrssm
    from geossm.stmodel import LRStateSpaceModel as lrssm


# %% Simulate random point in a convex space 

# set the domain as a circle with radius 10
center = (0, 0)
radius = 1 
circle = Point(center).buffer(radius)

points = np.random.uniform(-radius, radius, (100, 2))

# take the mask of the points that are inside the circle
mask = np.array([circle.contains(Point(p)) for p in points])
points = points[mask]

#plot the points and the domain
plt.figure(figsize=(6,6))
plt.scatter(points[:,0], points[:,1], color='blue', label='Random Points')
plt.plot(*circle.boundary.xy, color='red', label='Domain Boundary')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Random Points in a Circular Domain')
plt.legend()
plt.axis('equal')
plt.show()

# %% Create a mesh 

def buildMesh(poly, lc, points, lc_buffer=None, lc_points=1e22):
    with pygmsh.occ.Geometry() as geom:

        if lc_buffer is None:
            lc_buffer = lc

        coords = np.array(poly.buffer(
            lc_buffer).simplify(lc_buffer).exterior.coords[:-1])
        domain = geom.add_polygon(coords, mesh_size=lc_buffer*0.1)

        # 2. Add physical group for the domain surface (good practice)
        geom.add_physical(domain, label="surface_domain")

        # Add points for the boundary
        embedded_tags = []
        for p in points:
            t = gmsh.model.occ.addPoint(p[0], p[1], 0, lc_points)
            embedded_tags.append(t)

        gmsh.model.occ.synchronize()  # Synchronize OCC entities before using them in fields

        # fix the points
        gmsh.model.mesh.embed(
            0, embedded_tags, 2, domain._id)

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

mesh_io = buildMesh(circle, 1, points)
print(mesh_io)


# %% Create the covariance function

from geossm.covmodel import spdeAppoxCov as matern

cov_fun = matern([circle], latlon=False, nu=1, var = 2, rescale=4)
cov_fun = cov_fun.setup(mesh_io)

# PLot the mesh behind the cov. function 
fig, ax = plt.subplots(figsize=(8, 8))
cov_fun.fem_solver.plot_mesh(ax=ax)
ax.plot(points[:, 0], points[:, 1], 'ro', label="Observation Points")
ax.set_title("Mesh for $PM_{10}$")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
plt.show()  


# %% Build the model (which allows oly for geopandas)

# Create the spatio-temporal covariates (e.g. temperature and humidity)
n = points.shape[0]
T = 20

points = np.tile(points, (T, 1, 1)).reshape(-1, 2) # repeat the points T times for the temporal dimension

d = np.linspace(0, 2 * np.pi, n)

temperature = (np.sin(d).reshape(-1,1) + np.random.normal(0, 1, size=n * T).reshape(n, T)).reshape(-1,)
humidity = (1/2 * np.sin(d).reshape(-1,1) + np.random.normal(0, 1, size=n * T).reshape(n, T)).reshape(-1,)

tstart = date(2020,1,1)
time = np.tile([tstart + timedelta(days=d) for d in range(T)], n)

# Create a GeoDataFrame from the points
gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(points[:,0], points[:,1]))
gdf['temperature'] = temperature
gdf['humidity'] = humidity
gdf['Time'] = time
gdf.crs = "EPSG:4326"  # Set the coordinate reference system



# %% build the lrssm and set the covariance function 


model = lrssm(df=gdf, formulas=['humidity ~ 1 + temperature'], domain=[circle], verbose=True)
print(model)

model = model.setup(cov_fun=[cov_fun],domain=[circle])
print(model)

# %% Create the model parameters for the simulation
from geossm.stmodel import ModelParams

params = ModelParams(
    beta=[0.5, 1.0], 
    A=np.array([[0.8]]), 
    s2e=[0.5], 
    ks = [0.5],
    f = [0.9]
    )

# %% Simulate from the model

y_sim,  x_sim, tdelta = model.sim(params=params)
print(y_sim.shape)
print(x_sim.shape)


# %% Plot the simulated data for the time = 1, 10, 20

pt = np.array(model.points[0])

fix, ax = plt.subplots(1,3, figsize=(18,6))
for i, t in enumerate([0, 9, 19]):
    ax[i].scatter(pt[:,0], pt[:,1], c=y_sim[:, t], cmap='viridis')
    ax[i].plot(*circle.boundary.xy, color='red', label='Domain Boundary')
    ax[i].set_xlabel('X-axis')
    ax[i].set_ylabel('Y-axis')
    ax[i].set_title(f'Simulated $PM_{{10}}$ at Time Step {t+1}')
    ax[i].legend()
    ax[i].axis('equal')
plt.colorbar(ax[2].collections[0], ax=ax, orientation='vertical', fraction=0.035, pad=0.02, label='Simulated $PM_{10}$')
plt.show()
