#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:38:28 2026

@author: jacopo
"""

import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import numpy as np
import pygmsh
import gmsh
import geopandas as geodf
import geossm.datasets as df
from geossm.stmodel import LRStateSpaceModel as lrssm
from geossm.stmodel import FitOptions
from geossm.stmodel import ModelParams
from geossm.covmodel import FEMSolver

# %% Load the agrimonia dataset

agri, shape = df.load_dataset("agrimonia")


# %% From .csv to geopandas

ct = np.array([agri.Longitude.to_numpy(), agri.Latitude.to_numpy()]).T
agri["geometry"] = [Point(p[0], p[1]) for p in ct]  # (x,y) = (lat,lon)

agri = geodf.GeoDataFrame(agri, crs=4326)

domain = list(shape.geometry[0].geoms)[0].boundary
buffer = list(domain.buffer(0.3).boundary.geoms)[0]


# %% Build the model

model = lrssm(
    agri, ["np.sqrt(np.abs(AQ_pm10)) ~ 1 + WE_temp_2m"], verbose=True, domain=[Polygon(buffer)]
)


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
        # gmsh.model.mesh.embed(
        #     0, embedded_tags, 2, domain._id)

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
mesh_io = buildMesh(buffer, 0.35, points)
print(mesh_io)

# plot the mesh (use the fem_solver utlities)
fem_solver = FEMSolver(mesh_io, [Polygon(buffer)])

# plot the mesh using the utilities
fig, ax = plt.subplots(figsize=(8, 8))
fem_solver.plot_mesh(ax=ax)

# %% Set up the lrssm model (univiarte latent)

# add the mesh object and the domain where the laten domain is defined
# if None it is assumed to be the same of the observation
model = model.setup([mesh_io])


# %% Estimate the Model (default estimation options)
opt = FitOptions()
opt.max_iter = 10
opt.tol_relat = 1e-4

results = model.fit(options=opt)
print(results)  # resutls.summary()


# %% Do the prediction 
import pandas as pd

# import the covariates dataset
path = "/home/jacopo/Documents/Dottorato/Progetti/dataset.Agrimonia/DATA_grid/AGC_Dataset_3.0.0.csv"

def converter(value):
    try:
        # Convert the value to float
        return np.float64(value)
    except ValueError:
        # If conversion fails, return the original value
        return value

# grid = pd.read_csv(path, na_values=[
#                    "      NaN", "        NA", "", "NA", "null", "-"], keep_default_na=True, converters={col: converter for col in range(45)})

grid = pd.read_csv(path, na_values=["      NaN", "        NA", "", "NA", "null", "-"])
grid['Time'] = pd.to_datetime(grid['Time'], format="%Y-%m-%d")


cols_to_convert = grid.select_dtypes(include=["object"]).columns

grid[cols_to_convert] = grid[cols_to_convert].apply(
    pd.to_numeric,
    errors="coerce"
)


# Create Point(lat,lon) for each observation
ct = np.array([grid.Longitude.to_numpy(), grid.Latitude.to_numpy()]).T
pt = [Point(p[0], p[1]) for p in ct]  # (x,y) = (lat,lon)

# Add point to Agrimonia dataset
grid['geometry'] = pt

# create geopandas dataset
grid = geodf.GeoDataFrame(grid, crs=4326)


# %% Model prediction 

results.predict(grid, verbose=True)


