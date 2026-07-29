#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:38:28 2026

@author: jacopo
"""

# %% 

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

print(df.list_datasets())

agri, shape = df.load_dataset("agrimonia")


# %% From .csv to geopandas

ct = np.array([agri.Longitude.to_numpy(), agri.Latitude.to_numpy()]).T
agri["geometry"] = [Point(p[0], p[1]) for p in ct]  # (x,y) = (lat,lon)

agri = geodf.GeoDataFrame(agri, crs=4326)

domain = list(shape.geometry[0].geoms)[0].boundary
buffer = list(domain.buffer(0.3).boundary.geoms)[0]


# %% Build the model

model = lrssm(
    agri, ["np.sqrt(AQ_pm10) ~ 1 + WE_temp_2m"], verbose=True, domain=[Polygon(buffer)]
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

# %% Set up the lrssm model (univiarte latent)

# add the mesh object and the domain where the laten domain is defined
# if None it is assumed to be the same of the observation
model = model.setup([mesh_io])


# %% Estimate the Model (default estimation options)

opt = FitOptions()
opt.max_iter = 50
opt.tol_relat = 1e-3

results = model.fit(options=opt)

#print(results)  # resutls.summary(hessian=True, alpha=0.05) can be used to get a summary of the results
print(results.summary(hessian=False, alpha=0.05))  # resutls.summary(hessian=True, alpha=0.05) can be used to get a summary of the results

print(results.summary(hessian=True, alpha=0.05)) 

# %% Compute the standard errors of the parameters

hessian = results.compute_cov_params()

# %% Get the model parameter inference

# bse
params = results.bse
print(params.beta.bse)

# %% CI 
ci = results.conf_int(alpha=0.05)






