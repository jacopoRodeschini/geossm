#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:38:28 2026

@author: jacopo
"""


import mfem.ser as mfem

import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy as sc
import gstools as gs
from shapely.geometry import LineString, Point, Polygon, MultiPoint
import numpy as np
from gstools.covmodel import Matern
import warnings

from scipy.spatial import ConvexHull
import meshio
import pygmsh
import gmsh
import geopandas as geodf
import geossm.datasets as df
from geossm.stmodel import LRStateSpaceModel as lrssm
from geossm.covmodel import FEMSolver


# %% Load the agrimonia dataset

agri, shape = df.load_dataset('agrimonia')

# %% From .csv to geopandas

ct = np.array([agri.Longitude.to_numpy(), agri.Latitude.to_numpy()]).T
agri['geometry'] = [Point(p[0], p[1]) for p in ct]  # (x,y) = (lat,lon)

agri = geodf.GeoDataFrame(agri, crs=4326)

domain = list(shape.geometry[0].geoms)[0].boundary
buffer = list(domain.buffer(0.3).boundary.geoms)[0]


# %% Build the model

model = lrssm(agri, ['AQ_pm10 ~ 1 + WE_temp_2m'], verbose=True, domain = [Polygon(buffer)])

# %% [Utils] build mesh with gmsh

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
mesh_io = buildMesh(buffer, 0.5, points)
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

est_params, y_hat, x_T, P_T, cov_function, nstat = model.fit()

# plot the likelihood curve
fig, ax = plt.subplots()
ax.plot([i['deltaL'] for i in nstat])
ax.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('Log Likelihood')
ax.set_title('Log Likelihood Curve')
ax.grid()
plt.show()


# %% Estimate the model with other options  

from geossm.stmodel import FitOptions

opt = FitOptions()
opt.max_iter = 5

print(opt)


est_params, y_hat, x_T, P_T, cov_function, nstat = model.fit(options=opt)

# plot the likelihood curve
fig, ax = plt.subplots()
ax.plot([i['deltaL'] for i in nstat])
ax.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('Log Likelihood')
ax.set_title('Log Likelihood Curve')
ax.grid()
plt.show()

# %% Estimate the model with inital values
from geossm.stmodel import FitOptions, ModelParams

# set the option 
opt = FitOptions()
opt.max_iter = 500
opt.tol_relat = 1e-4

# set the pars0
par0 = ModelParams(beta=[10,1], s2e = 10)
print(par0)

# % fit the model
est_params, y_hat, x_T, P_T, cov_function, nstat = model.fit(params0=par0, options=opt)

# print the estimate paramiters
print(est_params)

# %% More controll on the model paramiters
from geossm.stmodel.lrssm import Param, ModelParams

b0 = Param("beta", [11,1], fixed=True)

# set the pars0
par0 = ModelParams(beta=b0, s2e = 10)

# %%
print(par0)

# % fit the model
est_params, y_hat, x_T, P_T, cov_function, nstat = model.fit(params0=par0, options=opt)

# print the estimate paramiters
print(est_params)




































