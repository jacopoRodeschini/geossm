#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:44:06 2025

@author: jacopo

create the mesh and the associate covariane function 
"""


import math
import pickle
import gmsh
import pygmsh
import matplotlib.tri as mtri  # For Triangulation object
import scipy.spatial
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
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

from scipy.sparse.linalg import splu


# %% import the geossm package

import geossm

# print("Version: ", geossm.__version__)
print("Load from: ", geossm.__file__)


# %% Import the Matern model based on the SPDE approach R^2

if geossm.__file__:
    from geossm.covmodel.covmodels import spdeAppoxCov

# %% Create random point on [0,1]^2

domain = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

n = 100
points = np.random.uniform(0, 1, size=(n, 2))


# plot the poitns
fix, ax = plt.subplots()
ax.plot(points[:, 0], points[:, 1], 'x', label='Random Points')
ax.plot(domain.exterior.xy[0], domain.exterior.xy[1],
        'r-', label='Domain Boundary')
ax.set_title('Random Points in [0,1]^2')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()
plt.show()

# %% [Utils] Build mesh gmsh


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
        # We find the node tag for each internal point and move it to the surface entity
        # for p_tag in embedded_tags:
        #     # Get the node tag associated with the geometry point
        #     node_tags, coords, _ = gmsh.model.mesh.getNodes(0, p_tag)
        #     if len(node_tags) > 0:
        #         # Reclassify this node to the surface (dim=2).
        #         # Now Gmsh thinks this is just a regular internal node.
        #         gmsh.model.mesh.setNodes(2, domain._id, node_tags, coords)

        # SOLUTION 2: Smoothing and Optimization
        # Set high smoothing iterations
        # gmsh.model.mesh.optimize("Relocate2D")

        gmsh.option.setNumber("Mesh.Smoothing", 10)

        # # This allows the optimizer to move nodes more freely
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        # Apply optimization/smoothing ONLY if requested (this might move the points slightly)
        # dimtags = [(2, t)for t in embedded_tags]
        # gmsh.model.mesh.optimize("Netgen")
        # gmsh.option.setNumber("Mesh.Smoothing", 10)

        # gmsh.model.mesh.optimize(
        #     "Laplace2D", force=False, niter=10, dimTags=dimtags)

        # gmsh.model.mesh.generate(2)

        # Optional: Extract nodes and elements for MFEM/PyVista
        # nodeTags, coords, _ = gmsh.model.mesh.getNodes()
        mesh = geom.generate_mesh()

    return mesh


# %% Create the mesh

mesh = buildMesh(domain, 1, points)

print(mesh)
print(len(mesh.cells_dict['triangle']))
print(len(mesh.cells_dict['vertex']))

# save the mesh on local disk
# mesh.write("mesh.vtu", mesh)     # VTK XML
# mesh.write("mesh.msh", mesh)     # Gmsh
# mesh.write("mesh.xdmf", mesh)    # XDMF (often used with FEM codes)
# mesh.write("mesh.obj", mesh)     # OBJ
# mesh.write("mesh.stl", mesh)     # STL (surface meshes)


# %% Create the covariance function

# Define the matern function (without any mesh definition)
cov_matern = spdeAppoxCov([domain], latlon=True)

print(cov_matern)
print(cov_matern.rescale)
print(cov_matern.dim)
print(cov_matern.var)
print(cov_matern.is_isotropic)


# %% Add the mesh support for the FE representation

cov_matern = cov_matern.setup(mesh)

print(cov_matern)

# get the FEM solver
print(cov_matern.fem_solver)

# plot the mesh
ax = cov_matern.fem_solver.plot_mesh()

# get the stiff and the mass matrix
mass = cov_matern.fem_solver.mass  # diagonal ()
stiff = cov_matern.fem_solver.stiff  # sparse

# Get the sparse precision represeation of the matern
Q = cov_matern.precision(rescale=1)

# %% Save the covariance to local disk
# pickable object

filename = 'matern_rescale_1.pkl'

with open(filename, 'wb') as f:
    pickle.dump(cov_matern, f)


# load
with open(filename, 'rb') as f:
    cov_load = pickle.load(f)

# check the load covariance
print(cov_load)
ax = cov_load.fem_solver.plot_mesh()

# %% Create different mesh (based on the LC)
lc_list = [0.3, 0.5, 1]

fix, axs = plt.subplots(1, len(lc_list), figsize=(15, 5))

for lc, ax in zip(lc_list, axs):

    mesh = buildMesh(domain, lc, points, lc_buffer=lc*2)

    cov_matern = cov_matern.setup(mesh)

    cov_matern.fem_solver.plot_mesh(ax=ax, title=f"Mesh with LC={lc}")
