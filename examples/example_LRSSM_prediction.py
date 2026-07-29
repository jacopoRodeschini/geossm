#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:38:28 2026

@author: jacopo
"""

# %% Imports 
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import numpy as np
import pygmsh
import gmsh
import pandas as pd
import geopandas as geodf
from geossm import datasets
from geossm.stmodel import LRStateSpaceModel as lrssm
from geossm.stmodel import FitOptions
from geossm.stmodel import ModelParams
from geossm.covmodel import FEMSolver

# %% Download the the Full agrimonia dataset from zenodo repository

agri_path = 'Download/Agrimonia_Dataset_v_3_0_0.csv'

# %% Load the agrimonia dataset

agri = pd.read_csv(agri_path, na_values=[
                   "      NaN", "", "NA", "null", "-"], keep_default_na=True)
agri['Time'] = pd.to_datetime(agri['Time'], format="%Y-%m-%d")

# Import the Lombardy shape file
_, shape = datasets.load_dataset("agrimonia")
buffer = shape.geometry.boundary[0].buffer(0.3).boundary
line_strings = list(buffer.geoms)
buffer = Polygon(line_strings[0].coords)

# %% Cut the dataset from 2019 to 2020
tmin = np.datetime64("2020-01-01")
tmax = np.datetime64("2020-12-31")
agri = agri[(agri["Time"] >= tmin) & (agri["Time"] <= tmax)].copy()


# %% From .csv to geopandas

ct = np.array([agri.Longitude.to_numpy(), agri.Latitude.to_numpy()]).T
agri["geometry"] = [Point(p[0], p[1]) for p in ct]  # (x,y) = (lat,lon)
agri = geodf.GeoDataFrame(agri, crs=4326)


# %% Build the model

model = lrssm(
    agri, ["AQ_pm10 ~ 1 + WE_temp_2m + WE_tot_precipitation + WE_wind_speed_10m_mean"], verbose=True, domain=[buffer])


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
opt.max_iter = 50
opt.tol_relat = 1e-3

results = model.fit(options=opt)
print(results)  # resutls.summary()


# %% Do the prediction 
import pandas as pd

# import the covariates dataset
path = "Download/AGC_Dataset_3.0.0.csv"

def converter(value):
    try:
        # Convert the value to float
        return np.float64(value)
    except ValueError:
        # If conversion fails, return the original value
        return value

# grid = pd.read_csv(path, na_values=[
#                    "      NaN", "        NA", "", "NA", "null", "-"], keep_default_na=True, converters={col: converter for col in range(45)})
grid = pd.read_csv(path, na_values=["      NaN", "        NA", "", "NA", "null", "-", '       NA'])
grid['Time'] = pd.to_datetime(grid['Time'], format="%Y-%m-%d")

# Remove columns with object type (non-numeric) to avoid issues during prediction
cols_to_remove = grid.select_dtypes(include=["object"]).columns
grid = grid.drop(columns=cols_to_remove)


# Create Point(lat,lon) for each observation
ct = np.array([grid.Longitude.to_numpy(), grid.Latitude.to_numpy()]).T
pt = [Point(p[0], p[1]) for p in ct]  # (x,y) = (lat,lon)

# Add point to Agrimonia dataset
grid['geometry'] = pt

# create geopandas dataset
grid = geodf.GeoDataFrame(grid, crs=4326)


# %% Model prediction 

# use the model.predict method to get the prediction and the uncertainty 
points, y_hat, Sigma_y_hat, tdelta = model.predict(grid, results)
# points, y_hat_v1, Sigma_y_hat, tdelta = results.predict(grid, verbose=True)

# %% PLot the results using imshow (for each time step)
from matplotlib.colors import Normalize

# Define month boundaries (assuming daily data for a year)
# Days: Jan(31), Feb(28), Mar(31), Apr(30), May(31), Jun(30), 
#       Jul(31), Aug(31), Sep(30), Oct(31), Nov(30), Dec(31)
month_ranges = [
    (0, 31, "January"),
    (31, 59, "February"),
    (59, 90, "March"),
    (90, 120, "April"),
    (120, 151, "May"),
    (151, 181, "June"),
    (181, 212, "July"),
    (212, 243, "August"),
    (243, 273, "September"),
    (273, 304, "October"),
    (304, 334, "November"),
    (334, 365, "December")
]

# Calculate monthly averages
monthly_avg = []
month_names = []
for start, end, name in month_ranges:
    avg = np.nanmean(y_hat[0][:, start:end], axis=1)
    monthly_avg.append(avg)
    month_names.append(name)

# Plot
fig, axs = plt.subplots(3, 4, figsize=(14, 10))

# Create normalization for consistent coloring
vmin = np.nanmin([np.nanmin(m) for m in monthly_avg])
vmax = np.nanmax([np.nanmax(m) for m in monthly_avg])
norm = Normalize(vmin=vmin, vmax=vmax)

for i, (avg_data, month_name) in enumerate(zip(monthly_avg, month_names)):
    ax = axs[i // 4, i % 4]

    # Get unique sorted coordinates
    x_unique = np.sort(np.unique(points[0][:, 0]))
    y_unique = np.sort(np.unique(points[0][:, 1]))
    
    # Create mapping dictionaries
    x_to_idx = {x: idx for idx, x in enumerate(x_unique)}
    y_to_idx = {y: idx for idx, y in enumerate(y_unique)}
    
    # Create 2D grid
    rows, cols = len(y_unique), len(x_unique)
    Z = np.full((rows, cols), np.nan)
    
    # Fill grid
    x_indices = np.array([x_to_idx[x] for x in points[0][:, 0]])
    y_indices = np.array([y_to_idx[y] for y in points[0][:, 1]])
    Z[y_indices, x_indices] = avg_data
    
    # Mask pixels outside boundary
    for row_idx in range(rows):
        for col_idx in range(cols):
            x_coord = x_unique[col_idx]
            y_coord = y_unique[row_idx]
            point = Point(x_coord, y_coord)
            if not shape.geometry.iloc[0].contains(point):
                Z[row_idx, col_idx] = np.nan
    
    # Set extent
    extent = [x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()]
    
    # Plot with white background for NaN values
    cmap = plt.cm.YlOrBr.copy()
    cmap.set_bad(color='white')
    
    im = ax.imshow(
        Z,
        cmap=cmap,
        origin="lower",
        extent=extent,
        interpolation="none",
        aspect="auto",
        norm=norm
    )
    shape.boundary.plot(ax=ax, color="black", linewidth=1)

    # Style
    ax.set_title(month_name, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

# Add shared colorbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label="PM10 (µg/m³)")

plt.tight_layout(rect=[0, 0, 0.88, 1])