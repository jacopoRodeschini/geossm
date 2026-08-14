#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""

# %% Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import polygon
import pygmsh
import gmsh
import geopandas as gpd
from datetime import date, timedelta
import geossm

if geossm.__file__:
    # import lrssm
    from geossm.covmodel import spdeAppoxCov as matern_spde
    from geossm.stmodel import ModelParams
    from geossm.stmodel import LRStateSpaceModel as lrssm
    from geossm.stmodel import FitOptions



# %% Simulate random point in a convex space (regular grind) and plot them

# set the domain as a square [0, 1] x [0, 1]
domain = polygon.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

# points = np.random.uniform(0, 1, (200, 2))
# Create regular grid points
x = np.linspace(0, 1, 15, endpoint=True)
y = np.linspace(0, 1, 15, endpoint=True)
points = np.array([[xi, yi] for xi in x for yi in y])

# plot the points and the domain
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], color="blue", label="Random Points")
plt.plot(*domain.boundary.xy, color="red", label="Domain Boundary")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Random Points in a Square Domain")
plt.legend()
plt.axis("equal")
plt.show()

# %% Create a mesh

def buildMesh(poly, lc, points, lc_buffer=None, lc_points=1e22):
    with pygmsh.occ.Geometry() as geom:

        if lc_buffer is None:
            lc_buffer = lc

        coords = np.array(
            poly.buffer(lc_buffer).simplify(lc_buffer).exterior.coords[:-1]
        )
        domain = geom.add_polygon(coords, mesh_size=lc_buffer)

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


mesh_io = buildMesh(domain, 0.1, points, lc_buffer=0.5)
print(mesh_io)


# %% Create the covariance function    
cov_fun = matern_spde([domain], latlon=False, nu=1, var=1, rescale=4)
cov_fun = cov_fun.setup(mesh_io)

print(cov_fun.summary())

# %% Plot the mesh behind the cov. function
fig, ax = plt.subplots(figsize=(8, 8))
cov_fun.fem_solver.plot_mesh(ax=ax)
ax.plot(points[:, 0], points[:, 1], "ro", label="Observation Points")
ax.set_title("Mesh for $PM_{10}$")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
# plt.show()


# %% Build the geopandas dataframe

# Create the spatio-temporal covariates (e.g. temperature and humidity)
n = points.shape[0]
T = 50

omega = 2 * np.pi / 30

# repeat the points T times for the temporal dimension
_points = np.tile(points, (T, 1, 1)).reshape(-1, 2) 

# Create the covariates
t = np.repeat(np.linspace(0, T, num=T), n)
t2m = 4*np.sin(omega * t).reshape(T,n).T  + np.random.normal(0, 1, size=n * T).reshape(n, T)
t2m = t2m.reshape(-1,order='F')

humidity = 2 * (np.sin(omega * t) + 0.5*np.cos(2* omega * t)).reshape(T,n).T  + np.random.normal(0, 1, size=n * T).reshape(n, T)
humidity = humidity.reshape(-1, order='F')

tstart = date(2020, 1, 1)
time = np.sort(np.tile([tstart + timedelta(days=d) for d in range(T)], n))

# create the response variable as a linear combination of the covariates plus some noise
beta = [2, 0.5, 0.3]  # coefficients for the covariates
s2e = 1  # variance of the noise
y = beta[0] + beta[1] * t2m + beta[2] * humidity + np.random.normal(0, np.sqrt(s2e), size=n * T).reshape(n, T).reshape(-1, )


# Create a GeoDataFrame from the points
gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(_points[:, 0], _points[:, 1]))
gdf["temperature"] = t2m
gdf["humidity"] = humidity
gdf['y'] = y
gdf["Time"] = time
gdf.crs = "EPSG:4326"  # Set the coordinate reference system

# %% Plot the response variable time series for one point
point_index = np.random.randint(0, n)  # Index of the point to plot

fig, ax = plt.subplots(figsize=(10, 6))
point_gdf = gdf.iloc[point_index::n]  # Extract the time series for the selected point
ax.plot(point_gdf["Time"], point_gdf["y"], marker="o", linestyle="-", color="blue", label="Response Variable")
ax.plot(point_gdf["Time"], point_gdf["temperature"], marker="x", linestyle="--", color="orange", label="Temperature")
ax.plot(point_gdf["Time"], point_gdf["humidity"], marker="s", linestyle="-.", color="green", label="Humidity")
ax.set_title(f"Time Series of Response Variable for Point {point_index}")
ax.set_xlabel("Time")
ax.set_ylabel("Response Variable (y)")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
plt.show()

# %% Build the lrssm and set the covariance function


model = lrssm(df=gdf, domain=[domain], verbose=True, backend="gpu")
model = model.setup(cov_fun=[cov_fun], domain_latent=[domain])

print(model)

# Set up the model cov. 
# print(model)

# %% Create the model parameters for the simulation
params = ModelParams(beta=[3], A=np.array([[1.5]]), s2e=[6], ks=[20], f=[0.7])
y_sim, x_sim, info, tdelta = model.sim(["1"], params=params, stats=True, verbose=True)

# %% Plot one response variable time series and state
x_sim_temp = info['sim_model'].H @ x_sim
point_index = 40 # np.random.randint(0, n)  # Index of the point to plot

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[point_index, :], marker="o", linestyle="-", color="blue", label="Simulated Response Variable")
# ax.plot(x_sim[point_index, :], marker="s", linestyle="-.", color="green", label="Simulated Latent State")
ax.plot(x_sim_temp[point_index, :], marker="x", linestyle="--", color="orange", label="Simulated Random Effect")
ax.set_title(f"Time Series of Simulated Response Variable and Latent State for Point {point_index}")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
plt.show()

# %% Plot the simulated response variable for one point
T = info['T'][0]
pt = np.array(info['points'][0])

times = np.arange(0, T, T//9)  # Select 9 time steps evenly spaced across the simulation
vmin, vmax = y_sim.min(), y_sim.max()  # or compute over the selected times

fig, ax = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)
for i, t in enumerate(times[:-1]):
    row, col = divmod(i, 3)
    xs = np.unique(pt[:, 0]) 
    ys = np.unique(pt[:, 1])
    grid = y_sim[:, t].reshape(xs.size, ys.size).T
    im = ax[row, col].imshow(
        grid,
        extent=(xs.min(), xs.max(), ys.min(), ys.max()),
        origin="lower",
        cmap="coolwarm",
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax
    )
    # ax[row, col].plot(*domain.boundary.xy, color="red")
    ax[row, col].set_title(f"Simulated $PM_{{10}}$ at Time Step {t+1}")

# single shared colorbar for all subplots
cbar = fig.colorbar(im, ax=ax)
plt.show()


# %% Estimate the model parameters from the simulated data

# 0) Create the geopandas dataframe with the simulated data
gdf["y_sim"] = y_sim.flatten(order='F')  # Flatten in column-major order to match the time series structure

# 1) Create the covariance matrix 
est_cov_fun = matern_spde([domain], latlon=False, nu=1, var=1, rescale=2)
est_cov_fun = est_cov_fun.setup(mesh_io)

# 2) Create the model
model = lrssm(
    df=gdf, 
    formulas=["y_sim ~ 1 + temperature"], 
    domain=[domain], 
    verbose=True)


# 3) Set up the model cov. 
model = model.setup(cov_fun=[est_cov_fun], domain_latent=[domain])
print(model)

# %% 4) fit the model 
 
opt = FitOptions()
opt.max_iter = 10
opt.tol_relat = 1e-5

results = model.fit(options=opt)
print(results)


