#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
@title: Example of estimating the parameters of a LRSSM from simulated data with random points in a convex domain (regular grid) and a Matern covariance function with nu=1. The model is fitted using the maximum likelihood estimation (MLE) method. 
The example also includes a summary of the variance of the simulated data and the theoretical variance 
of the model, as well as plots of the simulated response variable and the latent state for one point, 
and the simulated response variable for multiple time steps.
"""

# %% Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import polygon
import pygmsh
import gmsh
import geopandas as gpd
from datetime import date, timedelta
import jax
import geossm

if geossm.__file__:
    # import lrssm
    from geossm.covmodel import spdeAppoxCov as matern_spde
    from geossm.stmodel import ModelParams
    from geossm.stmodel import LRStateSpaceModel as lrssm
    from geossm.stmodel import FitOptions



# %% Simulate random point in a convex space (regular grind) and plot them

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

def build_dataframe(n, T=50):
    # %% Build the geopandas dataframe
    points = np.random.uniform(0, 1, (n, 2))

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

    return gdf, points

# %% Build the lrssm and set the covariance function

# Baseline dimensions, held fixed while sweeping one dimension at a time
n_base, T_base = 200, 1000

sweeps = {
    "T": {"values": [100, 200, 500, 1000, 2000, 5000], "fixed": {"n": n_base}},
    "n": {"values": [50, 100, 200, 500, 1000], "fixed": {"T": T_base}},
}

# Only benchmark backends that are actually available on this machine.
# jax.devices("gpu") raises a RuntimeError (rather than returning an empty
# list) when no GPU platform is registered, so this must be caught.
try:
    has_gpu = bool(jax.devices("gpu"))
except RuntimeError:
    has_gpu = False

backends = ["cpu", "gpu"] if has_gpu else ["cpu"]
if not has_gpu:
    print("No GPU device found: skipping the GPU backend in the timing comparison.")

# set the domain as a square [0, 1] x [0, 1]
domain = polygon.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

records = []
for sweep_name, cfg in sweeps.items():
    for value in cfg["values"]:
        dims = dict(cfg["fixed"])
        dims[sweep_name] = value

        # create the dataframe with the specified dimensions
        gdf, points = build_dataframe(dims["n"], T=dims["T"])

        # Create the mesh 
        mesh_io = buildMesh(domain, 0.25, points, lc_buffer=1)
        print(mesh_io)

        params = ModelParams(beta=[3], A=np.array([[1.5]]), s2e=[6], ks=[5], f=[0.7])

        # Create the covariance function used to simulate the "truth"
        cov_fun = matern_spde([domain], latlon=False, nu=1, var=1, rescale=4)
        cov_fun = cov_fun.setup(mesh_io)

        # Build a dedicated model to simulate the "truth" data and set up its covariance
        sim_model = lrssm(df=gdf, formulas=["1"], domain=[domain], verbose=False)
        sim_model = sim_model.setup(cov_fun=[cov_fun], domain_latent=[domain])

        # Print the var. statistics (verbose = True)
        y_sim, x_sim, info, tdelta = sim_model.sim(params=params, stats=True, verbose=True)

        # 0) Create the geopandas dataframe with the simulated data
        gdf["y_sim"] = y_sim.flatten(order='F')  # Flatten in column-major order to match the time series structure

        # 1) Create the covariance matrix 
        est_cov_fun = matern_spde([domain], latlon=False, nu=1, var=1, rescale=2)
        est_cov_fun = est_cov_fun.setup(mesh_io)

        # 2) Create the model
        for backend in backends:
            model = lrssm(
                df=gdf, 
                formulas=["y_sim ~ 1"], 
                domain=[domain], 
                verbose=False, backend=backend)


            # 3) Set up the model cov. 
            model = model.setup(cov_fun=[est_cov_fun], domain_latent=[domain])
            print(model)

            # 4) fit the model
            opt = FitOptions()
            opt.max_iter = 50
            opt.tol_relat = 1e-5

            results = model.fit(options=opt)


            records.append(
                {
                    "sweep": sweep_name,
                    "value": value,
                    "backend": backend,
                    "tsim_estep": results.runtime_tot_estep,
                    "tsim_mstep": results.runtime_tot_mstep,
                    }
            )
            print(
                f"[sweep={sweep_name:<1s}] backend={backend:>3s} "
                f"{sweep_name}={value:<5d} tsim_estep(s)={results.runtime_tot_estep:.4f} tsim_mstep(s)={results.runtime_tot_mstep:.4f}"
            )

timing_df = pd.DataFrame.from_records(records)

# %% 
