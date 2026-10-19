#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
@title: Example of estimating the parameters of a LRSSM for study the Air quality in Italy (PM10) using the GRINS dataset. 
The example is run on a GPU and CPU device.
"""

# %% Import the necessary libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import jax
import jax.numpy as jnp
import geossm
from shapely.geometry import Polygon, MultiPolygon
from shapely import ops
import matplotlib.pyplot as plt
from pathlib import Path


if geossm.__file__:
    from geossm.covmodel import spdeAppoxCov as matern_spde
    from geossm.stmodel import LRStateSpaceModel as lrssm
    from geossm.stmodel import FitOptions
    from geossm.covmodel import buildMesh2d_density
    import geossm.datasets as df


# Assert the presence of a GPU device
try:
    has_gpu = bool(jax.devices("gpu"))
except RuntimeError:
    has_gpu = False

assert has_gpu, "No GPU device found: this example requires a GPU device to run."


    
print("Start the first case study")
print("GRINS PM10 data simulation and LRSSM model fitting with CPU and GPU backends.")

# create the dataframe with the specified dimensions
grins, _ = df.load_dataset("aqclim_points")

# %% Import italian shapefile
shapepath = '/home/jrodeschini/geopy_casestudy/geopy_grins/EApplication/Ita_region/Reg01012024_g_WGS84.shp'

sh_italy = gpd.read_file(shapepath)
sh_italy.to_crs(4326, inplace=True)

def getLargestPoly(geometry):
    if isinstance(geometry, Polygon):
        return geometry
    elif isinstance(geometry, MultiPolygon):
        # Return the polygon with the largest area
        # (itarable, key = scalar ordering function)
        return max(geometry.geoms, key=lambda p: p.area)
    else:
        return geometry  # Just in case

# remove minly island
ct_italy = sh_italy.copy()

# get the largest polygon
ct_italy['geometry'] = ct_italy['geometry'].apply(
    lambda geo: getLargestPoly(geo))

# ct_italy.plot()
italy_union = ops.unary_union(ct_italy.geometry)

# %% 
# create the convex hull of the union of the polygons
# boundary = italy_union.convex_hull

stpm10 = grins[np.isnan(grins["AQ_mean_PM10"]) == False]["AirQualityStation"].unique()

# extract the PM10 sub dataset
gdfpm10_train = grins[grins["AirQualityStation"].isin(stpm10)].copy()

# Create the shared mesh (both backends)
points = np.array([geom.coords[0] for geom in gdfpm10_train.geometry.unique()])

# %% Create the mesh and the buffer for the SPDE approximation of the covariance function
"""
@points: array of shape (n_points, 2) with the coordinates of the points to create the mesh
@offset: float, the offset to create the buffer around the points
@lowrank: float, the low rank approximation of the covariance function
@density_neighbors: int, the number of neighbors to use for the density estimation
@min_edge: float, the minimum edge length of the mesh
@max_edge: float, the maximum edge length of the mesh
@return: mesh_io, buffer
"""
lowrank = [0.15, 0.25, 0.5, 0.75, 1.0]
# lowrank = [1]
cov_funcs = []
for lr in lowrank:

    if lr == 1.0:
        mesh_io, buffer = buildMesh2d_density(points, domain=list(italy_union.geoms), offset=2,
            lowrank=lr, max_edge=0.7, snap_to_points=False, relax_iters=1)
    else:
        mesh_io, buffer = buildMesh2d_density(points, domain=list(italy_union.geoms), offset=2,
            lowrank=lr, max_edge=0.7, snap_to_points=True)


    # Create the covariance function used to simulate the "truth"
    cov_fun = matern_spde(latlon=False, nu=1, var=1, rescale=4)
    cov_fun = cov_fun.setup(mesh_io, domain=list(italy_union.geoms))
    cov_funcs.append(cov_fun)

# Plot the generated mesh and the observed points
fig, ax = plt.subplots(1, len(cov_funcs), figsize=(6 * len(cov_funcs), 6))

if len(cov_funcs) == 1:
    ax = [ax]

for i, cov_fun in enumerate(cov_funcs):
    cov_fun.fem_solver.plot_mesh(ax=ax[i])
    ax[i].plot(points[:, 0], points[:, 1], "x", markersize=3)
    ax[i].set_title(f"Mesh and points (lowrank={lowrank[i]})")
    ax[i].set_xlabel("Longitude")
    ax[i].set_ylabel("Latitude")
    ax[i].set_aspect('equal')
    ax[i].grid(True)

for i, cov_fun in enumerate(cov_funcs):
    print(cov_fun)

# %% Remove outliers os the .999 percentile of the PM10 values
q999 = gdfpm10_train["AQ_mean_PM10"].quantile(0.999)
q001 = gdfpm10_train["AQ_mean_PM10"].quantile(0.001)

# Replace outlier values with nan
inx = (gdfpm10_train["AQ_mean_PM10"] > q999) | (gdfpm10_train["AQ_mean_PM10"] < q001)
gdfpm10_train.loc[inx, "AQ_mean_PM10"] = np.nan

# %% Standardise the covariates (CL_t2m, CL_windspeed, CL_rh) to have mean 0 and variance 1
fix, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].hist(np.sqrt(gdfpm10_train["AQ_mean_PM10"]), bins=30, alpha=0.5, label="AQ_mean_PM10")
ax[1].hist(gdfpm10_train["CL_t2m"], bins=30, alpha=0.5, label="CL_t2m")
ax[2].hist(gdfpm10_train["CL_windspeed"], bins=30, alpha=0.5, label="CL_windspeed")
ax[3].hist(gdfpm10_train["CL_rh"], bins=30, alpha=0.5, label="CL_rh")
ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()
ax[0].set_title("Histograms of the covariates before standardisation")
plt.show()


# %% 

# Define the formula used to define the fixed effects in the LRSSM model. 
# The formula is used to create the design matrices for the fixed effects.
formula = "AQ_mean_PM10 ~ 1 + CL_t2m + CL_windspeed + CL_rh"

# Create the model
backends = ["cpu", "gpu"] if has_gpu else ["cpu"]

# Set the options for the fitting procedure
opt = FitOptions()
opt.max_iter = 50
opt.tol_relat = 1e-4
opt.verbose = True

# Store the results in a list of dictionaries, one for each backend
# lowrank = [0.25, 0.5, 0.75, 1.0]
records = {lr: {} for lr in lowrank}
for cov_fun, lr in zip(cov_funcs, lowrank):
    temp = {backend: {} for backend in backends}
    for backend in backends:

        # Create the LRSSM model with the specified backend and data
        """
        @df: dataframe with the data to fit the model
        @formulas: list of formulas to define the fixed effects in the model
        @domain: list of polygons defining the spatial domain of the model
        @verbose: boolean to print the model summary
        @backend: string to specify the backend to use for the fitting procedure (cpu or gpu)
        @dtype: data type to use for the fitting procedure (float32 or float64)
        """
        model = lrssm(
            df=gdfpm10_train, 
            formulas=[formula], 
            domain=[italy_union], 
            verbose=False, backend=backend, dtype=jnp.float64)


        # Set up the model cov. 
        model = model.setup(cov_fun=[cov_fun])
        # print(model)

        # Fit the model using the EM algorithm with the specified options
        results = model.fit(options=opt)
        
        # Store the results in a dictionary and append it to the list of records
        temp[backend] = dict({
                "backend": backend,
                "lowrank": lr,
                "tsim_estep": results.runtime_tot_estep,
                "tsim_mstep": results.runtime_tot_mstep,
                "ttot": results.runtime_tot_estep + results.runtime_tot_mstep,
                "mse": results.mse(), 
                'llf': results.llf,
                'params': results.params,
        })

        # Print the results of the fitting procedure for the current backend
        print(
            f"backend={backend:>3s}, mse: {results.mse():.4f} - "
            f"tsim_estep(s)={results.runtime_tot_estep:.4f} tsim_mstep(s)={results.runtime_tot_mstep:.4f}"
        )

    records[lr] = temp

# Convert the list of dict to a pandas dataframe and add a column with the total simulation time
timing_df = pd.DataFrame([item for sublist in records.values() for item in sublist.values()])

# %% 
# Save the results in a pickle file for later analysis and plotting
dtype = "float64"
folderpath = Path().resolve() / "case_studies"
path = folderpath / f"grins_PM10_timing_results_{dtype}.pkl"

timing_df.to_pickle(path)
