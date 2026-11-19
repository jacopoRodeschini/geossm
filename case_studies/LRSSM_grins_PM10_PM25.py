#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
@title: Example of estimating the parameters of a LRSSM for study the Air quality in Italy (PM10) using the GRINS dataset. 
The example is run on a GPU and CPU device.
"""

# %% Import the necessary libraries
from examples.example_LRSSM_prediction import index
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

msg = """
JAX configuration (see the user guide page "JAX configuration"). These are
process-wide and must be set before JAX starts, i.e. before `import jax`.
This study compares CPU and GPU, so none is set here; uncomment as needed:

CPU only: JAX does not start its GPU backend (0 GB of GPU memory used).
Without it, JAX reserves GPU memory even for backend="cpu" models.
os.environ["JAX_PLATFORMS"] = "cpu"

GPU: allocate memory on demand instead of reserving 75 % of it upfront
(Agrimonia, float32: 1.3 GB after fitting, 9.0 GB peak with standard
errors, same speed), or cap the reserved fraction instead.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"

Reuse compiled kernels across runs (the first fit is mostly compilation).
os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.expanduser("~/.cache/jax")
"""
print(msg)

import numpy as np
import jax
import jax.numpy as jnp

# The models below use float64, which needs JAX's 64-bit mode enabled before
# any JAX array is created (geossm does not switch it on itself).
jax.config.update("jax_enable_x64", True)

from shapely import ops
import matplotlib.pyplot as plt

from geossm.covmodel import SPDEApproxCov as matern_spde
from geossm.stmodel import LRStateSpaceModel as lrssm
from geossm.stmodel import FitOptions
from geossm.covmodel import buildMesh2d
import geossm.datasets as df


# Assert the presence of a GPU device
try:
    has_gpu = bool(jax.devices("gpu"))
except RuntimeError:
    has_gpu = False

assert has_gpu, "No GPU device found: this example requires a GPU device to run."


print("="*50)    
print("Start the first case study")
print("GRINS PM10 data simulation and LRSSM model fitting with CPU and GPU backends.")

#%%  create the dataframe with the specified dimensions
print("="*50)
print("Loading the GRINS dataset...")
grins, shape = df.load_dataset("aqclim_points")

# Project the dataset and the shape into metric distances
grins = grins.to_crs(epsg=32632)
shape = shape.to_crs(epsg=32632)

# %% fix the shape into a list of poligons

from shapely.geometry import MultiLineString
from shapely.ops import polygonize

shape["geometry"] = shape.geometry.apply(
    lambda geom: MultiLineString(
        sorted(geom.geoms, key=lambda g: g.length, reverse=True)[:3]
    )
    if geom.geom_type == "MultiLineString"
    else geom
)

# create the union of the polygons in the shape dataframe
italy_union = ops.unary_union(shape.geometry) # you can also simply the geometry using the simplify method, but this will not preserve the topology of the polygons

# List of polygons in the union of the polygons in the shape dataframe
domain = list(polygonize(italy_union))

# %% Create a geopandas dataframe from training

stpm10 = grins[np.isnan(grins["AQ_mean_PM10"]) == False]["AirQualityStation"].unique()
stpm25 = grins[np.isnan(grins["AQ_mean_PM2.5"]) == False]["AirQualityStation"].unique()

# extract the PM10 sub dataset
gdfpm10_train = grins[grins["AirQualityStation"].isin(stpm10)].copy()
gdfpm25_train = grins[grins["AirQualityStation"].isin(stpm25)].copy()

# Create the shared mesh
points = [np.array([geom.coords[0] for geom in gdfpm10_train.geometry.unique()]),
          np.array([geom.coords[0] for geom in gdfpm25_train.geometry.unique()])]

# concatenate the two dataframes into a single dataframe
gdf_train = grins[grins["AirQualityStation"].isin(np.concatenate([stpm10, stpm25]))].copy()

# rename the columns of the dataframe to avoid issues with the formula parser
gdf_train = gdf_train.rename(columns={"AQ_mean_PM2.5": "AQ_mean_PM25"})


print("="*50)
print(f"#stations PM10: {len(points[0])}, #stations PM2.5: {len(points[1])}")

# %% Adding the season variable to the dataframe
def getseason(date):
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

gdf_train["season"] = gdf_train["time"].apply(getseason)

# %% Create the mesh and the buffer for the SPDE approximation of the covariance function
lowrank = [0.5]
names = ["PM10", "PM2.5"]
cov_funcs = {(lr, n): None for lr in lowrank for n in names}
for lr in lowrank:
    for n, pts in zip(names, points):
        mesh_io, hull, info = buildMesh2d(pts, lowrank=lr, domain=domain,
                                          eps=0.1, 
                                          gradation=0.5,
                                          bandwidth=0.1,
                                          anchor_spacing=0.3, 
                                          return_info=True)

        # Create the covariance function used to simulate the "truth"
        cov_fun = matern_spde(latlon=False, nu=1, var=1, rescale=4)
        cov_fun = cov_fun.setup(mesh_io, domain=domain)
        cov_funcs[(lr, n)] = cov_fun

# %% Plot the generated mesh and the observed points
fig, ax = plt.subplots(1, len(cov_funcs), figsize=(6 * len(cov_funcs), 6))

if len(cov_funcs) == 1:
    ax = [ax]

for i, ((lr, n), cov_fun) in enumerate(cov_funcs.items()):
    cov_fun.fem_solver.plot_mesh(ax=ax[i])
    ax[i].plot(points[i][:, 0], points[i][:, 1], "x", markersize=3)
    ax[i].set_title(f"{n}: mesh and points (lowrank={lr})")
    ax[i].set_xlabel("Longitude")
    ax[i].set_ylabel("Latitude")
    ax[i].set_aspect('equal')
    ax[i].grid(True)

for i, ((lr, n), cov_fun) in enumerate(cov_funcs.items()):
    print(cov_fun)

# %% Remove outliers os the .999 percentile of the PM10 values
pm10_q999 = gdfpm10_train["AQ_mean_PM10"].quantile(0.999)
pm10_q001 = gdfpm10_train["AQ_mean_PM10"].quantile(0.001)

pm25_q999 = gdfpm25_train["AQ_mean_PM2.5"].quantile(0.999)
pm25_q001 = gdfpm25_train["AQ_mean_PM2.5"].quantile(0.001)

# Replace outlier values with nan
inx = (gdf_train["AQ_mean_PM10"] > pm10_q999) | (gdf_train["AQ_mean_PM10"] < pm10_q001)
gdf_train.loc[inx, "AQ_mean_PM10"] = np.nan

inx = (gdf_train["AQ_mean_PM25"] > pm25_q999) | (gdf_train["AQ_mean_PM25"] < pm25_q001)
gdf_train.loc[inx, "AQ_mean_PM25"] = np.nan

# %% Define the formula used to define the fixed effects in the LRSSM model. 

# The formula is used to create the design matrices for the fixed effects.
formula_pm10 = "np.log(AQ_mean_PM10) ~ 1 + season + standardize(CL_lai_hv) + standardize(CL_t2m) + standardize(CL_windspeed) + standardize(CL_rh) + standardize(CL_blh)"
formula_pm25 = "np.log(AQ_mean_PM25) ~ 1 + season + standardize(CL_lai_hv) + standardize(CL_t2m) + standardize(CL_windspeed) + standardize(CL_rh) + standardize(CL_blh)"

# Create the model
backend = ["gpu"] if has_gpu else ["cpu"]

# Create the LRSSM model with the specified backend and data
domain = ops.unary_union(domain)
model = lrssm(
    df=gdf_train, 
    formulas=[formula_pm10, formula_pm25], 
    domain=[domain, domain], 
    verbose=True, backend=backend[0], dtype=jnp.float64)


# Set up the model cov. 
model = model.setup(cov_fun=[cov_funcs[(0.5, n)] for n in names])
print(model)

# Set the options for the fitting procedure
opt = FitOptions(max_iter=50, tol_relat=1e-5, verbose=True)

# Fit the model using the EM algorithm with the specified options
results = model.fit(options=opt)

# Print the results of the fitting procedure for the current backend
print(
    f"backend={backend[0]:>3s}, mse: {results.mse():.4f} - "
    f"tsim_estep(s)={results.runtime_tot_estep:.4f} tsim_mstep(s)={results.runtime_tot_mstep:.4f}"
)

# %% Compute the model standard error
results.compute_cov_params()
print(results)

# %% back transform the results to the original scale 
results = results.back_transform(g_invs=[jnp.exp, jnp.exp])
print(results)

# Compute the MSE and RMSE in back transformed scale
y_obs = results.y_obs_back_list
y_hat = results.y_hat_back_list

mse = [jnp.nanmean((y_obs[i] - y_hat[i])**2) for i in range(len(y_obs))]
rmse = [jnp.sqrt(mse[i]) for i in range(len(mse))]

# print
print("="*50)
print("Back transformed results:")
for i, n in enumerate(names):
    print(f"{n}: mse={mse[i]:.4f}, rmse={rmse[i]:.4f}")
