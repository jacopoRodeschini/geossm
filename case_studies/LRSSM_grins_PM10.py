#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
@title: Example of estimating the parameters of a LRSSM for study the Air quality in Italy (PM10) using the GRINS dataset. 
The example is run on a GPU and CPU device.
"""

# %% Import the necessary libraries
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
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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

import geossm
from shapely import ops
import matplotlib.pyplot as plt

if geossm.__file__:
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

# %% Project the dataset and the shape into metric distances
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

# extract the PM10 sub dataset
gdfpm10_train = grins[grins["AirQualityStation"].isin(stpm10)].copy()

# Create the shared mesh (both backends)
points = np.array([geom.coords[0] for geom in gdfpm10_train.geometry.unique()])


# %% Create the mesh and the buffer for the SPDE approximation of the covariance function
lowrank = [0.25, 0.5, 0.75]
cov_funcs = {lr: None for lr in lowrank}
for lr in lowrank:

    mesh_io, hull, info = buildMesh2d(points, n_vertices=int(lr*len(points)), domain=domain,
                                      eps=0.1, 
                                      gradation=0.5,
                                      bandwidth=0.1,
                                      anchor_spacing=0.3, 
                                      return_info=True)



    # Create the covariance function used to simulate the "truth"
    cov_fun = matern_spde(latlon=False, nu=1, var=1, rescale=4)
    cov_fun = cov_fun.setup(mesh_io, domain=domain)
    cov_funcs[lr] = cov_fun

# %% Plot the generated mesh and the observed points
fig, ax = plt.subplots(1, len(cov_funcs), figsize=(6 * len(cov_funcs), 6))

if len(cov_funcs) == 1:
    ax = [ax]

for i, (lr, cov_fun) in enumerate(cov_funcs.items()):
    cov_fun.fem_solver.plot_mesh(ax=ax[i])
    ax[i].plot(points[:, 0], points[:, 1], "x", markersize=3)
    ax[i].set_title(f"Mesh and points (lowrank={lr})")
    ax[i].set_xlabel("Longitude")
    ax[i].set_ylabel("Latitude")
    ax[i].set_aspect('equal')
    ax[i].grid(True)

for i, (lr, cov_fun) in enumerate(cov_funcs.items()):
    print(cov_fun)

# %% Remove outliers os the .999 percentile of the PM10 values
q999 = gdfpm10_train["AQ_mean_PM10"].quantile(0.999)
q001 = gdfpm10_train["AQ_mean_PM10"].quantile(0.001)

# Replace outlier values with nan
inx = (gdfpm10_train["AQ_mean_PM10"] > q999) | (gdfpm10_train["AQ_mean_PM10"] < q001)
gdfpm10_train.loc[inx, "AQ_mean_PM10"] = np.nan

# %% Define the formula used to define the fixed effects in the LRSSM model. 

# The formula is used to create the design matrices for the fixed effects.
# Formula without the standardisation
formula = "np.log(AQ_mean_PM10) ~ 1 + standardize(CL_t2m) + standardize(CL_windspeed) + standardize(CL_rh)"

# Create the model
backend = ["gpu"] if has_gpu else ["cpu"]

# Set the options for the fitting procedure
opt = FitOptions(max_iter=50, tol_relat=1e-3, verbose=True)
# Store the results in a list of dictionaries, one for each backend
# lowrank = [0.25, 0.5, 0.75, 1.0]

# Create the LRSSM model with the specified backend and data
union = ops.unary_union(domain)
results = {lr: {} for lr in lowrank}
for lr in lowrank:
    model = lrssm(
        df=gdfpm10_train, 
        formulas=[formula], 
        domain=[union], 
        verbose=False, backend=backend[0], dtype=jnp.float64)

    # Set up the model cov. 
    model = model.setup(cov_fun=[cov_funcs[lr]])
    print(model)

    # Fit the model using the EM algorithm with the specified options
    rs = model.fit(options=opt)

    # Print the results of the fitting procedure for the current backend
    print(
        f"lowrank={lr:.2f}, backend={backend[0]:>3s}, mse: {rs.mse():.4f} - "
        f"tsim_estep(s)={rs.runtime_tot_estep:.4f} tsim_mstep(s)={rs.runtime_tot_mstep:.4f}"
    )

    # %% add the resutls into the results dictionary
    results[lr] = dict({
        "backend": backend[0],
        "lowrank": float(lr),
        "tsim_estep": float(rs.runtime_tot_estep),
        "tsim_mstep": float(rs.runtime_tot_mstep),
        "ttot": float(rs.runtime_tot_estep + rs.runtime_tot_mstep),
        "mse": float(rs.mse()),
        "llf": float(rs.llf)
    })

# %% Make the results dictionary into a pandas dataframe
import pandas as pd
df_results = pd.DataFrame.from_dict(results, orient='index').sort_index(ascending=False)
