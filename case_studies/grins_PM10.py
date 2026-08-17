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
import pandas as pd
from shapely.geometry import polygon
import geopandas as gpd
from datetime import date, timedelta
import jax
import jax.numpy as jnp
import geossm
from shapely.geometry import Polygon, MultiPolygon, MultiLineString
from shapely import ops
import matplotlib.pyplot as plt

if geossm.__file__:
    # import lrssm
    from geossm.covmodel import spdeAppoxCov as matern_spde
    from geossm.stmodel import ModelParams
    from geossm.stmodel import LRStateSpaceModel as lrssm
    from geossm.stmodel import FitOptions
    from geossm.covmodel import buildMesh2d
    import geossm.datasets as df
    from geossm import DesignMatricesBuilder as grid

# assert the presence of a GPU device
try:
    has_gpu = bool(jax.devices("gpu"))
except RuntimeError:
    has_gpu = False

assert has_gpu, "No GPU device found: this example requires a GPU device to run."

# %% Utils function

def _fit_safely(model, opt, label):
    """
    Run model.fit(), skipping the current sweep point instead of crashing
    when the config doesn't fit in memory.

    Only helps for a GPU RESOURCE_EXHAUSTED: XLA raises that as a normal
    Python exception, so it can be caught and the process survives. A CPU
    (host RAM) OOM is a different beast - the Linux OOM-killer sends SIGKILL
    from outside the process, which nothing in Python can catch, so a config
    that's too large for the CPU backend still takes the whole run down.

    jax.clear_caches() is called after a hit to reduce (not guarantee) the
    chance that a fragmented/dirty allocator state causes a smaller, later
    config to spuriously fail too. If OOMs keep bleeding into later configs
    despite this, the robust fix is to isolate each config in its own
    subprocess so a blown-out run can't poison the ones after it.
    """
    try:
        return model.fit(options=opt)
    except jax.errors.JaxRuntimeError as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            print(f"[SKIP] {label}: out of memory, skipping this configuration.")
            jax.clear_caches()
            return None
        raise

# %% one simulation comparison 

print("Start the case study 1")
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
# thr = 0.8
# stpm10_train = stpm10[: int(len(stpm10) * thr)]
# stpm10_test = stpm10[int(len(stpm10) * thr) :] 

# # extract the PM10 sub dataset
gdfpm10_train = grins[grins["AirQualityStation"].isin(stpm10)].copy()
# gdfpm10_train = grins[grins["AirQualityStation"].isin(stpm10_train)].copy()
# gdfpm10_test = grins[grins["AirQualityStation"].isin(stpm10_test)].copy()



# Create the shared mesh (both backends)
points = np.array([geom.coords[0] for geom in gdfpm10_train.geometry.unique()])


mesh_io, buffer = buildMesh2d(points, offset=2,lowrank=0.75, density_neighbors=3, 
                    min_edge=0.3, max_edge=2)

print(mesh_io)


# Create the covariance function used to simulate the "truth"
cov_fun = matern_spde(list(italy_union.geoms), latlon=False, nu=1, var=1, rescale=4)
cov_fun = cov_fun.setup(mesh_io)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
cov_fun.fem_solver.plot_mesh(ax=ax)
ax.plot(points[:, 0], points[:, 1], "x", markersize=3)
ax.set_title("Mesh and points")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# %% 
# Build a dedicated model to simulate the "truth" data and set up its covariance
formula = "AQ_mean_PM10 ~ 1 + CL_t2m + CL_windspeed + CL_rh"
sim_model = lrssm(df=gdfpm10_train, formulas=[formula], domain=[italy_union], verbose=True)
sim_model = sim_model.setup(cov_fun=[cov_fun])

# Print the var. statistics (verbose = True)
params = ModelParams(beta=[3, 1, 0.5, 4], A=np.array([[3]]), s2e=[1], ks=[2], f=[0.85]) 
y_sim, x_sim, info, tdelta = sim_model.sim(params=params, stats=True, verbose=True)

# 0) Create the geopandas dataframe with the simulated data
stpm10["y_sim"] = y_sim.flatten(order='F')  # Flatten in column-major order to match the time series structure

# 1) Create the covariance matrix 
est_cov_fun = matern_spde([italy_union], latlon=False, nu=1, var=1, rescale=2)
est_cov_fun = est_cov_fun.setup(mesh_io)

# 2) Create the model
backends = ["cpu", "gpu"] if has_gpu else ["cpu"]


opt = FitOptions()
opt.max_iter = 50
opt.tol_relat = 1e-4

records = []
for backend in backends:
    model = lrssm(
        df=gdfpm10_train, 
        formulas=["y_sim ~ 1 + CL_t2m + CL_windspeed + CL_rh"], 
        domain=[italy_union], 
        verbose=False, backend=backend, dtype=jnp.float64)


    # 3) Set up the model cov. 
    model = model.setup(cov_fun=[est_cov_fun])
    # print(model)

    # 4) fit the model
    results = _fit_safely(model, opt, label=f"backend={backend}")
    if results is None:
        continue

    records.append(
        {
            "backend": backend,
            "tsim_estep": results.runtime_tot_estep,
            "tsim_mstep": results.runtime_tot_mstep,
            "mse": results.mse(), 
            'llf': results.llf,
            'params': results.params,
    }
    )

    print(
        f"backend={'gpu':>3s}, mse: {results.mse():.4f} - "
        f"tsim_estep(s)={results.runtime_tot_estep:.4f} tsim_mstep(s)={results.runtime_tot_mstep:.4f}"
    )

# print the timing results as a dataframe
print(records)

# %% 
from pathlib import Path

# create the absolute path where to save the timing results
folderpath = Path().resolve() / "test" / "case_study"
path = folderpath / "cs_1_timing_results.pkl"


timing_df = pd.DataFrame.from_records(records)
timing_df['tot'] = timing_df['tsim_estep'] + timing_df['tsim_mstep']
print(timing_df)

timing_df.to_pickle("cs_1_timing_results.pkl")


# %% Plot the results 
# from matplotlib import pyplot as plt


# # read the pickle file with the timing results
# timing_df = pd.read_pickle("timing_results.pkl")


# fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# backends = timing_df["backend"].unique()
# for ax, sweep_name in zip(axes, ["T", "n"]):
#     sub = timing_df[timing_df["sweep"] == sweep_name]
#     for backend in backends:
#         s = sub[sub["backend"] == backend].sort_values("value")
#         ax.errorbar(
#             s["value"],
#             s["tot"],
#             # yerr=s["t est"],
#             marker="o",
#             capsize=3,
#             label=backend.upper(),
#         )
#     ax.set_xlabel(sweep_name)
#     ax.set_ylabel("Simulation time (s)")
#     ax.set_title(f"Runtime vs {sweep_name}")
#     ax.set_yscale("log")
#     ax.grid(True, alpha=0.3)
#     ax.legend()

# fig.suptitle(f"CPU vs GPU estimation time")
# fig.tight_layout()
# # plt.show()

