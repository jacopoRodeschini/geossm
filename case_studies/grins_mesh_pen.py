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


# %% 

mesh, hull, info = buildMesh2d(points, n_vertices=len(points), domain=list(italy_union.geoms),
                               eps=0.1,
                               gradation=0.5,
                               bandwidth=0.1,
                               anchor_spacing=0.1,
                               return_info=True)

print("Inner vertices {}".format(info['n_inside']))
print("Anchor ratio {}".format(len(info['anchors'])/ len(points)))
print("min angle {}".format(info['min_angle']))
print("mean distance {}".format(np.mean(info['point_distance'])))


cov_fun = matern_spde(latlon=False, nu=1, var=1, rescale=4)
cov_fun = cov_fun.setup(mesh, domain=list(italy_union.geoms))

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
cov_fun.fem_solver.plot_mesh(ax=ax)
ax.plot(points[:, 0], points[:, 1], "x", markersize=3)
  