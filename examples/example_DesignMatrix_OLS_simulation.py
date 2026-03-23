#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""
# %% Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from geossm.data_preparation import DesignMatricesBuilder
import geopandas as gpd
from datetime import date, timedelta

# %% Simulate random point in a convex space

# set the domain as a circle with radius 10
center = (0, 0)
radius = 1
circle = Point(center).buffer(radius)

points = np.random.uniform(-radius, radius, (100, 2))

# take the mask of the points that are inside the circle
mask = np.array([circle.contains(Point(p)) for p in points])
points = points[mask]

# plot the points and the domain
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], color="blue", label="Random Points")
plt.plot(*circle.boundary.xy, color="red", label="Domain Boundary")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Random Points in a Circular Domain")
plt.legend()
plt.axis("equal")
plt.show()

# %% Create the spatio-temporal covariates (e.g. temperature and humidity)
n = points.shape[0]
T = 20

# repeat the points T times for the temporal dimension
_points = np.tile(points, (T, 1, 1)).reshape(-1, 2) 

d = np.linspace(0, 2 * np.pi, n)
t2m = np.sin(d).reshape(-1, 1) + np.random.normal(0, 1, size=n * T).reshape(n, T)
t2m = t2m.reshape(-1, )

humidity = 1 / 2 * np.sin(d).reshape(-1, 1) + np.random.normal(0, 1, size=n * T).reshape(n, T)
humidity = humidity.reshape(-1, )

tstart = date(2020, 1, 1)
time = np.sort(np.tile([tstart + timedelta(days=d) for d in range(T)], n))

# create the response variable as a linear combination of the covariates plus some noise
beta = [10, 0.5, 0.3]  # coefficients for the covariates
s2e = 0.5  # variance of the noise
y = beta[0] + beta[1] * t2m + beta[2] * humidity + np.random.normal(0, np.sqrt(s2e), size=n * T).reshape(n, T).reshape(-1, )


# %% Create a GeoDataFrame from the points

gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(_points[:, 0], _points[:, 1]))
gdf["temperature"] = t2m
gdf["humidity"] = humidity
gdf["y"] = y
gdf["Time"] = time
gdf.crs = "EPSG:4326"  # Set the coordinate reference system


# %% Create the spatio-temporal grid obgect

builder = DesignMatricesBuilder(gdf, "y ~ 1 + temperature + humidity", verbose=True)
dataset = builder.build(verbose=True)

Xbeta = np.array(dataset.X)
y = np.array(dataset.y)
# %% OLS spatio-temporal regression 

# expected value
cond_xy = np.einsum('nbt,nt->b', Xbeta, y)
cond_xx = np.einsum('nbt,nct->bc', Xbeta, Xbeta)
beta = np.linalg.solve(cond_xx, cond_xy)  
s2e = np.mean((y - np.einsum('nbt,b->nt', Xbeta, beta))**2)

# standard error
std_beta = np.sqrt(np.diag(s2e * np.linalg.inv(cond_xx)))

# print the results

print("Estimated coefficients:", beta.flatten())  
print("Standard errors:", std_beta.flatten())
print("Confidence intervals:")
for i in range(len(beta)):
    print(f"Beta {i}: {beta[i] - 1.96 * std_beta[i]} - {beta[i] + 1.96 * std_beta[i]}")

