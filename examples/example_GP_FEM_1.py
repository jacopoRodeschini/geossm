#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
@title: Estimate the Matern SPDE covariance function using Monte Carlo simulations.
"""

# %% [Imports]

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from shapely.geometry import Polygon
from gstools.covmodel import Matern
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from tqdm import tqdm
from geossm.covmodel import buildMesh2d, SPDEApproxCov

# %% Create a convex domain [0,1]^2
n = 400
domain = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

# generate points inside that domain
points = np.random.uniform(0, 1, size=(n, 2))

# plot the generate poinsts
fix, ax = plt.subplots()
ax.plot(points[:, 0], points[:, 1], "x", label="Random Points")
ax.plot(domain.exterior.xy[0], domain.exterior.xy[1], "r-", label="Pacman Boundary")
ax.set_title("Random Points in Pacman Domain")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.legend()
plt.show()

# %% Generate the observed dataset

theta = 0.5  # range
rescale = np.sqrt(8 * 1) / theta

covf = Matern(dim=2, var=1, rescale=rescale, nugget=0, latlon=False, nu=1)

hdist = cdist(points, points)

Sigma = covf.covariance(hdist)
Sigma_chol = np.linalg.cholesky(Sigma + 1e-10 * np.eye(len(points)))

s2 = 0.1
yobs = Sigma_chol @ np.random.normal(size=len(points)) + np.random.normal(
    0, s2, size=len(points)
)

# plot the observed data
fix, ax = plt.subplots()
s = ax.scatter(points[:, 0], points[:, 1], c=yobs, cmap="viridis")
ax.plot(domain.exterior.xy[0], domain.exterior.xy[1], "r-", label="Pacman Boundary")
ax.set_title("Simulated Observations in Pacman Domain")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
plt.colorbar(s, label="Observed Value")
plt.show()


# %% [Utils] Optimisation function (raw Python implementation)
def minfun(par, H, spdeCov, yObs, s2):

    n = len(yObs)
    # par comes in as a length-1 array from scipy's optimizer; gstools'
    # rescale setter requires a plain (0-d) Python scalar
    rescale = float(np.exp(par).item())

    Q = spdeCov.precision(rescale=rescale)

    inx = spdeCov.fem_solver.inner
    Q_11 = Q[inx, :][:, inx]
    Q_12 = Q[inx, :][:, ~inx]
    Q_22 = Q[~inx, :][:, ~inx]

    # Covarianza marginale
    Q_mar = Q_11 - Q_12 @ np.linalg.inv(Q_22.toarray()) @ Q_12.T
    Sigma_mar = np.linalg.inv(Q_mar)

    # use SWM formula
    M = H @ Sigma_mar @ H.T + s2 * np.eye(n)
    invM = np.linalg.inv(M)

    # Calculate the log-pdf using the multivariate normal formula
    logdet = np.linalg.slogdet(M)[1]
    logpdf = logdet + yObs @ invM @ yObs.T

    #  print(rescale, logpdf)

    return 0.5 * logpdf


# %% Create the approximate covariance function

# create the mesh: help(buildMesh2d) to see the parameters of the buildMesh2d function
mesh_io, poly, info = buildMesh2d(points, lowrank=0.5, 
    domain=domain, offset=0.1, return_info=True)

print(f"min angle: {info['min_angle']}")
print(f"min d: {info['h_min_D']}")

# Create the covariance function
cov_matern = SPDEApproxCov(latlon=False, nu=1, var=1, rescale=1)

# set the mesh
cov_matern = cov_matern.setup(mesh_io, domain=domain)
print(cov_matern)

# plot the mesh
fix, ax = plt.subplots()
s = ax.scatter(points[:, 0], points[:, 1], c=yobs, cmap="viridis")
ax.plot(domain.exterior.xy[0], domain.exterior.xy[1], "r-", label="Pacman Boundary")
cov_matern.fem_solver.plot_mesh(ax=ax)
ax.set_title("Simulated Observations in Pacman Domain")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
plt.colorbar(s, label="Observed Value")
plt.show()

# Compute the basis function on the obsereved points
count, notfind, H = cov_matern.fem_solver.eval_basis(points)
# Cut the H on the inner nodes of the mesh (the outer nodes are fixed to 0)
H = H[:, cov_matern.fem_solver.inner]

# minimise the - log_likelihood
# %% Estimate single process

res = sc.optimize.minimize(
    minfun, args=(H, cov_matern, yobs, 0.1), x0=np.log(5), method="Nelder-Mead"
)
print(np.exp(res.x))

# %% The estimate task

def estimate(n=100):

    # Create the random points within the domain
    points = np.random.uniform(0, 1, size=(n, 2))

    covf = Matern(dim=2, var=1, rescale=rescale, nugget=0, latlon=False, nu=1)
    hdist = cdist(points, points)

    Sigma = covf.covariance(hdist)
    Sigma_chol = np.linalg.cholesky(Sigma + 1e-10 * np.eye(len(points)))

    s2 = 0.1
    yobs = Sigma_chol @ np.random.normal(size=len(points)) + np.random.normal(
        0, s2, size=len(points)
    )

    # create the mesh
    mesh_io, _ = buildMesh2d(points, lowrank=0.5, 
    domain=domain, offset=0.1, return_info=False)

    # Create the covariance function
    cov_matern = SPDEApproxCov(latlon=False, nu=1, var=1, rescale=1)
    cov_matern = cov_matern.setup(mesh_io, domain=domain)

    # Compute the basis function on the observed points (while the mesh is fixed)
    _, _, H = cov_matern.fem_solver.eval_basis(points)
    H = H[:, cov_matern.fem_solver.inner]

    res = sc.optimize.minimize(
        minfun, args=(H, cov_matern, yobs, s2), x0=np.log(0.3), method="Nelder-Mead"
    )

    return np.exp(res.x)


# %% Run The MC simulation
boot = 10

# Parallel execution [faster]
results = Parallel(n_jobs=-1, backend="loky")(
    delayed(estimate)(n=100) for _ in tqdm(range(boot))
)

bias = rescale - np.mean(results)
rmse = np.sqrt(np.mean((rescale - results) ** 2))
