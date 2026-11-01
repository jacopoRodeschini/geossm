#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:38:28 2026

@author: jacopo
"""

# %% [import]
import matplotlib.pyplot as plt
import numpy as np
import geossm.datasets as df
from geossm import stmodel as st
from geossm import covmodel as cv 

# %% Load the agrimonia dataset

print(df.list_datasets())

agri, shape = df.load_dataset("agrimonia")
domain = list(shape["geometry"][0].geoms)[0].buffer(0.3)

# %% Build the lrssm model

model = st.LRStateSpaceModel(
    agri, ["np.sqrt(AQ_pm10) ~ 1 + WE_temp_2m"], verbose=True, domain=[domain])
print(model)

# %% Build the mesh for the AQ_pm10 observed variable

points = model.points[0]
mesh_io, poly = cv.buildMesh2d(points, lowrank=0.75)
print(mesh_io)

# plot the mesh (use the fem_solver utlities)
fem_solver = cv.FEMSolver(mesh_io, [domain], verbose=True)

# plot the mesh using the utilities
fig, ax = plt.subplots(figsize=(8, 8))
fem_solver.plot_mesh(ax=ax)

# %% Set up the lrssm model (univariate latent)

cov_fun = cv.SPDEApproxCov(latlon=True).setup(mesh_io, domain=[domain])
model = model.setup(cov_fun=[cov_fun])

# %% Estimate the Model (default estimation options)

results = model.fit()
print(results)  # results.summary()

# %% Plot the likelihood curve

fig, ax = plt.subplots()
ax.plot(-np.array(results.llf_path[1:]))
ax.set_yscale("log")
ax.set_xlabel("Iteration")
ax.set_ylabel("Log Likelihood")
ax.set_title("Negative Log-Likelihood Curve")
ax.grid()
plt.show()


# %% Estimate the model with other options

opt = st.FitOptions(max_iter =50, tol_relat=1e-4)
print(opt)

results = model.fit(options=opt)
print(results)


# %% Estimate the model with initial values

# %% More control on the model parameters

b0 = st.Param("beta", [29, -0.5], fixed=True)

# set the pars0
par0 = st.ModelParams(beta=b0, s2e=10)
print(par0)

# fit the model
restricted = model.fit(params0=par0, options=opt)

# %% Compare the likelihoods curve of the two models

fig, ax = plt.subplots()
ax.plot(-np.array(results.llf_path[1:]), label="unrestricted")
ax.plot(-np.array(restricted.llf_path[1:]), label="restricted")
ax.set_yscale("log")
ax.set_xlabel("Iteration")
ax.set_ylabel("Log Likelihood")
ax.set_title("Negative Log-Likelihood Curve")
ax.legend()
ax.grid()



