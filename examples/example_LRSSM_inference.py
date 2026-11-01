#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:38:28 2026

@author: jacopo
"""

# %% [imports]
import geossm.datasets as df
from geossm import stmodel as st
from geossm import covmodel as cv

# %% Load the agrimonia dataset

print(df.list_datasets())

agri, shape = df.load_dataset("agrimonia")
domain = list(shape["geometry"][0].geoms)[0].buffer(0.3)

# %% Build the lrssm model

model = st.LRStateSpaceModel(
    agri, ["np.sqrt(AQ_pm10) ~ 1 + WE_temp_2m"], 
    verbose=True, domain=[domain])

print(model)

# %% Build the mesh for the AQ_pm10 observed variable

points = model.points[0]
mesh_io, poly = cv.buildMesh2d(points, lowrank=0.75)
print(mesh_io)

# %% Set up the lrssm model (univiarte latent)

cov_fun = cv.SPDEApproxCov(latlon=True).setup(mesh_io, domain=[domain])
model = model.setup(cov_fun=[cov_fun])


# %% Estimate the Model (default estimation options)

opt = st.FitOptions(max_iter=50, tol_relat=1e-5)

results = model.fit(options=opt)

# Before using compute_cov_params(): bse/t/p/CI show as NaN placeholders
# print(results.summary(alpha=0.05))

# %% Compute the standard errors of the parameters

hessian = results.compute_cov_params()

# after compute_cov_params(): summary() now shows the real bse/t/p/CI
print(results.summary(alpha=0.05))

# %% Get the model parameter inference

# bse
params = results.bse
print(params.beta.bse)

# %% CI 
ci = results.conf_int(alpha=0.05)






